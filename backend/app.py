from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .prompt import extract_constraints_from_text
from .schemas import Employee, EmployeeRole, ScheduleRequest, ScheduleResult, ConstraintSet
from .solver import solve_schedule
from .validator import validate_feasibility, validate_assignments

app = FastAPI(title="AI Agent 智慧排班系統")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_EMPLOYEES = [
    Employee(id="A", name="A", role=EmployeeRole.regular),
    Employee(id="B", name="B", role=EmployeeRole.regular),
    Employee(id="C", name="C", role=EmployeeRole.regular),
    Employee(id="D", name="D", role=EmployeeRole.regular),
    Employee(id="E", name="E", role=EmployeeRole.new),
]


@app.post("/parse-constraints", response_model=ConstraintSet)
def parse_constraints(request: ScheduleRequest) -> ConstraintSet:
    employees = request.employees or DEFAULT_EMPLOYEES
    return extract_constraints_from_text(request.text, employees, request.daily_staff_count)


@app.post("/schedule", response_model=ScheduleResult)
def schedule(request: ScheduleRequest) -> ScheduleResult:
    employees = request.employees or DEFAULT_EMPLOYEES

    # ── Step 1: LLM extracts constraints (with RAG grounding + structured output) ──
    constraints = extract_constraints_from_text(request.text, employees, request.daily_staff_count)

    # ── Step 2 (pre-solve validator): are the constraints physically satisfiable? ──
    feasible, feasibility_errors = validate_feasibility(employees, constraints, request.text)
    if not feasible:
        explanation = "無法產生班表，原因如下：\n" + "\n".join(feasibility_errors)
        return ScheduleResult(
            assignments={},
            explanation=explanation,
            constraints=constraints,
            conflict_reasons=feasibility_errors,
        )

    # ── Step 3: OR-Tools solver ──
    result = solve_schedule(employees, constraints)
    if result["status"] == "INFEASIBLE":
        conflicts = result.get("conflicts", [])
        return ScheduleResult(
            assignments={},
            explanation="系統判斷排班約束無法滿足：\n" + "\n".join(conflicts),
            constraints=constraints,
            conflict_reasons=conflicts,
        )

    # ── Step 4 (post-solve validator): does the schedule respect all rules? ──
    violations = validate_assignments(result["assignments"], employees, constraints)

    assignment_texts = [
        f"{day}: {', '.join(names) if names else '（無人）'}"
        for day, names in result["assignments"].items()
    ]
    explanation = "排班已完成，請見下方班表。\n" + "\n".join(assignment_texts)
    if violations:
        explanation += "\n\n⚠ 驗證發現以下問題：\n" + "\n".join(violations)

    return ScheduleResult(
        assignments=result["assignments"],
        explanation=explanation,
        constraints=constraints,
        conflict_reasons=violations,
    )
