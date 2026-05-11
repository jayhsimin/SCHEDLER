from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from .prompt import extract_constraints_from_text
from .schemas import Employee, EmployeeRole, ScheduleRequest, ScheduleResult, ConstraintSet
from .solver import solve_schedule

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
    constraints = extract_constraints_from_text(request.text, employees, request.daily_staff_count)
    result = solve_schedule(employees, constraints)
    if result["status"] == "INFEASIBLE":
        explanation = (
            "系統判斷目前輸入的排班約束無法滿足。"
            "以下是可能的衝突原因：\n" + "\n".join(result.get("conflicts", []))
        )
        return ScheduleResult(
            assignments={},
            explanation=explanation,
            constraints=constraints,
            conflict_reasons=result.get("conflicts", []),
        )

    assignment_texts = [
        f"{day}: {', '.join(names) if names else 'None'}"
        for day, names in result["assignments"].items()
    ]
    explanation = (
        "排班已完成，請見下方日別排班結果。\n" + "\n".join(assignment_texts)
    )
    return ScheduleResult(
        assignments=result["assignments"],
        explanation=explanation,
        constraints=constraints,
        conflict_reasons=[],
    )
