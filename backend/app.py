from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .prompt import extract_constraints_from_text
from .schemas import Employee, EmployeeRole, ScheduleRequest, ScheduleResult, ConstraintSet
from .solver import solve_schedule, solve_schedule_hourly
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
    # 使用前端傳入的選定人員清單；若未提供則回退到預設清單
    employees = request.employees if request.employees is not None else DEFAULT_EMPLOYEES
    if not employees:
        return ScheduleResult(
            assignments={},
            explanation="請至少選擇一位排班人員。",
            constraints=ConstraintSet(),
            conflict_reasons=["未選擇任何排班人員"],
            business_start_hour=request.business_start_hour,
            business_end_hour=request.business_end_hour,
        )

    start_hour = request.business_start_hour
    end_hour = request.business_end_hour

    # ── Step 1: LLM 解析自然語言約束 ──
    constraints = extract_constraints_from_text(request.text, employees, request.daily_staff_count)
    ai_understanding = constraints.explanation

    # ── Step 2: 可行性預檢 ──
    feasible, feasibility_errors = validate_feasibility(employees, constraints, request.text)
    if not feasible:
        explanation = "無法產生班表，原因如下：\n" + "\n".join(feasibility_errors)
        return ScheduleResult(
            assignments={},
            explanation=explanation,
            ai_understanding=ai_understanding,
            constraints=constraints,
            conflict_reasons=feasibility_errors,
            business_start_hour=start_hour,
            business_end_hour=end_hour,
        )

    # ── Step 3: 小時粒度 OR-Tools 求解器 ──
    result = solve_schedule_hourly(
        employees,
        constraints,
        start_hour=start_hour,
        end_hour=end_hour,
        daily_staff_count=request.daily_staff_count,
    )

    if result["status"] == "INFEASIBLE":
        conflicts = result.get("conflicts", [])
        return ScheduleResult(
            assignments={},
            explanation="系統判斷排班約束無法滿足：\n" + "\n".join(conflicts),
            ai_understanding=ai_understanding,
            constraints=constraints,
            conflict_reasons=conflicts,
            business_start_hour=start_hour,
            business_end_hour=end_hour,
        )

    employee_hours: dict = result.get("employee_hours", {})

    # 組建說明文字，列出各員工本週時數
    hours_parts = []
    for emp in employees:
        h = employee_hours.get(emp.id, 0)
        hours_parts.append(f"{emp.name or emp.id}：{h}h")
    explanation = (
        f"排班完成（{start_hour:02d}:00–{end_hour:02d}:00）。\n"
        f"各員工本週時數：{'、'.join(hours_parts)}"
    )

    return ScheduleResult(
        assignments=result["assignments"],
        explanation=explanation,
        ai_understanding=ai_understanding,
        constraints=constraints,
        daily_staff_count=request.daily_staff_count,
        conflict_reasons=None,
        business_start_hour=start_hour,
        business_end_hour=end_hour,
        employee_hours=employee_hours,
    )
