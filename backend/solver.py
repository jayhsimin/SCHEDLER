from ortools.sat.python import cp_model
from typing import Dict, List, Optional, Tuple

from .schemas import ConstraintSet, DayMinimum, DayMaximum, MinShiftsPerEmployee, DayMinimumByRole, FairnessConstraint, Employee, Unavailability, Weekday, EmployeeRole

DAYS = [
    Weekday.monday,
    Weekday.tuesday,
    Weekday.wednesday,
    Weekday.thursday,
    Weekday.friday,
    Weekday.saturday,
    Weekday.sunday,
]

DAY_INDEX = {day: idx for idx, day in enumerate(DAYS)}


def expand_unavailability(unavail: Unavailability) -> List[Tuple[str, Weekday]]:
    if not unavail.end_day:
        return [(unavail.employee_id, unavail.start_day)]
    start = DAY_INDEX[unavail.start_day]
    end = DAY_INDEX[unavail.end_day]
    if end < start:
        end += 7
    result = []
    for index in range(start, end + 1):
        result.append((unavail.employee_id, DAYS[index % 7]))
    return result


def build_unavailability_map(constraints: ConstraintSet) -> Dict[Tuple[str, Weekday], Unavailability]:
    result: Dict[Tuple[str, Weekday], Unavailability] = {}
    for unavail in constraints.unavailabilities:
        for key in expand_unavailability(unavail):
            result[key] = unavail
    return result


def solve_schedule(staff_list: List[Employee], constraints: ConstraintSet) -> dict:
    model = cp_model.CpModel()
    work = {}
    for employee in staff_list:
        for day in DAYS:
            work[(employee.id, day)] = model.NewBoolVar(f"work_{employee.id}_{day}")

    unavail_map = build_unavailability_map(constraints)
    for employee in staff_list:
        for day in DAYS:
            if (employee.id, day) in unavail_map:
                model.Add(work[(employee.id, day)] == 0)

    for day_min in constraints.day_minimums:
        if day_min.day not in DAY_INDEX:
            continue
        model.Add(
            sum(work[(employee.id, day_min.day)] for employee in staff_list) >= day_min.min_staff
        )

    for day_max in constraints.day_maximums:
        if day_max.day is None:
            for day in DAYS:
                model.Add(
                    sum(work[(employee.id, day)] for employee in staff_list) <= day_max.max_staff
                )
        else:
            if day_max.day not in DAY_INDEX:
                continue
            model.Add(
                sum(work[(employee.id, day_max.day)] for employee in staff_list) <= day_max.max_staff
            )

    for emp_min in constraints.min_shifts_per_employee:
        model.Add(
            sum(work[(emp_min.employee_id, day)] for day in DAYS) >= emp_min.min_shifts
        )

    for role_min in constraints.day_minimums_by_role:
        role_employees = [e for e in staff_list if e.role == role_min.role]
        if role_min.day is None:
            for day in DAYS:
                model.Add(
                    sum(work[(e.id, day)] for e in role_employees) >= role_min.min_staff
                )
        else:
            if role_min.day not in DAY_INDEX:
                continue
            model.Add(
                sum(work[(e.id, role_min.day)] for e in role_employees) >= role_min.min_staff
            )

    for employee in staff_list:
        model.Add(
            sum(work[(employee.id, day)] for day in DAYS) <= employee.max_shifts_per_week
        )

    total_shifts = {}
    for employee in staff_list:
        total_shifts[employee.id] = model.NewIntVar(0, len(DAYS), f"total_shifts_{employee.id}")
        model.Add(total_shifts[employee.id] == sum(work[(employee.id, day)] for day in DAYS))

    objective_terms = []
    for pref in constraints.preferences:
        for day in DAYS:
            if day in pref.preferred_days:
                objective_terms.append(work[(pref.employee_id, day)] * pref.weight)
            if day in pref.avoided_days:
                objective_terms.append((1 - work[(pref.employee_id, day)]) * pref.weight)
    objective_terms.extend(work.values())
    
    # Fairness objective: minimize shift differences between employees in the same role
    if constraints.fairness_constraints:
        for role in [EmployeeRole.regular, EmployeeRole.new]:
            role_employees = [e for e in staff_list if e.role == role]
            for i, emp1 in enumerate(role_employees):
                for emp2 in role_employees[i + 1:]:
                    diff = model.NewIntVar(-len(DAYS), len(DAYS), f"diff_{emp1.id}_{emp2.id}")
                    abs_diff = model.NewIntVar(0, len(DAYS), f"abs_diff_{emp1.id}_{emp2.id}")
                    model.Add(diff == total_shifts[emp1.id] - total_shifts[emp2.id])
                    model.AddAbsEquality(abs_diff, diff)
                    objective_terms.append(-abs_diff * 10)
    
    model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
        return {
            "status": "INFEASIBLE",
            "conflicts": analyze_infeasibility(staff_list, constraints),
        }

    assignments = {
        day.value: [employee.id for employee in staff_list if solver.Value(work[(employee.id, day)]) == 1]
        for day in DAYS
    }
    return {
        "status": "OPTIMAL",
        "assignments": assignments,
        "objective": solver.ObjectiveValue(),
        "conflicts": [],
    }


def analyze_infeasibility(staff_list: List[Employee], constraints: ConstraintSet) -> List[str]:
    messages: List[str] = []
    unavail_map = build_unavailability_map(constraints)
    for day_min in constraints.day_minimums:
        available = [employee.id for employee in staff_list if (employee.id, day_min.day) not in unavail_map]
        if len(available) < day_min.min_staff:
            unavailable = [
                unavail_map[(employee.id, day_min.day)]
                for employee in staff_list
                if (employee.id, day_min.day) in unavail_map
            ]
            reasons = ", ".join(
                f'{u.employee_id}({u.reason or u.type or "unavailable"})' for u in unavailable
            )
            messages.append(
                f"{day_min.day.value} 需至少 {day_min.min_staff} 人，只有 {len(available)} 人可排班。" 
                f"造成衝突的不可用項目：{reasons}。"
            )
    
    for day_max in constraints.day_maximums:
        if day_max.day is None:
            available_count = sum(1 for emp in staff_list if not any((emp.id, day) in unavail_map for day in DAYS))
            if available_count > day_max.max_staff * 7:
                messages.append(
                    f"每天最多 {day_max.max_staff} 人，但可用員工數 {len(staff_list)} > {day_max.max_staff}。"
                    f"需要考慮增加班次或減少員工可上班天數。"
                )
    
    for emp_min in constraints.min_shifts_per_employee:
        employee = next((e for e in staff_list if e.id == emp_min.employee_id), None)
        if not employee:
            continue
        unavail_days = sum(1 for day in DAYS if (employee.id, day) in unavail_map)
        available_days = 7 - unavail_days
        if available_days < emp_min.min_shifts:
            messages.append(
                f"員工 {employee.id} 需至少上 {emp_min.min_shifts} 天班，但只有 {available_days} 天可排班。"
            )
    
    if not messages and constraints.unavailabilities:
        messages.append("現有約束導致排班無解，請檢查請假或出國需求是否過多。")
    return messages
