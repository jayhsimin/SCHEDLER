from ortools.sat.python import cp_model
from typing import Dict, List, Optional, Tuple

from .schemas import (
    ConstraintSet, Employee, Unavailability, Weekday,
)

DAYS = [
    Weekday.monday, Weekday.tuesday, Weekday.wednesday, Weekday.thursday,
    Weekday.friday, Weekday.saturday, Weekday.sunday,
]
DAY_INDEX = {day: idx for idx, day in enumerate(DAYS)}

# Objective weights
_COVERAGE = 2      # reward for each shift assigned
_GAP_PEN = 4 * len(DAYS)   # penalty per unit of max-min fairness gap (= 28)
# With these weights, closing a 1-shift gap is worth 28 points ≈ 14 shifts,
# so fairness dominates coverage when there is meaningful inequity.


def expand_unavailability(unavail: Unavailability) -> List[Tuple[str, Weekday]]:
    if not unavail.end_day:
        return [(unavail.employee_id, unavail.start_day)]
    start = DAY_INDEX[unavail.start_day]
    end = DAY_INDEX[unavail.end_day]
    if end < start:
        end += 7
    return [(unavail.employee_id, DAYS[i % 7]) for i in range(start, end + 1)]


def build_unavailability_map(
    constraints: ConstraintSet,
) -> Dict[Tuple[str, Weekday], Unavailability]:
    result: Dict[Tuple[str, Weekday], Unavailability] = {}
    for unavail in constraints.unavailabilities:
        for key in expand_unavailability(unavail):
            result[key] = unavail
    return result


def solve_schedule(staff_list: List[Employee], constraints: ConstraintSet) -> dict:
    model = cp_model.CpModel()
    emp_ids = {e.id for e in staff_list}

    # Decision variables: work[employee_id][day] ∈ {0, 1}
    work = {
        (e.id, d): model.NewBoolVar(f"w_{e.id}_{d.value}")
        for e in staff_list
        for d in DAYS
    }

    # ── Hard constraints ──

    # Unavailabilities
    unavail_map = build_unavailability_map(constraints)
    for (eid, day), _ in unavail_map.items():
        if (eid, day) in work:
            model.Add(work[(eid, day)] == 0)

    # Per-day minimums
    for dm in constraints.day_minimums:
        if dm.day in DAY_INDEX:
            model.Add(sum(work[(e.id, dm.day)] for e in staff_list) >= dm.min_staff)

    # Per-day maximums
    for dm in constraints.day_maximums:
        days_to_cap = DAYS if dm.day is None else ([dm.day] if dm.day in DAY_INDEX else [])
        for day in days_to_cap:
            model.Add(sum(work[(e.id, day)] for e in staff_list) <= dm.max_staff)

    # Per-employee shift cap (max_shifts_per_week, default 5)
    for e in staff_list:
        model.Add(sum(work[(e.id, d)] for d in DAYS) <= (e.max_shifts_per_week or 5))

    # Per-employee minimum shifts
    for em in constraints.min_shifts_per_employee:
        if em.employee_id in emp_ids:
            model.Add(sum(work[(em.employee_id, d)] for d in DAYS) >= em.min_shifts)

    # Per-employee maximum shifts ("只上N天")
    for em in constraints.max_shifts_per_employee:
        if em.employee_id in emp_ids:
            model.Add(sum(work[(em.employee_id, d)] for d in DAYS) <= em.max_shifts)

    # Role-based day minimums
    for rm in constraints.day_minimums_by_role:
        role_emps = [e for e in staff_list if e.role == rm.role]
        days_to_check = DAYS if rm.day is None else ([rm.day] if rm.day in DAY_INDEX else [])
        for day in days_to_check:
            model.Add(sum(work[(e.id, day)] for e in role_emps) >= rm.min_staff)

    # Mutual exclusions: at most 1 of the listed employees per day
    for excl in constraints.mutual_exclusions:
        valid = [eid for eid in excl.employee_ids if eid in emp_ids]
        if len(valid) >= 2:
            for day in DAYS:
                model.Add(sum(work[(eid, day)] for eid in valid) <= 1)

    # ── Fairness objective ──

    # Total shifts per employee
    total_shifts = {}
    for e in staff_list:
        ts = model.NewIntVar(0, len(DAYS), f"ts_{e.id}")
        model.Add(ts == sum(work[(e.id, d)] for d in DAYS))
        total_shifts[e.id] = ts

    # Max-min fairness gap across all employees
    n = len(staff_list)
    obj_terms = []

    if n > 1:
        max_s = model.NewIntVar(0, len(DAYS), "max_s")
        min_s = model.NewIntVar(0, len(DAYS), "min_s")
        model.AddMaxEquality(max_s, list(total_shifts.values()))
        model.AddMinEquality(min_s, list(total_shifts.values()))
        gap = model.NewIntVar(0, len(DAYS), "gap")
        model.Add(gap == max_s - min_s)
        obj_terms.append(-_GAP_PEN * gap)

    # Coverage reward (secondary to fairness)
    for w in work.values():
        obj_terms.append(_COVERAGE * w)

    # Soft preferences
    for pref in constraints.preferences:
        if pref.employee_id not in emp_ids:
            continue
        weight = pref.weight or 1
        for day in DAYS:
            if day in (pref.preferred_days or []):
                obj_terms.append(work[(pref.employee_id, day)] * weight)
            if day in (pref.avoided_days or []):
                obj_terms.append((1 - work[(pref.employee_id, day)]) * weight)

    model.Maximize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    solver.parameters.num_search_workers = 4
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "status": "INFEASIBLE",
            "conflicts": analyze_infeasibility(staff_list, constraints),
        }

    assignments = {
        day.value: [e.id for e in staff_list if solver.Value(work[(e.id, day)]) == 1]
        for day in DAYS
    }
    return {"status": "OPTIMAL", "assignments": assignments, "conflicts": []}


def solve_schedule_hourly(
    staff_list: List[Employee],
    constraints: ConstraintSet,
    start_hour: int = 9,
    end_hour: int = 18,
    daily_staff_count: Optional[int] = None,
) -> dict:
    """
    小時粒度排班求解器。
    以公平分配每人總上班時數為主要目標，覆蓋率為次要目標。
    assignments 格式：{day: {hour_str: [emp_ids]}}
    """
    model = cp_model.CpModel()
    emp_ids = {e.id for e in staff_list}
    HOURS = list(range(start_hour, end_hour))

    if not HOURS:
        return {"status": "INFEASIBLE", "conflicts": ["上班開始時間必須早於結束時間"]}

    max_total_hours = len(DAYS) * len(HOURS)

    # 決策變數：work[(emp_id, day, hour)] ∈ {0, 1}
    work = {
        (e.id, d, h): model.NewBoolVar(f"w_{e.id}_{d.value}_{h}")
        for e in staff_list
        for d in DAYS
        for h in HOURS
    }

    # ── 預先建立「是否當天有上班」指示變數（供多處約束共用）──
    works_on_day: Dict[Tuple[str, Weekday], object] = {}
    for e in staff_list:
        for d in DAYS:
            w_day = model.NewBoolVar(f"wday_{e.id}_{d.value}")
            # w_day = 1 當且僅當該員工當天有任何一個小時上班
            model.AddMaxEquality(w_day, [work[(e.id, d, h)] for h in HOURS])
            works_on_day[(e.id, d)] = w_day

    # ── 硬約束 ──

    # 不可排班（以天為單位，封鎖整天所有小時）
    unavail_map = build_unavailability_map(constraints)
    for (eid, day), _ in unavail_map.items():
        for h in HOURS:
            if (eid, day, h) in work:
                model.Add(work[(eid, day, h)] == 0)

    # 每小時最低人數（daily_staff_count 在小時模式下代表每小時最低人數）
    if daily_staff_count:
        for d in DAYS:
            for h in HOURS:
                model.Add(
                    sum(work[(e.id, d, h)] for e in staff_list) >= daily_staff_count
                )

    # 按天的最低人數（整天每小時均須達標）
    for dm in constraints.day_minimums:
        if dm.day in DAY_INDEX:
            for h in HOURS:
                model.Add(
                    sum(work[(e.id, dm.day, h)] for e in staff_list) >= dm.min_staff
                )

    # 按天的最高人數（每小時人數上限）
    for dm in constraints.day_maximums:
        days_to_cap = DAYS if dm.day is None else (
            [dm.day] if dm.day in DAY_INDEX else []
        )
        for day in days_to_cap:
            for h in HOURS:
                model.Add(
                    sum(work[(e.id, day, h)] for e in staff_list) <= dm.max_staff
                )

    # 角色最低人數（每小時均須達標）
    for rm in constraints.day_minimums_by_role:
        role_emps = [e for e in staff_list if e.role == rm.role]
        days_to_check = DAYS if rm.day is None else (
            [rm.day] if rm.day in DAY_INDEX else []
        )
        for day in days_to_check:
            for h in HOURS:
                model.Add(
                    sum(work[(e.id, day, h)] for e in role_emps) >= rm.min_staff
                )

    # 互斥約束（同一天同一小時不能同時上班）
    for excl in constraints.mutual_exclusions:
        valid = [eid for eid in excl.employee_ids if eid in emp_ids]
        if len(valid) >= 2:
            for d in DAYS:
                for h in HOURS:
                    model.Add(sum(work[(eid, d, h)] for eid in valid) <= 1)

    # 每人每週最多上班天數（max_shifts_per_week）
    for e in staff_list:
        max_days = e.max_shifts_per_week or 5
        model.Add(sum(works_on_day[(e.id, d)] for d in DAYS) <= max_days)

    # 個別員工最多上班天數
    for em in constraints.max_shifts_per_employee:
        if em.employee_id in emp_ids:
            model.Add(
                sum(works_on_day[(em.employee_id, d)] for d in DAYS) <= em.max_shifts
            )

    # 個別員工最少上班天數
    for em in constraints.min_shifts_per_employee:
        if em.employee_id in emp_ids:
            model.Add(
                sum(works_on_day[(em.employee_id, d)] for d in DAYS) >= em.min_shifts
            )

    # ── 公平性目標：最小化最大-最小總時數差距 ──

    total_hours_vars: Dict[str, object] = {}
    for e in staff_list:
        th = model.NewIntVar(0, max_total_hours, f"th_{e.id}")
        model.Add(th == sum(work[(e.id, d, h)] for d in DAYS for h in HOURS))
        total_hours_vars[e.id] = th

    n = len(staff_list)
    obj_terms = []

    # 差一小時的懲罰遠大於一個覆蓋獎勵，確保公平性優先
    _COVERAGE_H = 1
    _GAP_PEN_H = max_total_hours  # 差一小時 = 損失整體最大時數的獎勵

    if n > 1:
        max_h_var = model.NewIntVar(0, max_total_hours, "max_h_var")
        min_h_var = model.NewIntVar(0, max_total_hours, "min_h_var")
        model.AddMaxEquality(max_h_var, list(total_hours_vars.values()))
        model.AddMinEquality(min_h_var, list(total_hours_vars.values()))
        gap_var = model.NewIntVar(0, max_total_hours, "gap_var")
        model.Add(gap_var == max_h_var - min_h_var)
        obj_terms.append(-_GAP_PEN_H * gap_var)

    # 覆蓋獎勵（次要目標：鼓勵安排人員）
    for w in work.values():
        obj_terms.append(_COVERAGE_H * w)

    # 偏好（軟約束：整天所有小時）
    for pref in constraints.preferences:
        if pref.employee_id not in emp_ids:
            continue
        weight = pref.weight or 1
        for day in DAYS:
            for h in HOURS:
                if day in (pref.preferred_days or []):
                    obj_terms.append(work[(pref.employee_id, day, h)] * weight)
                if day in (pref.avoided_days or []):
                    obj_terms.append((1 - work[(pref.employee_id, day, h)]) * weight)

    model.Maximize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15
    solver.parameters.num_search_workers = 4
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "status": "INFEASIBLE",
            "conflicts": analyze_infeasibility(staff_list, constraints),
        }

    # 組建結果：{day: {hour_str: [emp_ids]}}
    assignments: Dict[str, Dict[str, List[str]]] = {}
    employee_hours: Dict[str, int] = {e.id: 0 for e in staff_list}

    for day in DAYS:
        day_data: Dict[str, List[str]] = {}
        for h in HOURS:
            workers = [
                e.id for e in staff_list
                if solver.Value(work[(e.id, day, h)]) == 1
            ]
            day_data[str(h)] = workers
            for eid in workers:
                employee_hours[eid] += 1
        assignments[day.value] = day_data

    return {
        "status": "OPTIMAL",
        "assignments": assignments,
        "employee_hours": employee_hours,
        "conflicts": [],
    }


def analyze_infeasibility(staff_list: List[Employee], constraints: ConstraintSet) -> List[str]:
    messages: List[str] = []
    unavail_map = build_unavailability_map(constraints)
    emp_ids = {e.id for e in staff_list}

    for dm in constraints.day_minimums:
        available = [e.id for e in staff_list if (e.id, dm.day) not in unavail_map]
        if len(available) < dm.min_staff:
            blocked = [
                f"{unavail_map[(e.id, dm.day)].employee_id}({unavail_map[(e.id, dm.day)].reason or '不可用'})"
                for e in staff_list if (e.id, dm.day) in unavail_map
            ]
            messages.append(
                f"{dm.day.value} 需至少 {dm.min_staff} 人，但只有 {len(available)} 人可排班。"
                f"受限員工：{', '.join(blocked)}。"
            )

    for em in constraints.min_shifts_per_employee:
        emp = next((e for e in staff_list if e.id == em.employee_id), None)
        if not emp:
            continue
        available_days = sum(1 for d in DAYS if (emp.id, d) not in unavail_map)
        if available_days < em.min_shifts:
            messages.append(
                f"員工 {emp.id} 需至少上 {em.min_shifts} 天班，但只有 {available_days} 天可排班。"
            )

    for excl in constraints.mutual_exclusions:
        valid = [eid for eid in excl.employee_ids if eid in emp_ids]
        if len(valid) >= 2:
            # Check if ALL days are blocked for at least one of them with min_shifts constraints
            pass  # OR-Tools will catch true infeasibility

    if not messages:
        messages.append("現有約束條件互相衝突，無法產生合法班表，請檢查約束是否過嚴。")
    return messages
