"""
Three-layer hallucination guard:
  1. validate_feasibility  – pre-solve: are the extracted constraints even possible?
  2. validate_assignments  – post-solve: does the generated schedule respect every rule?
"""
from typing import List, Tuple

from .schemas import ConstraintSet, Employee, SchedulingIntent, Weekday
from .solver import DAYS, build_unavailability_map

# Fallback keyword list — only used when LLM path is unavailable (regex fallback)
_IMPOSSIBLE_PHRASES = [
    "世界末日", "沒人可以上班", "沒有人可以上班", "全員無法上班",
    "所有人都無法上班", "全體請假", "所有人請假", "沒人可上班",
    "無人可排班", "末日", "大家都不能上班",
]
_OVERRIDE_PHRASES = [
    "不得不上班", "還是要上班", "仍須上班", "但不得不", "但還是要",
    "不得不排班", "還是得上班",
]


def _all_days_blocked(emp_id: str, unavail_map: dict) -> bool:
    return all((emp_id, day) in unavail_map for day in DAYS)


def validate_feasibility(
    staff_list: List[Employee],
    constraints: ConstraintSet,
    original_text: str = "",
) -> Tuple[bool, List[str]]:
    """
    Run before the solver.
    Returns (is_feasible, list_of_error_messages).
    """
    errors: List[str] = []
    unavail_map = build_unavailability_map(constraints)

    # ── Layer A: intent-based impossible-scenario detection ──
    if constraints.intent == SchedulingIntent.impossible:
        # LLM explicitly said scheduling is impossible
        reason = constraints.explanation or "LLM 判斷此情境為全員無法上班"
        errors.append(f"無法排班：{reason}")
        return False, errors

    if constraints.intent is None:
        # Fallback path (no LLM): use keyword heuristic
        triggered = (
            any(phrase in original_text for phrase in _IMPOSSIBLE_PHRASES)
            and not any(phrase in original_text for phrase in _OVERRIDE_PHRASES)
        )
        if triggered and not constraints.unavailabilities:
            errors.append(
                "輸入描述了全員無法上班的情境，但系統未能從中解析出具體約束。"
                "請明確說明哪些員工在哪些日期不可上班。"
            )
            return False, errors

    # ── Layer B: all employees blocked ──
    fully_blocked = [e for e in staff_list if _all_days_blocked(e.id, unavail_map)]
    if fully_blocked and len(fully_blocked) == len(staff_list):
        names = "、".join(e.id for e in fully_blocked)
        errors.append(f"所有員工（{names}）本週均無可排班日，無法產生有效班表。")
        return False, errors

    # ── Layer C: day minimums cannot be satisfied ──
    for dm in constraints.day_minimums:
        available = [e for e in staff_list if (e.id, dm.day) not in unavail_map]
        if len(available) < dm.min_staff:
            blocked_names = ", ".join(
                f"{e.id}（{unavail_map[(e.id, dm.day)].reason or '不可用'}）"
                for e in staff_list if (e.id, dm.day) in unavail_map
            )
            errors.append(
                f"{dm.day.value} 需至少 {dm.min_staff} 人，"
                f"但僅有 {len(available)} 人可排班。"
                f"受限員工：{blocked_names}。"
            )

    # ── Layer D: mutual exclusions make per-employee minimums impossible ──
    excl_pairs = {
        frozenset(excl.employee_ids)
        for excl in constraints.mutual_exclusions
    }
    for em in constraints.min_shifts_per_employee:
        emp = next((e for e in staff_list if e.id == em.employee_id), None)
        if emp is None:
            continue
        available_days = sum(1 for d in DAYS if (emp.id, d) not in unavail_map)
        if available_days < em.min_shifts:
            errors.append(
                f"員工 {emp.id} 需至少 {em.min_shifts} 天班，"
                f"但只有 {available_days} 天可排班。"
            )

    if errors:
        return False, errors
    return True, []


def validate_assignments(
    assignments: dict,
    staff_list: List[Employee],
    constraints: ConstraintSet,
) -> List[str]:
    """
    Run after the solver produces a schedule.
    Returns a list of constraint violations (should be empty if solver is correct).
    """
    violations: List[str] = []
    unavail_map = build_unavailability_map(constraints)
    emp_ids = {e.id for e in staff_list}
    emp_map = {e.id: e for e in staff_list}

    shift_counts: dict[str, int] = {e.id: 0 for e in staff_list}

    for day_str, working_ids in assignments.items():
        day_enum = next((d for d in DAYS if d.value == day_str), None)
        if day_enum is None:
            continue

        for eid in working_ids:
            shift_counts[eid] = shift_counts.get(eid, 0) + 1

            # Unavailability check
            if (eid, day_enum) in unavail_map:
                u = unavail_map[(eid, day_enum)]
                violations.append(
                    f"⚠ 員工 {eid} 在 {day_str} 被排班，但該日因「{u.reason or u.type or '不可用'}」應休息。"
                )

        # Mutual exclusion check
        for excl in constraints.mutual_exclusions:
            colliding = [eid for eid in excl.employee_ids if eid in working_ids]
            if len(colliding) > 1:
                violations.append(
                    f"⚠ {' 與 '.join(colliding)} 在 {day_str} 同時上班，違反互斥約束。"
                )

        # Day maximum check
        for dm in constraints.day_maximums:
            applies = dm.day is None or (dm.day is not None and dm.day.value == day_str)
            if applies and len(working_ids) > dm.max_staff:
                violations.append(
                    f"⚠ {day_str} 安排了 {len(working_ids)} 人，超過上限 {dm.max_staff} 人。"
                )

        # Day minimum check
        for dm in constraints.day_minimums:
            if dm.day.value == day_str and len(working_ids) < dm.min_staff:
                violations.append(
                    f"⚠ {day_str} 只有 {len(working_ids)} 人，未達最低要求 {dm.min_staff} 人。"
                )

    # Per-employee shift minimum check
    for em in constraints.min_shifts_per_employee:
        count = shift_counts.get(em.employee_id, 0)
        if count < em.min_shifts:
            violations.append(
                f"⚠ 員工 {em.employee_id} 只排了 {count} 天，未達最低要求 {em.min_shifts} 天。"
            )

    # Per-employee shift cap check
    for e in staff_list:
        cap = e.max_shifts_per_week or 5
        count = shift_counts.get(e.id, 0)
        if count > cap:
            violations.append(
                f"⚠ 員工 {e.id} 被排了 {count} 天，超過每週上限 {cap} 天。"
            )

    # Sanity: empty schedule with no infeasibility reason
    total = sum(len(v) for v in assignments.values())
    if total == 0 and not constraints.unavailabilities:
        violations.append("⚠ 班表完全空白（無人排班），可能是系統錯誤或約束過嚴。")

    return violations
