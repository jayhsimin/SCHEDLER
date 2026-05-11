"""
Eval runner: executes each EvalCase end-to-end and returns structured results.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..prompt import extract_constraints_from_text
from ..schemas import ConstraintSet, Weekday
from ..solver import DAYS, build_unavailability_map, solve_schedule
from ..validator import validate_feasibility, validate_assignments
from .cases import EvalCase, ExtractionCheck, ScheduleCheck


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CaseResult:
    case_id: str
    description: str
    passed: bool
    extraction_checks: List[CheckResult] = field(default_factory=list)
    schedule_checks: List[CheckResult] = field(default_factory=list)
    fairness_gap: Optional[int] = None
    error: Optional[str] = None


# ── Extraction checks ─────────────────────────────────────────────────────────

def _check_extraction(ec: ExtractionCheck, constraints: ConstraintSet) -> List[CheckResult]:
    results: List[CheckResult] = []
    unavail_map = build_unavailability_map(constraints)

    # unavailable_pairs
    for emp_id, day_str in ec.unavailable_pairs:
        day_enum = next((d for d in DAYS if d.value == day_str), None)
        ok = day_enum is not None and (emp_id, day_enum) in unavail_map
        results.append(CheckResult(
            name=f"unavail({emp_id},{day_str})",
            passed=ok,
            detail="" if ok else f"({emp_id},{day_str}) not in extracted unavailabilities",
        ))

    # mutual_exclusions
    extracted_excl = [frozenset(m.employee_ids) for m in constraints.mutual_exclusions]
    for group in ec.mutual_excl_groups:
        ok = frozenset(group) in extracted_excl
        label = "+".join(sorted(group))
        results.append(CheckResult(
            name=f"mutual_excl({label})",
            passed=ok,
            detail="" if ok else f"mutual exclusion {group} not extracted",
        ))

    # day_minimums
    for day_str, min_staff in ec.day_minimums:
        ok = any(
            dm.day.value == day_str and dm.min_staff >= min_staff
            for dm in constraints.day_minimums
        )
        results.append(CheckResult(
            name=f"day_min({day_str}>={min_staff})",
            passed=ok,
            detail="" if ok else f"day_minimum {day_str}>={min_staff} not extracted",
        ))

    # min_shifts_employees
    extracted_emp_mins = {m.employee_id for m in constraints.min_shifts_per_employee}
    for emp_id in ec.min_shifts_employees:
        ok = emp_id in extracted_emp_mins
        results.append(CheckResult(
            name=f"min_shifts({emp_id})",
            passed=ok,
            detail="" if ok else f"min_shifts for {emp_id} not extracted",
        ))

    return results


# ── Schedule checks ───────────────────────────────────────────────────────────

def _check_schedule(sc: ScheduleCheck, assignments: Dict[str, List[str]]) -> List[CheckResult]:
    results: List[CheckResult] = []

    # must_not_work
    for emp_id, day_str in sc.must_not_work:
        working = assignments.get(day_str, [])
        ok = emp_id not in working
        results.append(CheckResult(
            name=f"not_working({emp_id},{day_str})",
            passed=ok,
            detail="" if ok else f"{emp_id} is scheduled on {day_str} but should not be",
        ))

    # must_no_same_day
    for id1, id2 in sc.must_no_same_day:
        violations = [
            day for day, workers in assignments.items()
            if id1 in workers and id2 in workers
        ]
        ok = len(violations) == 0
        results.append(CheckResult(
            name=f"no_same_day({id1},{id2})",
            passed=ok,
            detail="" if ok else f"{id1} and {id2} both work on: {violations}",
        ))

    # min_per_day (specific days)
    for day_str, min_count in sc.min_per_day:
        count = len(assignments.get(day_str, []))
        ok = count >= min_count
        results.append(CheckResult(
            name=f"min_per_day({day_str}>={min_count})",
            passed=ok,
            detail="" if ok else f"{day_str} has {count} workers, need {min_count}",
        ))

    # global_min_per_day
    if sc.global_min_per_day is not None:
        for day_str, workers in assignments.items():
            count = len(workers)
            ok = count >= sc.global_min_per_day
            results.append(CheckResult(
                name=f"global_min({day_str}>={sc.global_min_per_day})",
                passed=ok,
                detail="" if ok else f"{day_str} has {count} workers, global min is {sc.global_min_per_day}",
            ))

    # global_max_per_day
    if sc.global_max_per_day is not None:
        for day_str, workers in assignments.items():
            count = len(workers)
            ok = count <= sc.global_max_per_day
            results.append(CheckResult(
                name=f"global_max({day_str}<={sc.global_max_per_day})",
                passed=ok,
                detail="" if ok else f"{day_str} has {count} workers, global max is {sc.global_max_per_day}",
            ))

    return results


def _fairness_gap(assignments: Dict[str, List[str]], emp_ids: List[str]) -> int:
    counts = {eid: 0 for eid in emp_ids}
    for workers in assignments.values():
        for eid in workers:
            counts[eid] = counts.get(eid, 0) + 1
    if not counts:
        return 0
    return max(counts.values()) - min(counts.values())


# ── Main runner ───────────────────────────────────────────────────────────────

def run_case(case: EvalCase) -> CaseResult:
    try:
        # Step 1: extract constraints
        constraints = extract_constraints_from_text(
            case.text, case.employees, case.daily_staff_count
        )

        extraction_results = _check_extraction(case.extraction, constraints)

        # Step 2: pre-solve feasibility
        feasible, _ = validate_feasibility(case.employees, constraints, case.text)

        if case.must_be_infeasible:
            if not feasible:
                return CaseResult(
                    case_id=case.id,
                    description=case.description,
                    passed=True,
                    extraction_checks=extraction_results,
                    schedule_checks=[CheckResult("infeasible_correctly_rejected", True)],
                )
            # feasible when it shouldn't be
            result = solve_schedule(case.employees, constraints)
            if result["status"] == "INFEASIBLE":
                return CaseResult(
                    case_id=case.id,
                    description=case.description,
                    passed=True,
                    extraction_checks=extraction_results,
                    schedule_checks=[CheckResult("infeasible_correctly_rejected", True)],
                )
            return CaseResult(
                case_id=case.id,
                description=case.description,
                passed=False,
                extraction_checks=extraction_results,
                schedule_checks=[CheckResult(
                    "infeasible_correctly_rejected", False,
                    "Expected INFEASIBLE but solver produced a schedule"
                )],
            )

        if not feasible:
            return CaseResult(
                case_id=case.id,
                description=case.description,
                passed=False,
                extraction_checks=extraction_results,
                schedule_checks=[CheckResult(
                    "pre_solve_feasible", False,
                    "Pre-solve validator rejected a case that should be feasible"
                )],
            )

        # Step 3: solve
        result = solve_schedule(case.employees, constraints)
        if result["status"] == "INFEASIBLE":
            return CaseResult(
                case_id=case.id,
                description=case.description,
                passed=False,
                extraction_checks=extraction_results,
                schedule_checks=[CheckResult(
                    "solver_feasible", False,
                    "Solver returned INFEASIBLE for a case that should be solvable"
                )],
            )

        assignments: Dict[str, List[str]] = result["assignments"]

        # Step 4: schedule checks
        schedule_results = _check_schedule(case.schedule, assignments)

        # Step 5: fairness gap
        emp_ids = [e.id for e in case.employees]
        gap = _fairness_gap(assignments, emp_ids)

        all_checks = extraction_results + schedule_results
        passed = all(c.passed for c in all_checks)

        return CaseResult(
            case_id=case.id,
            description=case.description,
            passed=passed,
            extraction_checks=extraction_results,
            schedule_checks=schedule_results,
            fairness_gap=gap,
        )

    except Exception as exc:
        return CaseResult(
            case_id=case.id,
            description=case.description,
            passed=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_all(cases: List[EvalCase]) -> List[CaseResult]:
    return [run_case(c) for c in cases]
