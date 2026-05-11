"""
Eval test case definitions.

Each EvalCase has:
  - id / description
  - input text + employees + optional daily_staff_count
  - extraction_checks: facts that must appear in the extracted ConstraintSet
  - schedule_checks:   facts the final schedule must satisfy
  - must_be_infeasible: True when we expect NO schedule to be produced
"""
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from ..schemas import Employee, EmployeeRole

# ── Shared employee pools ─────────────────────────────────────────────────────

def _make_employees(ids: str) -> List[Employee]:
    """'ABCDE' → five Employee objects."""
    result = []
    for ch in ids:
        role = EmployeeRole.new if ch in ("E", "F") else EmployeeRole.regular
        result.append(Employee(id=ch, name=ch, role=role))
    return result

FIVE  = _make_employees("ABCDE")
TEN   = _make_employees("ABCDEFGHIJ")


@dataclass
class ExtractionCheck:
    """Assert that specific constraints were extracted."""
    # (employee_id, day_english) pairs that must appear in unavailabilities
    unavailable_pairs: List[Tuple[str, str]] = field(default_factory=list)
    # employee-id sets that must appear in mutual_exclusions
    mutual_excl_groups: List[Set[str]] = field(default_factory=list)
    # (day, min_staff) tuples that must appear in day_minimums
    day_minimums: List[Tuple[str, int]] = field(default_factory=list)
    # employee ids that must appear in min_shifts_per_employee
    min_shifts_employees: List[str] = field(default_factory=list)
    # (employee_id, max_shifts) tuples that must appear in max_shifts_per_employee
    max_shifts_employees: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class ScheduleCheck:
    """Assert that the returned schedule satisfies these rules."""
    # (employee_id, day_english) pairs where the employee must NOT appear
    must_not_work: List[Tuple[str, str]] = field(default_factory=list)
    # pairs of employee ids that must never appear on the same day
    must_no_same_day: List[Tuple[str, str]] = field(default_factory=list)
    # (day, min_count) — that day must have at least this many workers
    min_per_day: List[Tuple[str, int]] = field(default_factory=list)
    # every day must have at least this many workers
    global_min_per_day: Optional[int] = None
    # every day must have at most this many workers
    global_max_per_day: Optional[int] = None
    # (employee_id, max_shifts) — employee must not exceed this many shifts total
    max_shifts_per_emp: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class EvalCase:
    id: str
    description: str
    text: str
    employees: List[Employee]
    daily_staff_count: Optional[int] = None
    extraction: ExtractionCheck = field(default_factory=ExtractionCheck)
    schedule: ScheduleCheck = field(default_factory=ScheduleCheck)
    must_be_infeasible: bool = False


# ── Test cases ────────────────────────────────────────────────────────────────

CASES: List[EvalCase] = [

    EvalCase(
        id="TC-01",
        description="Single multi-day vacation",
        text="A周二開始要出國三天",
        employees=FIVE,
        extraction=ExtractionCheck(
            unavailable_pairs=[("A", "Tuesday"), ("A", "Wednesday"), ("A", "Thursday")],
        ),
        schedule=ScheduleCheck(
            must_not_work=[("A", "Tuesday"), ("A", "Wednesday"), ("A", "Thursday")],
        ),
    ),

    EvalCase(
        id="TC-02",
        description="Single day leave",
        text="B周三上午要請事假",
        employees=FIVE,
        extraction=ExtractionCheck(
            unavailable_pairs=[("B", "Wednesday")],
        ),
        schedule=ScheduleCheck(
            must_not_work=[("B", "Wednesday")],
        ),
    ),

    EvalCase(
        id="TC-03",
        description="Day minimum constraint",
        text="這周三排班至少要有3個人",
        employees=FIVE,
        extraction=ExtractionCheck(
            day_minimums=[("Wednesday", 3)],
        ),
        schedule=ScheduleCheck(
            min_per_day=[("Wednesday", 3)],
        ),
    ),

    EvalCase(
        id="TC-04",
        description="Mutual exclusion I/J",
        text="員工I和員工J不能同時上班",
        employees=TEN,
        extraction=ExtractionCheck(
            mutual_excl_groups=[{"I", "J"}],
        ),
        schedule=ScheduleCheck(
            must_no_same_day=[("I", "J")],
        ),
    ),

    EvalCase(
        id="TC-05",
        description="Vacation + leave + day minimum",
        text="A周二開始要出國三天，B周三上午要請事假，這周三排班至少要有3個人",
        employees=FIVE,
        extraction=ExtractionCheck(
            unavailable_pairs=[("A", "Tuesday"), ("A", "Wednesday"), ("A", "Thursday"), ("B", "Wednesday")],
            day_minimums=[("Wednesday", 3)],
        ),
        schedule=ScheduleCheck(
            must_not_work=[("A", "Tuesday"), ("A", "Wednesday"), ("A", "Thursday"), ("B", "Wednesday")],
            min_per_day=[("Wednesday", 3)],
        ),
    ),

    EvalCase(
        id="TC-06",
        description="User complaint case: vacation + leave + mutual exclusion",
        text="A周二開始要出國三天，B周三上午要請事假，員工I、員工J不能同時上班",
        employees=TEN,
        extraction=ExtractionCheck(
            unavailable_pairs=[("A", "Tuesday"), ("A", "Wednesday"), ("A", "Thursday"), ("B", "Wednesday")],
            mutual_excl_groups=[{"I", "J"}],
        ),
        schedule=ScheduleCheck(
            must_not_work=[("A", "Tuesday"), ("A", "Wednesday"), ("A", "Thursday"), ("B", "Wednesday")],
            must_no_same_day=[("I", "J")],
        ),
    ),

    EvalCase(
        id="TC-07",
        description="daily_staff_count=3 applied globally",
        text="正常排班",
        employees=FIVE,
        daily_staff_count=3,
        schedule=ScheduleCheck(
            global_min_per_day=3,
        ),
    ),

    EvalCase(
        id="TC-08",
        description="Impossible scenario — all blocked",
        text="世界末日了，沒人可以上班",
        employees=FIVE,
        must_be_infeasible=True,
    ),

    EvalCase(
        id="TC-09",
        description="Range-format vacation + daily_staff_count",
        text="A周一到周三出國",
        employees=FIVE,
        daily_staff_count=2,
        extraction=ExtractionCheck(
            unavailable_pairs=[("A", "Monday"), ("A", "Tuesday"), ("A", "Wednesday")],
        ),
        schedule=ScheduleCheck(
            must_not_work=[("A", "Monday"), ("A", "Tuesday"), ("A", "Wednesday")],
            global_min_per_day=2,
        ),
    ),

    EvalCase(
        id="TC-10",
        description="Multiple employees different-day leave",
        text="A周一請假，B周二請假，C周三請假",
        employees=FIVE,
        extraction=ExtractionCheck(
            unavailable_pairs=[("A", "Monday"), ("B", "Tuesday"), ("C", "Wednesday")],
        ),
        schedule=ScheduleCheck(
            must_not_work=[("A", "Monday"), ("B", "Tuesday"), ("C", "Wednesday")],
        ),
    ),

    EvalCase(
        id="TC-11",
        description="daily_staff_count + mutual exclusion combo",
        text="員工I和員工J不能同時上班",
        employees=TEN,
        daily_staff_count=2,
        extraction=ExtractionCheck(
            mutual_excl_groups=[{"I", "J"}],
        ),
        schedule=ScheduleCheck(
            must_no_same_day=[("I", "J")],
            global_min_per_day=2,
        ),
    ),

    EvalCase(
        id="TC-12",
        description="Travel + mutual exclusion + day minimum",
        text="A周三開始出國兩天，員工I和員工J不能同時上班，周五至少要有4人排班",
        employees=TEN,
        extraction=ExtractionCheck(
            unavailable_pairs=[("A", "Wednesday"), ("A", "Thursday")],
            mutual_excl_groups=[{"I", "J"}],
            day_minimums=[("Friday", 4)],
        ),
        schedule=ScheduleCheck(
            must_not_work=[("A", "Wednesday"), ("A", "Thursday")],
            must_no_same_day=[("I", "J")],
            min_per_day=[("Friday", 4)],
        ),
    ),

    EvalCase(
        id="TC-13",
        description="Fairness: no constraint should distribute shifts evenly",
        text="正常排班，盡量公平",
        employees=FIVE,
        daily_staff_count=3,
        schedule=ScheduleCheck(
            global_min_per_day=3,
        ),
    ),

    EvalCase(
        id="TC-14",
        description="Max one staff per day (extreme max constraint)",
        text="一天最多只需1人",
        employees=FIVE,
        schedule=ScheduleCheck(
            global_max_per_day=1,
        ),
    ),

    EvalCase(
        id="TC-15",
        description="週 normalization + multi-day vacation",
        text="C週二出國四天",
        employees=FIVE,
        extraction=ExtractionCheck(
            unavailable_pairs=[("C", "Tuesday"), ("C", "Wednesday"), ("C", "Thursday"), ("C", "Friday")],
        ),
        schedule=ScheduleCheck(
            must_not_work=[("C", "Tuesday"), ("C", "Wednesday"), ("C", "Thursday"), ("C", "Friday")],
        ),
    ),

    EvalCase(
        id="TC-16",
        description="Per-employee max shifts (只上N天)",
        text="B只上兩天班",
        employees=FIVE,
        extraction=ExtractionCheck(
            max_shifts_employees=[("B", 2)],
        ),
        schedule=ScheduleCheck(
            max_shifts_per_emp=[("B", 2)],
        ),
    ),

    EvalCase(
        id="TC-17",
        description="New-employee max shifts (新人只上N天)",
        text="新人一周只上一天班",
        employees=FIVE,
        extraction=ExtractionCheck(
            max_shifts_employees=[("E", 1)],
        ),
        schedule=ScheduleCheck(
            max_shifts_per_emp=[("E", 1)],
        ),
    ),
]
