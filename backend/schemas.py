from enum import Enum
from typing import List, Optional
from pydantic import BaseModel

class EmployeeRole(str, Enum):
    regular = "regular"
    new = "new"

class Weekday(str, Enum):
    monday = "Monday"
    tuesday = "Tuesday"
    wednesday = "Wednesday"
    thursday = "Thursday"
    friday = "Friday"
    saturday = "Saturday"
    sunday = "Sunday"

    @classmethod
    def from_chinese(cls, name: str) -> "Weekday":
        mapping = {
            "周一": cls.monday, "周二": cls.tuesday, "周三": cls.wednesday,
            "周四": cls.thursday, "周五": cls.friday, "周六": cls.saturday,
            "周日": cls.sunday, "禮拜一": cls.monday, "禮拜二": cls.tuesday,
            "禮拜三": cls.wednesday, "禮拜四": cls.thursday, "禮拜五": cls.friday,
            "禮拜六": cls.saturday, "禮拜日": cls.sunday,
        }
        return mapping.get(name, name)  # type: ignore

class Employee(BaseModel):
    id: str
    name: str
    max_shifts_per_week: Optional[int] = 5  # default 5 days to avoid over-scheduling
    role: Optional[EmployeeRole] = EmployeeRole.regular

class Unavailability(BaseModel):
    employee_id: str
    start_day: Weekday
    end_day: Optional[Weekday] = None
    reason: Optional[str] = None
    type: Optional[str] = None

class DayMinimum(BaseModel):
    day: Weekday
    min_staff: int

class DayMaximum(BaseModel):
    day: Optional[Weekday] = None
    max_staff: int

class MinShiftsPerEmployee(BaseModel):
    employee_id: str
    min_shifts: int

class DayMinimumByRole(BaseModel):
    role: EmployeeRole
    day: Optional[Weekday] = None
    min_staff: int

class FairnessConstraint(BaseModel):
    role: Optional[EmployeeRole] = None
    target_shifts_per_person: Optional[int] = None

class ShiftPreference(BaseModel):
    employee_id: str
    preferred_days: Optional[List[Weekday]] = []
    avoided_days: Optional[List[Weekday]] = []
    weight: Optional[int] = 1

class MutualExclusion(BaseModel):
    """These employees cannot work on the same day."""
    employee_ids: List[str]

class ConstraintSet(BaseModel):
    unavailabilities: List[Unavailability] = []
    day_minimums: List[DayMinimum] = []
    day_maximums: List[DayMaximum] = []
    day_minimums_by_role: List[DayMinimumByRole] = []
    min_shifts_per_employee: List[MinShiftsPerEmployee] = []
    fairness_constraints: List[FairnessConstraint] = []
    preferences: List[ShiftPreference] = []
    mutual_exclusions: List[MutualExclusion] = []

class ScheduleRequest(BaseModel):
    text: str
    employees: Optional[List[Employee]] = None
    daily_staff_count: Optional[int] = None

class ScheduleResult(BaseModel):
    assignments: dict
    explanation: str
    constraints: ConstraintSet
    daily_staff_count: Optional[int] = None
    conflict_reasons: Optional[List[str]] = None
