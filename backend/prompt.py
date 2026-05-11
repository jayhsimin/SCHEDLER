import json
import os
import re
from typing import List, Optional

from .schemas import Employee, ConstraintSet, DayMinimum, DayMaximum, MinShiftsPerEmployee, DayMinimumByRole, FairnessConstraint, ShiftPreference, Unavailability, Weekday, EmployeeRole

LLM_PROMPT_TEMPLATE = '''
你是排班約束抽取器。請將使用者的中文或中英混合文字說明，解析成結構化 JSON。
輸出內容必須完全符合 JSON Schema，不要包含多餘文字。

輸出格式：
{
  "unavailabilities": [
    {"employee_id":"A", "start_day":"Tuesday", "end_day":"Thursday", "reason":"出國", "type":"vacation"},
    {"employee_id":"B", "start_day":"Wednesday", "reason":"事假", "type":"leave"}
  ],
  "day_minimums": [
    {"day":"Wednesday", "min_staff":3}
  ],
  "preferences": [
    {"employee_id":"C", "preferred_days":["Monday"], "avoided_days":[], "weight":1}
  ]
}

請將日期統一為英文星期：Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday。
若無 end_day，請省略該欄位。
若無偏好，可輸出空陣列。
'''

WEEKDAY_MAP = {
    "周一": "Monday",
    "周二": "Tuesday",
    "周三": "Wednesday",
    "周四": "Thursday",
    "周五": "Friday",
    "周六": "Saturday",
    "周日": "Sunday",
    "禮拜一": "Monday",
    "禮拜二": "Tuesday",
    "禮拜三": "Wednesday",
    "禮拜四": "Thursday",
    "禮拜五": "Friday",
    "禮拜六": "Saturday",
    "禮拜日": "Sunday",
}

def make_prompt(text: str, employees: List[Employee], daily_staff_count: Optional[int] = None) -> str:
    names = ", ".join(employee.name for employee in employees)
    prompt = LLM_PROMPT_TEMPLATE + "\n已知員工：" + names + "\n"
    if daily_staff_count is not None:
        prompt += f"本次排班每日應排 {daily_staff_count} 人。\n"
    prompt += "使用者輸入：" + text
    return prompt


def normalize_weekday(chinese: str) -> Optional[str]:
    return WEEKDAY_MAP.get(chinese)


def chinese_numeral_to_int(text: str) -> Optional[int]:
    mapping = {
        "一": 1,
        "二": 2,
        "兩": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    if text.isdigit():
        return int(text)
    if text in mapping:
        return mapping[text]
    if text == "十":
        return 10
    return None


def parse_simple_constraints(text: str, employees: List[Employee], daily_staff_count: Optional[int] = None) -> ConstraintSet:
    text = text.replace("禮拜", "周")
    constraints = ConstraintSet()
    for emp in employees:
        if emp.name in text:
            pattern = rf"{emp.name}周([一二三四五六日])(.*?)((?:三天|一天|二天|四天|五天)|$)"
            match = re.search(pattern, text)
            if match:
                start = normalize_weekday("周" + match.group(1))
                if start and "三天" in text[match.start():match.end()+10]:
                    end = None
                    if match.group(1) == "二":
                        end = "Thursday"
                    elif match.group(1) == "三":
                        end = "Friday"
                    elif match.group(1) == "一":
                        end = "Wednesday"
                    if end:
                        constraints.unavailabilities.append(
                            Unavailability(
                                employee_id=emp.id,
                                start_day=start,
                                end_day=end,
                                reason="出國",
                                type="vacation",
                            )
                        )
    # simple leave patterns
    for emp in employees:
        if emp.name + "周三上午" in text and "事假" in text:
            constraints.unavailabilities.append(
                Unavailability(
                    employee_id=emp.id,
                    start_day="Wednesday",
                    reason="事假",
                    type="leave",
                )
            )
    # simple day minimum patterns
    min_match = re.search(r"這?周三排班至少要有([一二三四五六七八九十\d]+)個人", text)
    if min_match:
        min_staff = chinese_numeral_to_int(min_match.group(1))
        if min_staff is not None:
            constraints.day_minimums.append(
                DayMinimum(day="Wednesday", min_staff=min_staff)
            )
    
    # day maximum patterns (一天最多 N 人)
    max_match = re.search(r"一天最多(?:只)?(?:需)?([一二三四五六七八九十\d]+)人?", text)
    if max_match:
        max_staff = chinese_numeral_to_int(max_match.group(1))
        if max_staff is not None:
            constraints.day_maximums.append(
                DayMaximum(max_staff=max_staff)
            )
    
    # role-based day minimum (老員工每天至少要有兩個人) - MUST be before individual employee patterns
    if "老員工" in text or "老" in text:
        pattern = r"老員工?每天至少要有(.+?)個?人"
        match = re.search(pattern, text)
        if match:
            min_staff = chinese_numeral_to_int(match.group(1))
            if min_staff is not None:
                constraints.day_minimums_by_role.append(
                    DayMinimumByRole(role=EmployeeRole.regular, min_staff=min_staff)
                )
    
    # daily_staff_count: treat as a per-day maximum (and minimum) for all days
    if daily_staff_count is not None:
        constraints.day_maximums.append(DayMaximum(max_staff=daily_staff_count))
        for day in Weekday:
            constraints.day_minimums.append(DayMinimum(day=day, min_staff=daily_staff_count))

    # fairness constraint (公時要盡量公平)
    if "公平" in text or "均勻" in text or "平衡" in text:
        constraints.fairness_constraints.append(
            FairnessConstraint(target_shifts_per_person=None)
        )
    
    # employee minimum shifts pattern (E 至少一周要上兩天班) - individual employee constraints
    for emp in employees:
        pattern = rf"{emp.name}(?:至少|最少)?(?:一)?周?要上(.+?)天班?"
        match = re.search(pattern, text)
        if match:
            min_shifts = chinese_numeral_to_int(match.group(1))
            if min_shifts is not None:
                constraints.min_shifts_per_employee.append(
                    MinShiftsPerEmployee(employee_id=emp.id, min_shifts=min_shifts)
                )
    
    # new employee pattern (新人一周至少要上兩天班)
    new_pattern = r"新人一?周?(?:至少)?(?:最少)?要上(.+?)天班?"
    new_match = re.search(new_pattern, text)
    if new_match:
        min_shifts = chinese_numeral_to_int(new_match.group(1))
        if min_shifts is not None:
            new_emps = [e for e in employees if e.role == EmployeeRole.new]
            for emp in new_emps:
                constraints.min_shifts_per_employee.append(
                    MinShiftsPerEmployee(employee_id=emp.id, min_shifts=min_shifts)
                )
    
    return constraints


def extract_constraints_from_text(text: str, employees: List[Employee], daily_staff_count: Optional[int] = None) -> ConstraintSet:
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    if api_key:
        try:
            from requests import post
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": make_prompt(text, employees, daily_staff_count)}],
                "max_tokens": 512,
                "temperature": 0.0,
            }
            response = post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            raw_text = data["choices"][0]["message"]["content"]
            parsed = json.loads(raw_text)
            return ConstraintSet.parse_obj(parsed)
        except Exception:
            pass
    return parse_simple_constraints(text, employees, daily_staff_count)
