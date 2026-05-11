import json
import os
import re
from typing import List, Optional

from .schemas import (
    Employee, ConstraintSet, DayMinimum, DayMaximum,
    MinShiftsPerEmployee, MaxShiftsPerEmployee,
    DayMinimumByRole, FairnessConstraint, Unavailability,
    Weekday, EmployeeRole, MutualExclusion,
)

LLM_PROMPT_TEMPLATE = '''你是排班約束抽取器。將使用者的自然語言描述精確解析成 JSON。
只輸出合法 JSON 物件，不包含任何其他文字、說明或 Markdown 標記。

必要欄位（每次都要輸出）：
- intent：只能是 "schedulable" 或 "impossible"
  - "schedulable"：可以排班（包含一般情況、部分人請假、員工不情願但仍須上班等）
  - "impossible"：因事實/客觀因素導致全員真的無法上班（如天災、停班公告、世界末日等）
  - 「不想上班但不得不上班」屬於 schedulable，不是 impossible
- explanation：用一句繁體中文說明你如何理解這個需求（給使用者確認用）

約束欄位：
{
  "intent": "schedulable",
  "explanation": "A 周二起出國三天，B 周三請事假，周三至少三人排班",
  "unavailabilities": [
    {"employee_id":"A", "start_day":"Tuesday", "end_day":"Thursday", "reason":"出國", "type":"vacation"},
    {"employee_id":"B", "start_day":"Wednesday", "reason":"事假", "type":"leave"}
  ],
  "day_minimums": [
    {"day":"Wednesday", "min_staff":3}
  ],
  "day_maximums": [
    {"day":null, "max_staff":4}
  ],
  "mutual_exclusions": [
    {"employee_ids":["I","J"]}
  ],
  "min_shifts_per_employee": [
    {"employee_id":"E", "min_shifts":2}
  ],
  "max_shifts_per_employee": [
    {"employee_id":"B", "max_shifts":2}
  ],
  "fairness_constraints": [
    {"target_shifts_per_person":null}
  ],
  "preferences": [
    {"employee_id":"C", "preferred_days":["Monday"], "avoided_days":["Friday"], "weight":1}
  ]
}

規則：
- 日期統一用英文：Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday
- 若連續請假/出國，start_day 填起始、end_day 填結束（含），否則省略 end_day
- mutual_exclusions 表示列出的員工不可在同一天上班，用於「A 和 B 不能同時上班」
- day_maximums 中 day 為 null 表示每天都套用此上限
- 「只上N天」/「最多N天」/「不超過N天」→ max_shifts_per_employee（上限）
- 「至少上N天」/「要上N天」→ min_shifts_per_employee（下限）
- 「不想上班」/「偏好不上班」→ preferences.avoided_days（軟偏好），不是 unavailabilities
- 「無法上班」/「請假」/「出國」→ unavailabilities（硬約束）
- 所有陣列若無對應約束請輸出空陣列 []
- 不要輸出 JSON 以外的任何文字
'''

WEEKDAY_MAP = {
    "周一": "Monday", "周二": "Tuesday", "周三": "Wednesday", "周四": "Thursday",
    "周五": "Friday", "周六": "Saturday", "周日": "Sunday",
    "禮拜一": "Monday", "禮拜二": "Tuesday", "禮拜三": "Wednesday",
    "禮拜四": "Thursday", "禮拜五": "Friday", "禮拜六": "Saturday", "禮拜日": "Sunday",
}

DAYS_OFFSET = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6}
DAYS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _build_rag_context(employees: List[Employee], daily_staff_count: Optional[int]) -> str:
    """RAG grounding: inject factual employee data so the LLM cannot hallucinate phantom workers."""
    lines = ["【員工事實依據（RAG）】"]
    lines.append(f"本次排班共 {len(employees)} 名員工，以下是他們的確切資料：")
    for emp in employees:
        role_str = "正職" if emp.role and emp.role.value == "regular" else "新進"
        cap = emp.max_shifts_per_week or 5
        lines.append(f"  - 員工 {emp.id}（{role_str}）：每週最多排 {cap} 天")
    if daily_staff_count is not None:
        lines.append(f"每日排班最低人數：{daily_staff_count} 人（每天至少須有這麼多人排班）")
    lines.append("排班週期：Monday 至 Sunday（共 7 天）")
    lines.append(
        "【intent 判斷規則】\n"
        "- intent=\"impossible\"：僅限員工因客觀事實（天災、世界末日、停班公告）真的「無法」上班。\n"
        "- intent=\"schedulable\"：所有其他情況，包含員工「不想」但仍須上班（如「沒人想上班，但不得不上班」）。\n"
        "- 「不想」≠「無法」。有轉折語（不得不、還是要、仍須）時，intent 一律為 schedulable，unavailabilities 為空陣列。"
    )
    return "\n".join(lines)


def make_prompt(text: str, employees: List[Employee], daily_staff_count: Optional[int] = None) -> str:
    rag_context = _build_rag_context(employees, daily_staff_count)
    return f"{LLM_PROMPT_TEMPLATE}\n{rag_context}\n\n使用者輸入：{text}"


def normalize_weekday(chinese: str) -> Optional[str]:
    return WEEKDAY_MAP.get(chinese)


def chinese_numeral_to_int(text: str) -> Optional[int]:
    mapping = {"一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5,
               "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    if text.isdigit():
        return int(text)
    return mapping.get(text)


def _extract_json(raw: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"```", "", cleaned).strip()
    # Find the outermost JSON object
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        return json.loads(match.group(0))
    return json.loads(cleaned)


def parse_simple_constraints(text: str, employees: List[Employee]) -> ConstraintSet:
    text = text.replace("禮拜", "周").replace("週", "周")
    constraints = ConstraintSet()
    emp_ids = {e.id for e in employees}

    # ── Vacation / leave unavailabilities ──
    for emp in employees:
        if emp.name not in text:
            continue
        # "X周N到周M出國/旅遊/出差" → explicit range absence
        vac_range = re.search(
            rf"{re.escape(emp.name)}周([一二三四五六日])到(?:周|禮拜)?([一二三四五六日])(?:要)?(?:出國|旅遊|出差|請假|不能上班|無法上班)",
            text,
        )
        if vac_range:
            start_idx = DAYS_OFFSET.get(vac_range.group(1), 0)
            end_idx = DAYS_OFFSET.get(vac_range.group(2), 6)
            if end_idx < start_idx:
                end_idx = 6
            constraints.unavailabilities.append(
                Unavailability(
                    employee_id=emp.id,
                    start_day=DAYS_EN[start_idx],
                    end_day=DAYS_EN[end_idx] if end_idx != start_idx else None,
                    reason="出國",
                    type="vacation",
                )
            )
            continue  # skip the "N天" pattern for this employee

        # "X周N開始要出國/旅遊M天" → multi-day absence
        vac = re.search(
            rf"{re.escape(emp.name)}周([一二三四五六日])(?:開始)?(?:要)?(?:出國|旅遊|出差)([二三四五]?)天?",
            text,
        )
        if vac:
            start_ch = vac.group(1)
            days_ch = vac.group(2) or "三"
            start_idx = DAYS_OFFSET.get(start_ch, 0)
            n_days = chinese_numeral_to_int(days_ch) or 3
            end_idx = min(start_idx + n_days - 1, 6)
            constraints.unavailabilities.append(
                Unavailability(
                    employee_id=emp.id,
                    start_day=DAYS_EN[start_idx],
                    end_day=DAYS_EN[end_idx] if end_idx != start_idx else None,
                    reason="出國",
                    type="vacation",
                )
            )
        # "X周N上午要請事假" or "X周N請假"
        leave = re.search(
            rf"{re.escape(emp.name)}周([一二三四五六日])(?:上午|下午)?(?:要)?(?:請事假|請假|請病假)",
            text,
        )
        if leave:
            day_ch = leave.group(1)
            constraints.unavailabilities.append(
                Unavailability(
                    employee_id=emp.id,
                    start_day=DAYS_EN[DAYS_OFFSET.get(day_ch, 0)],
                    reason="請假",
                    type="leave",
                )
            )

    # ── Day minimums ──
    # Pattern A: "這周N排班至少要有M人"
    for m in re.finditer(r"(?:這?周([一二三四五六日]))?排班至少(?:要)?有?([一二兩三四五六七八九十\d]+)個?人", text):
        day_ch = m.group(1)
        min_staff = chinese_numeral_to_int(m.group(2))
        if min_staff:
            day = DAYS_EN[DAYS_OFFSET[day_ch]] if day_ch else None
            if day:
                constraints.day_minimums.append(DayMinimum(day=day, min_staff=min_staff))
    # Pattern B: "周N至少要有M人排班"
    for m in re.finditer(r"這?周([一二三四五六日])至少(?:要)?有?([一二兩三四五六七八九十\d]+)個?人(?:排班)?", text):
        day_ch = m.group(1)
        min_staff = chinese_numeral_to_int(m.group(2))
        if min_staff:
            day = DAYS_EN[DAYS_OFFSET[day_ch]]
            constraints.day_minimums.append(DayMinimum(day=day, min_staff=min_staff))

    # ── Day maximums ──
    max_m = re.search(r"一天最多(?:只)?(?:需)?([一二兩三四五六七八九十\d]+)人?", text)
    if max_m:
        max_staff = chinese_numeral_to_int(max_m.group(1))
        if max_staff:
            constraints.day_maximums.append(DayMaximum(max_staff=max_staff))

    # ── Mutual exclusions: "員工X(?:、|和|與)員工Y不能同時上班" ──
    mutual_pattern = re.compile(
        r"(?:員工)?([A-Ja-j])(?:、|和|與|及)(?:員工)?([A-Ja-j])(?:不能同時上班|不能同時|不能一起上班|不能一起)",
        re.IGNORECASE,
    )
    for m in mutual_pattern.finditer(text):
        id1, id2 = m.group(1).upper(), m.group(2).upper()
        if id1 in emp_ids and id2 in emp_ids:
            constraints.mutual_exclusions.append(MutualExclusion(employee_ids=[id1, id2]))

    # ── Role-based minimum ──
    role_m = re.search(r"老員工?每天至少要有(.+?)個?人", text)
    if role_m:
        min_staff = chinese_numeral_to_int(role_m.group(1))
        if min_staff:
            constraints.day_minimums_by_role.append(
                DayMinimumByRole(role=EmployeeRole.regular, min_staff=min_staff)
            )

    # ── Per-employee min shifts ──
    for emp in employees:
        m = re.search(rf"{re.escape(emp.name)}(?:至少|最少)?(?:一)?周?要上([一二兩三四五六七八九十\d]+)天班?", text)
        if m:
            n = chinese_numeral_to_int(m.group(1))
            if n:
                constraints.min_shifts_per_employee.append(
                    MinShiftsPerEmployee(employee_id=emp.id, min_shifts=n)
                )

    # ── Per-employee max shifts ("只上N天" / "最多N天") ──
    for emp in employees:
        m = re.search(
            rf"{re.escape(emp.name)}[^\n]*(?:只上|最多上|最多只上|不超過)([一二兩三四五六七八九十\d]+)天班?",
            text,
        )
        if m:
            n = chinese_numeral_to_int(m.group(1))
            if n:
                constraints.max_shifts_per_employee.append(
                    MaxShiftsPerEmployee(employee_id=emp.id, max_shifts=n)
                )

    # ── New-employee min shifts ──
    new_m = re.search(r"新人一?周?(?:至少|最少)?要上([一二兩三四五六七八九十\d]+)天班?", text)
    if new_m:
        n = chinese_numeral_to_int(new_m.group(1))
        if n:
            for emp in employees:
                if emp.role == EmployeeRole.new:
                    constraints.min_shifts_per_employee.append(
                        MinShiftsPerEmployee(employee_id=emp.id, min_shifts=n)
                    )

    # ── New-employee max shifts ("新人只上N天") ──
    new_max_m = re.search(r"新人一?周?(?:只上|最多上|最多只上|不超過)([一二兩三四五六七八九十\d]+)天班?", text)
    if new_max_m:
        n = chinese_numeral_to_int(new_max_m.group(1))
        if n:
            for emp in employees:
                if emp.role == EmployeeRole.new:
                    constraints.max_shifts_per_employee.append(
                        MaxShiftsPerEmployee(employee_id=emp.id, max_shifts=n)
                    )

    # ── Fairness keyword ──
    if any(kw in text for kw in ("公平", "均勻", "平衡", "平均")):
        constraints.fairness_constraints.append(FairnessConstraint())

    return constraints


def _supplement_regex(constraints: ConstraintSet, text: str, employees: List[Employee]) -> None:
    """
    After LLM extraction, run targeted regex checks to fill in what small models often miss.
    Only adds constraints NOT already present — avoids duplicates.
    """
    norm = text.replace("禮拜", "周").replace("週", "周")
    from .solver import build_unavailability_map, DAYS as _DAYS
    unavail_map = build_unavailability_map(constraints)

    for emp in employees:
        if emp.name not in norm:
            continue

        # ── Leave / vacation the LLM missed ──
        # Range: "X周N到周M出國"
        vac_range = re.search(
            rf"{re.escape(emp.name)}周([一二三四五六日])到(?:周|禮拜)?([一二三四五六日])(?:要)?(?:出國|旅遊|出差|請假|不能上班|無法上班)",
            norm,
        )
        if vac_range:
            start_idx = DAYS_OFFSET.get(vac_range.group(1), 0)
            end_idx = DAYS_OFFSET.get(vac_range.group(2), 6)
            if end_idx < start_idx:
                end_idx = 6
            for i in range(start_idx, end_idx + 1):
                day_enum = _DAYS[i]
                if (emp.id, day_enum) not in unavail_map:
                    constraints.unavailabilities.append(
                        Unavailability(employee_id=emp.id, start_day=DAYS_EN[i], reason="出國", type="vacation")
                    )
                    unavail_map[(emp.id, day_enum)] = constraints.unavailabilities[-1]
        else:
            # N-days vacation
            vac = re.search(
                rf"{re.escape(emp.name)}周([一二三四五六日])(?:開始)?(?:要)?(?:出國|旅遊|出差)([二三四五]?)天?",
                norm,
            )
            if vac:
                start_idx = DAYS_OFFSET.get(vac.group(1), 0)
                n_days = chinese_numeral_to_int(vac.group(2) or "三") or 3
                for i in range(start_idx, min(start_idx + n_days, 7)):
                    day_enum = _DAYS[i]
                    if (emp.id, day_enum) not in unavail_map:
                        constraints.unavailabilities.append(
                            Unavailability(employee_id=emp.id, start_day=DAYS_EN[i], reason="出國", type="vacation")
                        )
                        unavail_map[(emp.id, day_enum)] = constraints.unavailabilities[-1]

            # Leave
            leave = re.search(
                rf"{re.escape(emp.name)}周([一二三四五六日])(?:上午|下午)?(?:要)?(?:請事假|請假|請病假)",
                norm,
            )
            if leave:
                day_idx = DAYS_OFFSET.get(leave.group(1), 0)
                day_enum = _DAYS[day_idx]
                if (emp.id, day_enum) not in unavail_map:
                    constraints.unavailabilities.append(
                        Unavailability(employee_id=emp.id, start_day=DAYS_EN[day_idx], reason="請假", type="leave")
                    )
                    unavail_map[(emp.id, day_enum)] = constraints.unavailabilities[-1]

        # ── Max shifts the LLM missed ──
        existing_max = {m.employee_id for m in constraints.max_shifts_per_employee}
        if emp.id not in existing_max:
            m = re.search(
                rf"{re.escape(emp.name)}[^\n]*(?:只上|最多上|最多只上|不超過)([一二兩三四五六七八九十\d]+)天班?",
                norm,
            )
            if m:
                n = chinese_numeral_to_int(m.group(1))
                if n:
                    constraints.max_shifts_per_employee.append(
                        MaxShiftsPerEmployee(employee_id=emp.id, max_shifts=n)
                    )

    # ── New-employee max shifts ──
    existing_max = {m.employee_id for m in constraints.max_shifts_per_employee}
    new_max_m = re.search(r"新人一?周?(?:只上|最多上|最多只上|不超過)([一二兩三四五六七八九十\d]+)天班?", norm)
    if new_max_m:
        n = chinese_numeral_to_int(new_max_m.group(1))
        if n:
            for emp in employees:
                if emp.role == EmployeeRole.new and emp.id not in existing_max:
                    constraints.max_shifts_per_employee.append(
                        MaxShiftsPerEmployee(employee_id=emp.id, max_shifts=n)
                    )


def extract_constraints_from_text(
    text: str, employees: List[Employee], daily_staff_count: Optional[int] = None
) -> ConstraintSet:
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    if api_key:
        try:
            from requests import post
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": make_prompt(text, employees, daily_staff_count)}],
                "response_format": {"type": "json_object"},  # structured output
                "max_tokens": 1024,
                "temperature": 0.0,
            }
            response = post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=20,
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
            parsed = _extract_json(raw)
            constraints = ConstraintSet.model_validate(parsed)
            # Supplement: fill in what the small LLM missed using targeted regex
            _supplement_regex(constraints, text, employees)
        except Exception:
            constraints = parse_simple_constraints(text, employees)
    else:
        constraints = parse_simple_constraints(text, employees)

    # Always apply daily_staff_count as per-day minimum (regardless of LLM or fallback path)
    if daily_staff_count is not None:
        for day in Weekday:
            constraints.day_minimums.append(DayMinimum(day=day, min_staff=daily_staff_count))

    return constraints
