import json
import os
import re
from typing import List, Optional

from langsmith import traceable

from .schemas import (
    Employee, ConstraintSet, DayMinimum, DayMaximum,
    MinShiftsPerEmployee, MaxShiftsPerEmployee,
    DayMinimumByRole, FairnessConstraint, Unavailability,
    Weekday, EmployeeRole, MutualExclusion,
)

# ── Static system message (schema definition + rules) ─────────────────────────
# Kept separate from RAG context so the model treats it as stable instructions.

_LLM_SYSTEM_MSG = '''你是排班約束抽取器。將使用者的自然語言描述精確解析成 JSON。
只輸出合法 JSON 物件，不包含任何其他文字、說明或 Markdown 標記。

必要欄位（每次都要輸出，缺一不可）：
- intent: "schedulable" 或 "impossible"
  - schedulable：可以排班（一般情況、部分請假、不情願但仍須上班）
  - impossible：因客觀事實（天災、停班、世界末日）全員真的無法上班
- explanation：一句繁體中文說明你如何理解此需求

所有約束欄位（無對應項目輸出空陣列 []）：
{
  "intent": "schedulable",
  "explanation": "...",
  "unavailabilities": [
    {"employee_id":"A", "start_day":"Tuesday", "end_day":"Thursday", "reason":"出國", "type":"vacation"},
    {"employee_id":"B", "start_day":"Wednesday", "reason":"請假", "type":"leave"}
  ],
  "day_minimums":        [{"day":"Wednesday", "min_staff":3}],
  "day_maximums":        [{"day":null, "max_staff":4}],
  "day_minimums_by_role":[],
  "min_shifts_per_employee": [{"employee_id":"E", "min_shifts":2}],
  "max_shifts_per_employee": [{"employee_id":"B", "max_shifts":2}],
  "fairness_constraints":    [],
  "preferences":             [{"employee_id":"C", "preferred_days":["Monday"], "avoided_days":["Friday"], "weight":1}],
  "mutual_exclusions":       [{"employee_ids":["I","J"]}]
}

規則：
- 日期用英文：Monday Tuesday Wednesday Thursday Friday Saturday Sunday
- 連續不在 → start_day 填起始、end_day 填結束（含）；單天 → 省略 end_day
- mutual_exclusions：列出的員工不可同一天上班
- day_maximums 的 day=null 表示每天都套用
- 「只上N天」/「最多N天」/「不超過N天」→ max_shifts_per_employee
- 「至少上N天」/「要上N天」→ min_shifts_per_employee
- 「不想上班」/「偏好不上班」→ preferences.avoided_days（軟偏好，非 unavailabilities）
- 「無法上班」/「請假」/「出國」→ unavailabilities（硬約束）
- 「新人」/「新員工」→ 對照 RAG 角色表，對每位新進員工各產生一筆記錄
- 「老員工」/「正職」→ 同上，展開成個別 ID
- 「不想…但不得不」→ intent=schedulable，unavailabilities=[]'''

# ── Few-shot examples (user/assistant message pairs) ──────────────────────────
# Each tuple: (user_text, assistant_json_string)
# These demonstrate the hardest patterns so even small models can follow.

_FEW_SHOT: list[tuple[str, str]] = [
    # Case 1: vacation range + single leave + mutual exclusion
    (
        "A周二開始出國三天，B周三請事假，員工I和員工J不能同時上班",
        '{"intent":"schedulable","explanation":"A周二起出國三天不可排班，B周三請假，I與J不可同日上班",'
        '"unavailabilities":['
        '{"employee_id":"A","start_day":"Tuesday","end_day":"Thursday","reason":"出國","type":"vacation"},'
        '{"employee_id":"B","start_day":"Wednesday","reason":"請假","type":"leave"}],'
        '"day_minimums":[],"day_maximums":[],"day_minimums_by_role":[],'
        '"min_shifts_per_employee":[],'
        '"max_shifts_per_employee":[],'
        '"fairness_constraints":[],"preferences":[],'
        '"mutual_exclusions":[{"employee_ids":["I","J"]}]}',
    ),
    # Case 2: new-employee max + per-employee max (demonstrates role expansion)
    (
        "新人一周只上一天班，B只上兩天班\n【角色對照】「新人」= 員工I、員工J",
        '{"intent":"schedulable","explanation":"新進員工I、J每週最多1班，B最多2班",'
        '"unavailabilities":[],"day_minimums":[],"day_maximums":[],"day_minimums_by_role":[],'
        '"min_shifts_per_employee":[],'
        '"max_shifts_per_employee":['
        '{"employee_id":"I","max_shifts":1},'
        '{"employee_id":"J","max_shifts":1},'
        '{"employee_id":"B","max_shifts":2}],'
        '"fairness_constraints":[],"preferences":[],"mutual_exclusions":[]}',
    ),
    # Case 3: reluctant-but-available (intent=schedulable, NOT impossible)
    (
        "大家都不想上班，但還是要來",
        '{"intent":"schedulable","explanation":"員工不情願但仍須出勤，無實際不可排班情況",'
        '"unavailabilities":[],"day_minimums":[],"day_maximums":[],"day_minimums_by_role":[],'
        '"min_shifts_per_employee":[],"max_shifts_per_employee":[],'
        '"fairness_constraints":[],"preferences":[],"mutual_exclusions":[]}',
    ),
]

# Required JSON keys — used in retry error message
_REQUIRED_KEYS = (
    "intent explanation unavailabilities day_minimums day_maximums "
    "day_minimums_by_role min_shifts_per_employee max_shifts_per_employee "
    "fairness_constraints preferences mutual_exclusions"
).split()

WEEKDAY_MAP = {
    "周一": "Monday", "周二": "Tuesday", "周三": "Wednesday", "周四": "Thursday",
    "周五": "Friday", "周六": "Saturday", "周日": "Sunday",
    "禮拜一": "Monday", "禮拜二": "Tuesday", "禮拜三": "Wednesday",
    "禮拜四": "Thursday", "禮拜五": "Friday", "禮拜六": "Saturday", "禮拜日": "Sunday",
}

DAYS_OFFSET = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6}
DAYS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _build_rag_context(employees: List[Employee], daily_staff_count: Optional[int]) -> str:
    """Dynamic RAG: inject factual employee data so the LLM cannot hallucinate."""
    lines = ["【員工事實依據（RAG）】"]
    lines.append(f"本次排班共 {len(employees)} 名員工：")
    for emp in employees:
        role_str = "正職" if emp.role and emp.role.value == "regular" else "新進"
        cap = emp.max_shifts_per_week or 5
        lines.append(f"  - 員工 {emp.id}（名稱：{emp.name}，{role_str}）：每週最多排 {cap} 天")

    new_emps = [e for e in employees if e.role and e.role.value == "new"]
    reg_emps = [e for e in employees if not e.role or e.role.value == "regular"]
    if new_emps:
        ids = "、".join(f"員工{e.id}（{e.name}）" for e in new_emps)
        lines.append(f"【角色對照】「新人」/「新員工」/「新進員工」= {ids}")
        lines.append("  → 凡提到「新人」，必須對上列每位新進員工各產生一筆約束記錄")
    if reg_emps:
        ids = "、".join(f"員工{e.id}（{e.name}）" for e in reg_emps)
        lines.append(f"【角色對照】「老員工」/「正職」= {ids}")

    if daily_staff_count is not None:
        lines.append(f"每日最低排班人數：{daily_staff_count} 人")
    lines.append("排班週期：Monday 至 Sunday（共 7 天）")
    lines.append(
        "【intent 判斷】impossible 僅限客觀事實（天災/停班/世界末日）；"
        "「不想但不得不」屬於 schedulable。"
    )
    return "\n".join(lines)


def make_messages(
    text: str, employees: List[Employee], daily_staff_count: Optional[int] = None
) -> list[dict]:
    """Build the full messages array: system + few-shot pairs + actual user input."""
    rag = _build_rag_context(employees, daily_staff_count)
    msgs: list[dict] = [{"role": "system", "content": _LLM_SYSTEM_MSG}]
    for user_ex, assistant_ex in _FEW_SHOT:
        msgs.append({"role": "user",      "content": user_ex})
        msgs.append({"role": "assistant", "content": assistant_ex})
    msgs.append({"role": "user", "content": f"{rag}\n\n使用者輸入：{text}"})
    return msgs


# kept for backward compatibility
def make_prompt(text: str, employees: List[Employee], daily_staff_count: Optional[int] = None) -> str:
    rag = _build_rag_context(employees, daily_staff_count)
    return f"{_LLM_SYSTEM_MSG}\n{rag}\n\n使用者輸入：{text}"


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
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"```", "", cleaned).strip()
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
            continue

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
        # "X周N請假"
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
    for m in re.finditer(r"(?:這?周([一二三四五六日]))?排班至少(?:要)?有?([一二兩三四五六七八九十\d]+)個?人", text):
        day_ch = m.group(1)
        min_staff = chinese_numeral_to_int(m.group(2))
        if min_staff:
            day = DAYS_EN[DAYS_OFFSET[day_ch]] if day_ch else None
            if day:
                constraints.day_minimums.append(DayMinimum(day=day, min_staff=min_staff))
    for m in re.finditer(r"這?周([一二三四五六日])至少(?:要)?有?([一二兩三四五六七八九十\d]+)個?人(?:排班)?", text):
        day_ch = m.group(1)
        min_staff = chinese_numeral_to_int(m.group(2))
        if min_staff:
            constraints.day_minimums.append(DayMinimum(day=DAYS_EN[DAYS_OFFSET[day_ch]], min_staff=min_staff))

    # ── Day maximums ──
    max_m = re.search(r"一天最多(?:只)?(?:需)?([一二兩三四五六七八九十\d]+)人?", text)
    if max_m:
        max_staff = chinese_numeral_to_int(max_m.group(1))
        if max_staff:
            constraints.day_maximums.append(DayMaximum(max_staff=max_staff))

    # ── Mutual exclusions ──
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

    # ── Per-employee max shifts ──
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

    # ── New-employee max shifts ──
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
    """After LLM extraction, fill gaps the small model often misses."""
    norm = text.replace("禮拜", "周").replace("週", "周")
    from .solver import build_unavailability_map, DAYS as _DAYS
    unavail_map = build_unavailability_map(constraints)

    for emp in employees:
        if emp.name not in norm:
            continue

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
                    u = Unavailability(employee_id=emp.id, start_day=DAYS_EN[i], reason="出國", type="vacation")
                    constraints.unavailabilities.append(u)
                    unavail_map[(emp.id, day_enum)] = u
        else:
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
                        u = Unavailability(employee_id=emp.id, start_day=DAYS_EN[i], reason="出國", type="vacation")
                        constraints.unavailabilities.append(u)
                        unavail_map[(emp.id, day_enum)] = u

            leave = re.search(
                rf"{re.escape(emp.name)}周([一二三四五六日])(?:上午|下午)?(?:要)?(?:請事假|請假|請病假)",
                norm,
            )
            if leave:
                day_idx = DAYS_OFFSET.get(leave.group(1), 0)
                day_enum = _DAYS[day_idx]
                if (emp.id, day_enum) not in unavail_map:
                    u = Unavailability(employee_id=emp.id, start_day=DAYS_EN[day_idx], reason="請假", type="leave")
                    constraints.unavailabilities.append(u)
                    unavail_map[(emp.id, day_enum)] = u

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


@traceable(name="groq_llm_call", run_type="llm")
def _call_groq(messages: list[dict], model: str, api_key: str) -> str:
    """單次 Groq HTTP 呼叫，抽離成獨立函式讓 LangSmith 記錄完整 inputs/output。"""
    from requests import post
    payload = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_tokens": 1024,
        "temperature": 0.0,
    }
    resp = post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


@traceable(name="extract_constraints", run_type="chain")
def extract_constraints_from_text(
    text: str, employees: List[Employee], daily_staff_count: Optional[int] = None
) -> ConstraintSet:
    api_key = os.getenv("GROQ_API_KEY")
    # Default upgraded to 70B; override with GROQ_MODEL env var
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    constraints: Optional[ConstraintSet] = None

    if api_key:
        messages = make_messages(text, employees, daily_staff_count)
        last_raw: Optional[str] = None

        for attempt in range(2):
            try:
                last_raw = _call_groq(messages, model, api_key)
                parsed = _extract_json(last_raw)
                constraints = ConstraintSet.model_validate(parsed)
                _supplement_regex(constraints, text, employees)
                break  # success
            except Exception as exc:
                if attempt == 0:
                    # Build a correction message and retry once
                    missing = [k for k in _REQUIRED_KEYS if k not in (last_raw or "")]
                    correction = (
                        f"輸出有誤（{type(exc).__name__}）。"
                        f"請確保 JSON 包含所有必要欄位：{', '.join(_REQUIRED_KEYS)}。"
                        + (f"疑似缺少：{missing}。" if missing else "")
                        + "重新輸出完整 JSON："
                    )
                    messages = messages + (
                        [{"role": "assistant", "content": last_raw}] if last_raw else []
                    ) + [{"role": "user", "content": correction}]
                # attempt 1 failure → fall through to regex

    if constraints is None:
        constraints = parse_simple_constraints(text, employees)

    # Always apply daily_staff_count as per-day minimum
    if daily_staff_count is not None:
        for day in Weekday:
            constraints.day_minimums.append(DayMinimum(day=day, min_staff=daily_staff_count))

    return constraints
