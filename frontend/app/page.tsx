"use client";

import React, { useState } from 'react';

interface Employee {
  id: string;
  name: string;
  role: 'regular' | 'new';
}

interface ScheduleResponse {
  // 小時模式：{day: {hour_str: emp_ids[]}}
  assignments: Record<string, Record<string, string[]>>;
  explanation: string;
  ai_understanding?: string;
  conflict_reasons?: string[];
  business_start_hour?: number;
  business_end_hour?: number;
  employee_hours?: Record<string, number>;
}

const DEFAULT_EMPLOYEES: Employee[] = [
  { id: 'A', name: 'A', role: 'regular' },
  { id: 'B', name: 'B', role: 'regular' },
  { id: 'C', name: 'C', role: 'regular' },
  { id: 'D', name: 'D', role: 'regular' },
  { id: 'E', name: 'E', role: 'regular' },
  { id: 'F', name: 'F', role: 'regular' },
  { id: 'G', name: 'G', role: 'regular' },
  { id: 'H', name: 'H', role: 'regular' },
  { id: 'I', name: 'I', role: 'new' },
  { id: 'J', name: 'J', role: 'new' },
];

// 動態顏色調色盤，依員工在清單中的索引循環使用
const COLOR_PALETTE = [
  '#3b82f6', '#10b981', '#f59e0b', '#ef4444',
  '#8b5cf6', '#06b6d4', '#f97316', '#ec4899',
  '#14b8a6', '#a855f7', '#84cc16', '#f43f5e',
  '#0ea5e9', '#d946ef', '#22c55e', '#fb923c',
];

const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
const DAY_ZH: Record<string, string> = {
  Monday: '周一', Tuesday: '周二', Wednesday: '周三', Thursday: '周四',
  Friday: '周五', Saturday: '周六', Sunday: '周日',
};
const DAY_EN: Record<string, string> = {
  Monday: 'Mon', Tuesday: 'Tue', Wednesday: 'Wed', Thursday: 'Thu',
  Friday: 'Fri', Saturday: 'Sat', Sunday: 'Sun',
};

export default function Home() {
  const [employees, setEmployees] = useState<Employee[]>(
    DEFAULT_EMPLOYEES.map(e => ({ ...e }))
  );
  const [selectedIds, setSelectedIds] = useState<Set<string>>(
    new Set(DEFAULT_EMPLOYEES.map(e => e.id))
  );
  const [text, setText] = useState('');
  const [dailyStaffCount, setDailyStaffCount] = useState<number | ''>('');
  // 上下班時間（整點）
  const [startHour, setStartHour] = useState(9);
  const [endHour, setEndHour] = useState(18);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ScheduleResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const toggle = (id: string) => setSelectedIds(prev => {
    const next = new Set(prev);
    next.has(id) ? next.delete(id) : next.add(id);
    return next;
  });

  const updateName = (id: string, name: string) => {
    setEmployees(prev => prev.map(e => e.id === id ? { ...e, name } : e));
  };

  const nameMap = Object.fromEntries(employees.map(e => [e.id, e.name]));

  // 依員工在清單中的順序取色
  const getEmpColor = (id: string): string => {
    const idx = employees.findIndex(e => e.id === id);
    return COLOR_PALETTE[(idx >= 0 ? idx : 0) % COLOR_PALETTE.length];
  };

  const handleSubmit = async (e: React.SyntheticEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    setResult(null);
    const selectedEmployees = employees.filter(emp => selectedIds.has(emp.id));
    const body: Record<string, unknown> = {
      text,
      employees: selectedEmployees,
      business_start_hour: startHour,
      business_end_hour: endHour,
    };
    if (dailyStaffCount !== '') body.daily_staff_count = dailyStaffCount;
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const res = await fetch(`${apiUrl}/schedule`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`API 錯誤：${res.status}`);
      setResult(await res.json());
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : '未知錯誤');
    } finally {
      setLoading(false);
    }
  };

  // 計算結果的小時列表
  const resultHours = result?.business_start_hour !== undefined && result?.business_end_hour !== undefined
    ? Array.from({ length: result.business_end_hour - result.business_start_hour },
        (_, i) => result.business_start_hour! + i)
    : [];

  // 計算總班次數（人·時）
  const totalPersonHours = result?.employee_hours
    ? Object.values(result.employee_hours).reduce((a, b) => a + b, 0)
    : 0;

  const hasAssignments = result && Object.keys(result.assignments).length > 0;

  return (
    <div className="app">
      <header className="header">
        <h1>智慧排班系統</h1>
      </header>

      <main className="main">
        {/* ── 左側欄 ── */}
        <aside className="sidebar">
          <section className="panel">
            <h2 className="panel-title">參與人員</h2>
            <p className="emp-hint">點名稱可修改；點其他處勾選</p>

            {(['regular', 'new'] as const).map(role => (
              <div key={role} className="emp-group">
                <p className="emp-group-label">{role === 'regular' ? '正職員工' : '新進員工'}</p>
                <div className="emp-list">
                  {employees.filter(e => e.role === role).map((emp, _ri) => {
                    const color = getEmpColor(emp.id);
                    return (
                      <div
                        key={emp.id}
                        className={`emp-chip ${selectedIds.has(emp.id) ? 'selected' : ''}`}
                        style={{ '--c': color } as React.CSSProperties}
                        onClick={() => toggle(emp.id)}
                      >
                        <span className="emp-avatar">{emp.id}</span>
                        <input
                          className="emp-name-input"
                          value={emp.name}
                          onChange={ev => updateName(emp.id, ev.target.value)}
                          onClick={ev => ev.stopPropagation()}
                          placeholder={emp.id}
                          maxLength={10}
                        />
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}

            <p className="emp-count">
              已選 <strong>{selectedIds.size}</strong> / {employees.length} 人
            </p>
          </section>

          <section className="panel">
            <h2 className="panel-title">排班設定</h2>
            <form onSubmit={handleSubmit} className="form">
              {/* 上下班時間 */}
              <div className="field">
                <label>上班時段</label>
                <div className="hour-range-row">
                  <input
                    type="number" min={0} max={23}
                    value={startHour}
                    onChange={e => setStartHour(Number(e.target.value))}
                    className="hour-input"
                  />
                  <span className="hour-sep">:00 到</span>
                  <input
                    type="number" min={1} max={24}
                    value={endHour}
                    onChange={e => setEndHour(Number(e.target.value))}
                    className="hour-input"
                  />
                  <span className="hour-sep">:00</span>
                </div>
                <span className="field-hint">
                  共 {Math.max(0, endHour - startHour)} 小時 / 天
                </span>
              </div>

              <div className="field">
                <label>每小時最低人數（選填）</label>
                <input
                  type="number" min={1} value={dailyStaffCount}
                  onChange={e => setDailyStaffCount(e.target.value === '' ? '' : Number(e.target.value))}
                  placeholder="例：2"
                />
              </div>
              <div className="field">
                <label>需求描述</label>
                <textarea
                  value={text} onChange={e => setText(e.target.value)} rows={5}
                  placeholder="例：小明周二開始出國三天，小華周三請事假，周三至少三人…"
                />
              </div>
              <button type="submit" disabled={loading || selectedIds.size === 0}>
                {loading ? <><span className="btn-spinner" />排班中…</> : '產生班表'}
              </button>
            </form>
          </section>
        </aside>

        {/* ── 右側內容 ── */}
        <div className="content">
          {!result && !error && !loading && (
            <div className="empty-state">
              <span className="empty-icon">📅</span>
              <p>選好人員並輸入需求後，點擊「產生班表」</p>
            </div>
          )}

          {loading && (
            <div className="loading-state">
              <div className="big-spinner" />
              <p>AI 正在分析排班需求…</p>
            </div>
          )}

          {error && (
            <section className="panel error-panel">
              <h2>發生錯誤</h2>
              <p>{error}</p>
            </section>
          )}

          {result && (
            <>
              {result.ai_understanding && (
                <div className="ai-understanding">
                  <span className="ai-understanding-icon">💡</span>
                  <span><strong>AI 理解：</strong>{result.ai_understanding}</span>
                </div>
              )}

              {hasAssignments ? (
                <>
                  {/* ── 小時時間軸班表 ── */}
                  <section className="panel">
                    <div className="result-header">
                      <h2 className="panel-title" style={{ margin: 0 }}>本週班表</h2>
                      <span className="result-badge">
                        共 {totalPersonHours} 人·時
                      </span>
                    </div>

                    <div className="hourly-grid-wrapper">
                      <div
                        className="hourly-grid"
                        style={{ gridTemplateColumns: `52px repeat(7, 1fr)` }}
                      >
                        {/* 標題列 */}
                        <div className="hgrid-corner" />
                        {DAYS.map(day => {
                          const isWeekend = day === 'Saturday' || day === 'Sunday';
                          return (
                            <div key={day} className={`hgrid-day-head${isWeekend ? ' weekend' : ''}`}>
                              <span className="cal-zh">{DAY_ZH[day]}</span>
                              <span className="cal-en">{DAY_EN[day]}</span>
                            </div>
                          );
                        })}

                        {/* 每小時一列 */}
                        {resultHours.map(h => (
                          <React.Fragment key={h}>
                            <div className="hgrid-time">
                              {String(h).padStart(2, '0')}:00
                            </div>
                            {DAYS.map(day => {
                              const isWeekend = day === 'Saturday' || day === 'Sunday';
                              const dayData = result.assignments[day] as Record<string, string[]> | undefined;
                              const workers = dayData?.[String(h)] ?? [];
                              return (
                                <div
                                  key={`${day}-${h}`}
                                  className={`hgrid-cell${isWeekend ? ' weekend' : ''}${workers.length === 0 ? ' empty' : ''}`}
                                >
                                  {workers.map(id => (
                                    <span
                                      key={id}
                                      className="hgrid-tag"
                                      style={{
                                        background: getEmpColor(id) + '25',
                                        borderColor: getEmpColor(id),
                                        color: getEmpColor(id),
                                      }}
                                    >
                                      {nameMap[id] || id}
                                    </span>
                                  ))}
                                </div>
                              );
                            })}
                          </React.Fragment>
                        ))}
                      </div>
                    </div>
                  </section>

                  {/* ── 員工時數統計 ── */}
                  {result.employee_hours && (
                    <section className="panel">
                      <h2 className="panel-title">員工時數統計</h2>
                      <div className="emp-hours-list">
                        {Object.entries(result.employee_hours)
                          .sort(([, a], [, b]) => b - a)
                          .map(([id, hrs]) => {
                            const maxHrs = Math.max(...Object.values(result.employee_hours!));
                            const color = getEmpColor(id);
                            return (
                              <div key={id} className="emp-hours-row">
                                <span
                                  className="emp-hours-avatar"
                                  style={{ background: color }}
                                >
                                  {id}
                                </span>
                                <span className="emp-hours-name">{nameMap[id] || id}</span>
                                <div className="emp-hours-bar-wrap">
                                  <div
                                    className="emp-hours-bar"
                                    style={{
                                      width: `${maxHrs > 0 ? (hrs / maxHrs) * 100 : 0}%`,
                                      background: color,
                                    }}
                                  />
                                </div>
                                <span className="emp-hours-val">{hrs}h</span>
                              </div>
                            );
                          })}
                      </div>
                    </section>
                  )}
                </>
              ) : (
                <section className="panel conflict-panel">
                  <h2 className="panel-title">無法產生班表</h2>
                  <ul className="conflict-list">
                    {(result.conflict_reasons ?? [result.explanation]).map((r, i) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                </section>
              )}

              {hasAssignments && result.conflict_reasons && result.conflict_reasons.length > 0 && (
                <section className="panel conflict-panel">
                  <h2 className="panel-title">驗證警告</h2>
                  <ul className="conflict-list">
                    {result.conflict_reasons.map((r, i) => <li key={i}>{r}</li>)}
                  </ul>
                </section>
              )}
            </>
          )}
        </div>
      </main>
    </div>
  );
}
