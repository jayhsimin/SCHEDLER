"use client";

import { useState } from 'react';

interface Employee {
  id: string;
  name: string;
  role: 'regular' | 'new';
}

interface ScheduleResponse {
  assignments: Record<string, string[]>;
  explanation: string;
  conflict_reasons?: string[];
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

const EMP_COLORS: Record<string, string> = {
  A: '#3b82f6', B: '#10b981', C: '#f59e0b', D: '#ef4444',
  E: '#8b5cf6', F: '#06b6d4', G: '#f97316', H: '#ec4899',
  I: '#14b8a6', J: '#a855f7',
};

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

  const handleSubmit = async (e: React.SyntheticEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    setResult(null);
    const selectedEmployees = employees.filter(emp => selectedIds.has(emp.id));
    const body: Record<string, unknown> = { text, employees: selectedEmployees };
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

  return (
    <div className="app">
      <header className="header">
        <h1>AI Agent 智慧排班系統</h1>
        <p>以自然語言描述需求，AI 自動生成最佳班表</p>
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
                  {employees.filter(e => e.role === role).map(emp => (
                    <div
                      key={emp.id}
                      className={`emp-chip ${selectedIds.has(emp.id) ? 'selected' : ''}`}
                      style={{ '--c': EMP_COLORS[emp.id] } as React.CSSProperties}
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
                  ))}
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
              <div className="field">
                <label>每日排班人數（選填）</label>
                <input
                  type="number" min={1} value={dailyStaffCount}
                  onChange={e => setDailyStaffCount(e.target.value === '' ? '' : Number(e.target.value))}
                  placeholder="例：3"
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
              <section className="panel">
                <div className="result-header">
                  <h2 className="panel-title" style={{ margin: 0 }}>本週班表</h2>
                  <span className="result-badge">
                    共 {Object.values(result.assignments).flat().length} 班次
                  </span>
                </div>

                <div className="calendar">
                  {DAYS.map(day => {
                    const ids = result.assignments[day] ?? [];
                    const weekend = day === 'Saturday' || day === 'Sunday';
                    return (
                      <div key={day} className={`cal-col${weekend ? ' weekend' : ''}`}>
                        <div className="cal-head">
                          <span className="cal-zh">{DAY_ZH[day]}</span>
                          <span className="cal-en">{DAY_EN[day]}</span>
                        </div>
                        <div className="cal-body">
                          {ids.length > 0
                            ? ids.map(id => (
                              <div
                                key={id} className="cal-tag"
                                style={{
                                  background: EMP_COLORS[id] + '1a',
                                  borderColor: EMP_COLORS[id],
                                  color: EMP_COLORS[id],
                                }}
                              >
                                <span
                                  className="cal-dot"
                                  style={{ background: EMP_COLORS[id] }}
                                >{id}</span>
                                {nameMap[id] || id}
                              </div>
                            ))
                            : <span className="cal-rest">休</span>
                          }
                        </div>
                        <div className="cal-foot">{ids.length} 人</div>
                      </div>
                    );
                  })}
                </div>
              </section>

              {result.conflict_reasons && result.conflict_reasons.length > 0 && (
                <section className="panel conflict-panel">
                  <h2 className="panel-title">衝突說明</h2>
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
