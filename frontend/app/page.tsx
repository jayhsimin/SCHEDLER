"use client";

import { useState } from 'react';

interface ScheduleResponse {
  assignments: Record<string, string[]>;
  explanation: string;
  conflict_reasons?: string[];
}

export default function Home() {
  const [text, setText] = useState('A周二開始要出國三天，B周三上午要請事假，這周三排班至少要有三個人');
  const [dailyStaffCount, setDailyStaffCount] = useState<number | ''>('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ScheduleResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);
    setLoading(true);
    setResult(null);

    try {
      const body: Record<string, unknown> = { text };
      if (dailyStaffCount !== '') body.daily_staff_count = dailyStaffCount;

      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/schedule`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        throw new Error(`API 錯誤：${response.status}`);
      }
      const data = await response.json();
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : '未知錯誤');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container">
      <section className="card">
        <h1>AI Agent 智慧排班系統</h1>
        <p>輸入自然語言需求，系統將呼叫後端排班邏輯並顯示班表結果。</p>

        <form onSubmit={handleSubmit} className="form">
          <label htmlFor="dailyStaffCount">每日排班人數（選填）</label>
          <input
            id="dailyStaffCount"
            type="number"
            min={1}
            value={dailyStaffCount}
            onChange={(e) => setDailyStaffCount(e.target.value === '' ? '' : Number(e.target.value))}
            placeholder="例如：3（每天需排幾人）"
          />
          <label htmlFor="requestText">排班需求</label>
          <textarea
            id="requestText"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={5}
            placeholder="例如：A周二開始要出國三天，B周三上午要請事假，這周三排班至少要有三個人"
          />
          <button type="submit" disabled={loading}>
            {loading ? '排班中…' : '送出排班請求'}
          </button>
        </form>
      </section>

      {error && (
        <section className="card error">
          <h2>錯誤</h2>
          <p>{error}</p>
        </section>
      )}

      {result && (
        <section className="card result">
          <h2>排班結果</h2>
          <p>{result.explanation}</p>

          <div className="grid">
            {Object.entries(result.assignments).map(([day, names]) => (
              <div key={day} className="day-card">
                <strong>{day}</strong>
                <p>{names.length ? names.join(', ') : '無人排班'}</p>
              </div>
            ))}
          </div>

          {result.conflict_reasons && result.conflict_reasons.length > 0 && (
            <div className="conflicts">
              <h3>衝突原因</h3>
              <ul>
                {result.conflict_reasons.map((reason, index) => (
                  <li key={index}>{reason}</li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}
    </main>
  );
}
