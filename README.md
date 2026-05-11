# AI Agent 智慧排班系統 MVP

## 簡介
此專案包含 FastAPI 後端範例，實作：
- 自然語言約束抽取提示模板
- OR-Tools 排班求解器
- 排班衝突分析
- API 端點整合流程

## 安裝
```bash
cd c:\Users\USER\Desktop\ai_agent\backend
python -m pip install -r requirements.txt
```

## 啟動
```bash
uvicorn backend.app:app --reload
```

## 範例請求
POST `/schedule`
```json
{
  "text": "A周二開始要出國三天，B周三上午要請事假，這周三排班至少要有三個人"
}
```

此 API 將回傳安排結果或衝突原因。

## 注意
- 若要使用 Groq API，可設定環境變數 `GROQ_API_KEY` 及 `GROQ_MODEL`。
- 若未設定 LLM API，系統會使用內建簡單解析器。

## 前端介面
前端專案位於 `frontend/`，提供自然語言輸入與排班結果顯示。

安裝前端依賴：
```bash
cd c:\Users\USER\Desktop\ai_agent\frontend
npm install
```

啟動前端：
```bash
npm run dev
```

預設會向 `http://localhost:8000/schedule` 發送排班請求。
