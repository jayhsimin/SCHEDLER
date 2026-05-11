import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'AI Agent 智慧排班系統',
  description: '自然語言排班輸入與結果顯示',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-Hant">
      <body>{children}</body>
    </html>
  );
}
