@echo off
echo ═══════════════════════════════════════
echo   RAG Assistant v2 — Frontend
echo ═══════════════════════════════════════

if not exist node_modules (
    echo Installing dependencies...
    npm install
)

echo Starting React dev server on http://localhost:3000
npm start