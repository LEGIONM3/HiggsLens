#!/usr/bin/env bash
set -e

echo "Starting HiggsLens development environment..."

# Ensure Python virtualenv is active or available via uv
echo "Starting FastAPI backend server on port 8000..."
uv run --python 3.12 uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

echo "Starting Next.js frontend server on port 3000..."
cd frontend
npm run dev &
FRONTEND_PID=$!

cd ..

trap "echo 'Shutting down servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true" SIGINT SIGTERM

echo "Servers running! Backend: http://localhost:8000 | Frontend: http://localhost:3000"
wait
