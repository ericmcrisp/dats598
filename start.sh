#!/bin/bash
echo "Starting development servers..."
echo ""

# Start backend
echo "Starting backend on http://localhost:8000..."
cd backend
source .venv/bin/activate
python run.py > ../backend.log 2>&1 &  # ← Add this to save logs
BACKEND_PID=$!
cd ..

echo "Waiting for backend to start (this may take 30-60 seconds for ML models to load)..."
COUNTER=0
until curl -s http://localhost:8000 > /dev/null 2>&1; do
    COUNTER=$((COUNTER+1))
    if [ $COUNTER -eq 60 ]; then
        echo ""
        echo "❌ Backend failed to start after 60 seconds"
        echo "Backend log:"
        tail -20 backend.log
        kill $BACKEND_PID 2>/dev/null
        exit 1
    fi
    echo -n "."
    sleep 1
done

echo ""
echo "✅ Backend is ready! (took ${COUNTER} seconds)"

# Start frontend
echo "Starting frontend on http://localhost:5173..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "Both servers started!"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "📱 Frontend: http://localhost:5173"
echo "🔧 Backend: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "✅ Servers stopped"
    exit 0
}

# Trap Ctrl+C
trap cleanup INT TERM

# Wait for both processes
wait