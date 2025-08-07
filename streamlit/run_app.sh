#!/bin/bash

# INTEGRATE Rejection Sampling Web Interface Launcher
# This script sets up the environment and starts the Streamlit application

echo "🧮 Starting INTEGRATE Rejection Sampling Web Interface..."

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing requirements..."
    pip install -r requirements.txt
fi

# Check if integrate module is available
python -c "import integrate" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ INTEGRATE module not found. Please install it first:"
    echo "   cd .. && pip install -e ."
    exit 1
fi

# Set the Python path to include parent directory
export PYTHONPATH="${PYTHONPATH}:$(dirname $(pwd))"

echo "✅ Environment ready"
echo "🚀 Starting Streamlit application..."
echo "📡 The application will open in your browser automatically"
echo "🔗 If it doesn't open, navigate to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Start streamlit application
streamlit run streamlit_app.py