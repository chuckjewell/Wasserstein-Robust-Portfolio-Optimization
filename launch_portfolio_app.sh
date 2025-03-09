#!/bin/bash

# Portfolio Optimization App Launch Script

# Display banner
echo "========================================================"
echo "  Wasserstein-Robust Portfolio Optimization App Launcher"
echo "========================================================"
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Found virtual environment. Activating..."
    source venv/bin/activate
else
    echo "Virtual environment not found. Creating new one..."
    python -m venv venv
    source venv/bin/activate
    
    echo "Installing required packages..."
    pip install -U streamlit numpy pandas matplotlib scipy cvxpy stockdex
fi

# Check if stockdex is installed
if ! python -c "import stockdex" &> /dev/null; then
    echo "Installing stockdex package..."
    pip install -U stockdex
fi

echo ""
echo "Starting Portfolio Optimization App..."
echo "Press Ctrl+C to stop the app"
echo ""

# Launch the app
streamlit run app_stockdex.py