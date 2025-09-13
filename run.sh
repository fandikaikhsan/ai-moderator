#!/usr/bin/env bash
# AI Discussion Moderator - Quick Start Script

set -euo pipefail

echo "🤖 AI Discussion Moderator - Quick Start"
echo "========================================"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment (detect shell)
echo "🔧 Activating virtual environment..."
if [ -n "${FISH_VERSION:-}" ]; then
    source .venv/bin/activate.fish
else
    source .venv/bin/activate
fi

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check basic functionality
echo "🧪 Running basic tests..."
python3 test_basic.py

echo ""
echo "🚀 Starting AI Discussion Moderator..."
echo ""
echo "Choose mode:"
echo "1) Basic Camera Test (recommended for first run)"
echo "2) Headless Mode (face detection + mouth tracking)"
echo "3) GUI Mode (may have some features disabled on Python 3.13)"
echo "4) Enhanced Headless Mode (best for macOS without tkinter)"
echo ""

read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "Starting basic camera test..."
        python3 test_basic.py
        ;;
    2)
        echo "Starting headless mode (press 'q' to quit)..."
        python3 main.py --no-gui
        ;;
    3)
        echo "Starting GUI mode..."
        echo "Note: Some advanced analytics may be limited on Python 3.13"
        python3 main.py || {
            echo ""
            echo "⚠️  GUI mode failed (tkinter not available)"
            echo "💡 This is common on macOS Python installations"
            echo "🔄 Falling back to enhanced headless mode..."
            echo ""
            sleep 2
            python3 main_headless.py
        }
        ;;
    4)
        echo "Starting enhanced headless mode (press 'q' to quit)..."
        echo "📊 Live statistics will be shown in console every 5 seconds"
        python3 main_headless.py
        ;;
    *)
        echo "Invalid choice. Starting basic test..."
        python3 test_basic.py
        ;;
esac

echo ""
echo "✨ Thanks for using AI Discussion Moderator!"
echo ""
echo "💡 TO FIX GUI MODE (tkinter issue):"
echo "   For Homebrew Python: brew install python-tk"
echo "   For pyenv Python: Install Python with tkinter support"
echo "   Alternative: Use option 4 (Enhanced Headless Mode)"
