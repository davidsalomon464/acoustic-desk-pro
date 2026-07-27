#!/bin/bash
echo "========================================================"
echo " 📦 Building Standalone Desktop Application (PyInstaller)"
echo "========================================================"

# Make sure we're in project directory
CD_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$CD_PATH"

./venv/bin/pyinstaller --noconfirm --windowed \
    --name "AcousticDesk" \
    --add-data "frontend:frontend" \
    --add-data "backend:backend" \
    desktop_app.py

echo "========================================================"
echo " ✅ Build Complete!"
echo " Output App Location:"
echo " $CD_PATH/dist/AcousticDesk.app"
echo "========================================================"
