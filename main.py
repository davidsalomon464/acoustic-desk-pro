import os
import sys
import time
import subprocess
import uvicorn

from backend.server import app

def launch_browser():
    """Single window launcher check"""
    pass

if __name__ == "__main__":
    print("=" * 60)
    print("  🔊 Acoustic Desk Virtual Buttons (שולחן עבודה אקוסטי)")
    print("  Starting server at http://localhost:8000 ...")
    print("=" * 60)
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
