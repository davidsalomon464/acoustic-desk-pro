import os
import sys
import time
import socket
import threading
import uvicorn
import webview
import urllib.request

if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
    os.chdir(sys._MEIPASS)
    sys.path.insert(0, sys._MEIPASS)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FRONTEND_INDEX = os.path.join(BASE_DIR, "frontend", "index.html")
PORT = 8000

def run_server():
    try:
        from backend.server import app
        config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="info", install_signal_handlers=False)
        server = uvicorn.Server(config)
        server.run()
    except Exception as e:
        print(f"Uvicorn server error: {e}")

if __name__ == "__main__":
    # Start server in daemon thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Wait until HTTP server is live and responding to requests
    server_url = f"http://127.0.0.1:{PORT}"
    server_ready = False
    
    for _ in range(40):  # Try for up to 4 seconds
        try:
            with urllib.request.urlopen(f"{server_url}/api/status", timeout=0.5) as resp:
                if resp.status == 200:
                    server_ready = True
                    print("FastAPI Backend Server is LIVE and ready!")
                    break
        except Exception:
            time.sleep(0.1)

    # Open PyWebView Desktop Window pointing to the live local server URL
    window = webview.create_window(
        title="שולחן עבודה אקוסטי - Acoustic Desk",
        url=server_url,
        width=1120,
        height=760,
        resizable=True,
        min_size=(900, 600)
    )
    
    webview.start()
