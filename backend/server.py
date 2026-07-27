import os
import json
import asyncio
import logging
import signal
import sys
import threading
import time
from typing import List, Dict, Any, Optional
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, Response
from pydantic import BaseModel

from backend.audio_engine import AudioEngine, ZONES, ZONE_NAMES_HEBREW, PROFILES, NOTE_NAMES_HEBREW
from backend.actions import execute_action

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Server")

if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_FILE = os.path.join(BASE_DIR, "backend", "actions_config.json")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

DEFAULT_ACTIONS = PROFILES["work"]

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.loop = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected ({len(self.active_connections)} active)")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("WebSocket client disconnected")

    async def broadcast(self, data: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.error(f"Error sending websocket message: {e}")

    def broadcast_sync(self, data: dict):
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(data), self.loop)

app = FastAPI(title="Acoustic Desk Buttons")
manager = ConnectionManager()
audio_engine = AudioEngine()

def normalize_actions(data: dict) -> dict:
    normalized = {}
    for zone in ZONES:
        default_z = DEFAULT_ACTIONS.get(zone, {})
        raw_z = data.get(zone, {}) if isinstance(data, dict) else {}

        if "single" in raw_z and "double" in raw_z:
            normalized[zone] = raw_z
        else:
            single_act = {
                "type": raw_z.get("type", default_z.get("single", {}).get("type", "url")),
                "value": raw_z.get("value", default_z.get("single", {}).get("value", "https://google.com"))
            }
            double_act = default_z.get("double", {"type": "shortcut", "value": "mute"})
            normalized[zone] = {
                "name": raw_z.get("name", ZONE_NAMES_HEBREW.get(zone, zone)),
                "single": single_act,
                "double": double_act
            }
    return normalized

def load_actions():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "top_left" in data:
                    return normalize_actions(data)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    return DEFAULT_ACTIONS

def save_actions(actions_data: dict):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(actions_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving config file: {e}")

actions_config = load_actions()

def on_audio_frame(waveform, rms, peak):
    cam_active_quadrant = audio_engine.camera_engine.get_active_quadrant() if audio_engine.is_camera_fusion_enabled else None
    
    manager.broadcast_sync({
        "event": "audio_frame",
        "rms": round(rms, 4),
        "peak": round(peak, 4),
        "waveform": waveform,
        "cam_quadrant": cam_active_quadrant
    })

def on_tap_detected(zone, confidence, mode="single"):
    zone_cfg = actions_config.get(zone, {})
    action = zone_cfg.get(mode, {"type": "none", "value": ""})
    zone_name = ZONE_NAMES_HEBREW.get(zone, zone)
    mode_name = "נקישה כפולה ⚡" if mode == "double" else "נקישה בודדת ☝️"
    
    logger.info(f"Tap Triggered! Zone: {zone_name} ({zone}), Mode: {mode}, Action: {action}")

    manager.broadcast_sync({
        "event": "tap_detected",
        "zone": zone,
        "zone_name": zone_name,
        "mode": mode,
        "mode_name": mode_name,
        "confidence": round(confidence, 2),
        "action": action
    })

    success, msg = execute_action(action)
    logger.info(f"Action Execution Result: {success} -> {msg}")

def on_calib_progress(zone, count, total, quality=90):
    manager.broadcast_sync({
        "event": "calib_progress",
        "zone": zone,
        "count": count,
        "total": total,
        "quality": quality
    })

def on_whistle_note_progress(note, note_name_hebrew, duration, progress_pct, peak_freq):
    manager.broadcast_sync({
        "event": "whistle_progress",
        "note": note,
        "note_name": note_name_hebrew,
        "duration": round(duration, 2),
        "progress_pct": progress_pct,
        "peak_freq": round(peak_freq, 1)
    })

def on_cam_calib_progress(zone, remaining, progress_pct, finished=False, reset=False):
    manager.broadcast_sync({
        "event": "cam_calib_progress",
        "zone": zone,
        "remaining": remaining,
        "progress_pct": progress_pct,
        "finished": finished,
        "reset": reset
    })

audio_engine.on_audio_frame = on_audio_frame
audio_engine.on_tap_detected = on_tap_detected
audio_engine.on_calib_progress = on_calib_progress
audio_engine.on_whistle_note_progress = on_whistle_note_progress
audio_engine.camera_engine.on_cam_calib_progress = on_cam_calib_progress

@app.on_event("startup")
async def startup_event():
    manager.loop = asyncio.get_running_loop()
    try:
        audio_engine.start_stream()
    except Exception as e:
        logger.error(f"Could not auto-start mic stream: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    audio_engine.stop_stream()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_json({
            "event": "init_status",
            "is_calibrated": audio_engine.is_calibrated,
            "is_listening_active": audio_engine.is_listening_active,
            "is_camera_fusion_enabled": audio_engine.is_camera_fusion_enabled,
            "sensitivity": audio_engine.sensitivity_threshold,
            "mode": audio_engine.mode,
            "active_profile": audio_engine.active_profile_key,
            "actions": actions_config
        })
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

class SensitivityUpdate(BaseModel):
    sensitivity: float

class CalibStart(BaseModel):
    zone: str

class CamQuadrantUpdate(BaseModel):
    quadrant: str

class WebAudioTransient(BaseModel):
    samples: List[float]

@app.get("/api/status")
def get_status():
    return {
        "is_running": audio_engine.is_running,
        "is_listening_active": audio_engine.is_listening_active,
        "is_camera_fusion_enabled": audio_engine.is_camera_fusion_enabled,
        "is_calibrated": audio_engine.is_calibrated,
        "sensitivity": audio_engine.sensitivity_threshold,
        "mode": audio_engine.mode,
        "active_profile": audio_engine.active_profile_key,
        "zones": ZONES,
        "actions": actions_config
    }

@app.post("/api/set_mode/{mode}")
def set_mode(mode: str):
    success = audio_engine.set_mode(mode)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid mode")
    manager.broadcast_sync({"event": "mode_switched", "mode": audio_engine.mode})
    return {"status": "success", "mode": audio_engine.mode}

@app.post("/api/profile/{profile_key}")
def switch_profile(profile_key: str):
    global actions_config
    if profile_key in PROFILES:
        actions_config = normalize_actions(PROFILES[profile_key])
        audio_engine.active_profile_key = profile_key
        save_actions(actions_config)
        manager.broadcast_sync({"event": "profile_switched", "active_profile": profile_key, "actions": actions_config})
        return {"status": "success", "active_profile": profile_key, "actions": actions_config}
    raise HTTPException(status_code=400, detail="Invalid profile key")

@app.post("/api/toggle_engine")
def toggle_engine():
    active_state = audio_engine.toggle_listening()
    manager.broadcast_sync({"event": "engine_toggled", "is_listening_active": active_state})
    return {"status": "success", "is_listening_active": active_state}

@app.post("/api/toggle_camera")
def toggle_camera():
    audio_engine.is_camera_fusion_enabled = not audio_engine.is_camera_fusion_enabled
    manager.broadcast_sync({"event": "camera_toggled", "is_camera_fusion_enabled": audio_engine.is_camera_fusion_enabled})
    return {"status": "success", "is_camera_fusion_enabled": audio_engine.is_camera_fusion_enabled}

@app.post("/api/process_web_tap")
def process_web_tap(data: WebAudioTransient):
    if not audio_engine.is_listening_active:
        return {"status": "ignored", "reason": "listening inactive"}
    
    samples = np.array(data.samples, dtype=np.float32)
    
    if audio_engine.mode == "whistle":
        note, zone, peak_freq, purity = audio_engine.detect_whistle_note(samples)
        if zone is not None:
            now = time.time()
            if audio_engine.whistle_current_zone == zone:
                duration = now - audio_engine.whistle_start_time
                progress_pct = min(100, int((duration / audio_engine.whistle_duration_required) * 100))
                on_whistle_note_progress(note, NOTE_NAMES_HEBREW.get(note, note), duration, progress_pct, peak_freq)
                if duration >= audio_engine.whistle_duration_required and (now - audio_engine.last_whistle_trigger_time) > 0.8:
                    audio_engine.last_whistle_trigger_time = now
                    audio_engine.whistle_start_time = None
                    audio_engine.whistle_current_zone = None
                    audio_engine.whistle_current_note = None
                    on_tap_detected(zone, 0.98, "single")
            else:
                audio_engine.whistle_start_time = now
                audio_engine.whistle_current_zone = zone
                audio_engine.whistle_current_note = note
        return {"status": "success"}
    else:
        features, quality = audio_engine.extract_features_stereo(samples)
        if features is not None:
            audio_engine._handle_tap(features, quality)
            return {"status": "success"}
        return {"status": "ignored", "reason": "invalid features"}

@app.post("/api/set_camera_quadrant")
def set_camera_quadrant(update: CamQuadrantUpdate):
    if update.quadrant in ZONES:
        audio_engine.camera_engine.active_quadrant = update.quadrant
    else:
        audio_engine.camera_engine.active_quadrant = None
    return {"status": "success", "quadrant": audio_engine.camera_engine.active_quadrant}

@app.post("/api/shutdown_app")
def shutdown_app():
    logger.info("Shutdown app requested by user. Force exiting process...")
    manager.broadcast_sync({"event": "app_shutting_down"})
    
    def kill_process():
        time.sleep(0.3)
        audio_engine.stop_stream()
        os._exit(0)

    threading.Thread(target=kill_process, daemon=True).start()
    return {"status": "success", "message": "Process terminating immediately..."}

@app.post("/api/actions")
async def update_actions(request: Request):
    global actions_config
    try:
        body = await request.json()
        if "actions" in body and isinstance(body["actions"], dict):
            new_actions = body["actions"]
        elif isinstance(body, dict) and "top_left" in body:
            new_actions = body
        else:
            raise ValueError("Invalid payload structure")

        actions_config = normalize_actions(new_actions)
        save_actions(actions_config)
        logger.info(f"Updated actions configuration successfully: {actions_config}")
        manager.broadcast_sync({"event": "actions_updated", "actions": actions_config})
        return {"status": "success", "actions": actions_config}
    except Exception as e:
        logger.error(f"Failed to update actions: {e}")
        return {"status": "error", "message": str(e), "actions": actions_config}

@app.post("/api/sensitivity")
def set_sensitivity(update: SensitivityUpdate):
    audio_engine.sensitivity_threshold = max(0.005, min(0.5, update.sensitivity))
    manager.broadcast_sync({"event": "sensitivity_updated", "sensitivity": audio_engine.sensitivity_threshold})
    return {"status": "success", "sensitivity": audio_engine.sensitivity_threshold}

@app.post("/api/calibration/start")
def start_calibration(data: CalibStart):
    success, msg = audio_engine.start_calibration(data.zone)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    manager.broadcast_sync({"event": "calib_started", "zone": data.zone})
    return {"status": "success", "message": msg}

@app.post("/api/calibration/finish")
def finish_calibration():
    success, msg = audio_engine.finish_calibration()
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    manager.broadcast_sync({"event": "calib_finished", "is_calibrated": True})
    return {"status": "success", "message": msg}

@app.post("/api/test_action/{zone}/{mode}")
def test_action(zone: str, mode: str = "single"):
    if zone not in actions_config:
        raise HTTPException(status_code=404, detail="Zone not found")
    action = actions_config[zone].get(mode, {})
    success, msg = execute_action(action)
    return {"status": "success" if success else "failed", "message": msg}

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

@app.get("/")
def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        response = FileResponse(index_path)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    return HTMLResponse(f"<h1>Acoustic Desk Buttons Backend Running</h1><p>Frontend path: {index_path}</p>")

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
