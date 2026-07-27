import os
os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

import cv2
import time
import base64
import logging
import threading
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CameraEngine")

ZONES = ["top_left", "top_right", "bottom_left", "bottom_right"]

class CameraEngine:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
        self.is_camera_enabled = False
        self.thread = None

        self.current_frame_base64 = None
        self.active_quadrant = None
        self.quadrant_scores = {"top_left": 0, "top_right": 0, "bottom_left": 0, "bottom_right": 0}
        self.prev_frame = None

        # Esc Key Anchor Coordinates (normalized 0.0 - 1.0)
        self.esc_anchor_x = 0.25
        self.esc_anchor_y = 0.35
        self.is_esc_calibrated = False

        # Camera Calibration Wizard State
        self.is_calibrating_cam = False
        self.current_cam_zone = "esc_anchor"
        self.cam_hold_start_time = None
        self.required_hold_sec = 3.0
        self.on_cam_calib_progress = None

    def start_camera(self):
        if self.is_running:
            return True
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.warning(f"Could not open camera index {self.camera_index}")
                return False
            
            self.is_running = True
            self.is_camera_enabled = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            logger.info("Camera capture engine started.")
            return True
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            return False

    def stop_camera(self):
        self.is_running = False
        self.is_camera_enabled = False
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("Camera capture engine stopped.")

    def toggle_camera(self):
        if self.is_camera_enabled:
            self.stop_camera()
            return False
        else:
            return self.start_camera()

    def start_esc_key_calibration(self):
        self.is_calibrating_cam = True
        self.current_cam_zone = "esc_anchor"
        self.cam_hold_start_time = None
        logger.info("Started Esc-key anchor calibration. Waiting for finger at Esc key for 3s...")
        return True

    def set_esc_anchor_point(self, norm_x, norm_y):
        self.esc_anchor_x = norm_x
        self.esc_anchor_y = norm_y
        self.is_esc_calibrated = True
        logger.info(f"🟢 Esc Key Anchor Calibrated at X={norm_x:.2f}, Y={norm_y:.2f}")

    def _capture_loop(self):
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            small_frame = cv2.resize(frame, (240, 180))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            if self.prev_frame is None:
                self.prev_frame = gray
                continue

            diff = cv2.absdiff(self.prev_frame, gray)
            _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
            self.prev_frame = gray

            # Use Esc Key Anchor to split frame into 4 spatial zones
            mid_x = int(self.esc_anchor_x * 240) + 40
            mid_y = int(self.esc_anchor_y * 180) + 30

            mid_x = max(40, min(200, mid_x))
            mid_y = max(30, min(150, mid_y))

            scores = {
                "top_left": int(np.sum(thresh[0:mid_y, 0:mid_x])),
                "top_right": int(np.sum(thresh[0:mid_y, mid_x:240])),
                "bottom_left": int(np.sum(thresh[mid_y:180, 0:mid_x])),
                "bottom_right": int(np.sum(thresh[mid_y:180, mid_x:240]))
            }

            self.quadrant_scores = scores
            max_zone = max(scores, key=scores.get)
            motion_val = scores[max_zone]

            if motion_val > 1500:
                self.active_quadrant = max_zone
            else:
                self.active_quadrant = None

            # Esc Key 3-Second Hold Calibration
            if self.is_calibrating_cam:
                now = time.time()
                # Find motion centroid
                M = cv2.moments(thresh)
                if M["m00"] > 500:
                    cx = float(M["m10"] / M["m00"]) / 240.0
                    cy = float(M["m01"] / M["m00"]) / 180.0

                    if self.cam_hold_start_time is None:
                        self.cam_hold_start_time = now

                    elapsed = now - self.cam_hold_start_time
                    remaining = max(0.0, self.required_hold_sec - elapsed)
                    progress_pct = min(100.0, (elapsed / self.required_hold_sec) * 100.0)

                    if self.on_cam_calib_progress:
                        self.on_cam_calib_progress("esc_anchor", round(remaining, 1), round(progress_pct, 1))

                    if elapsed >= self.required_hold_sec:
                        self.set_esc_anchor_point(cx, cy)
                        self.is_calibrating_cam = False
                        if self.on_cam_calib_progress:
                            self.on_cam_calib_progress("esc_anchor", 0.0, 100.0, finished=True)
                else:
                    self.cam_hold_start_time = None
                    if self.on_cam_calib_progress:
                        self.on_cam_calib_progress("esc_anchor", 3.0, 0.0, reset=True)

            # Draw visual grid overlay & Esc Anchor point
            vis_frame = frame.copy()
            anchor_px = int(self.esc_anchor_x * w)
            anchor_py = int(self.esc_anchor_y * h)

            cv2.circle(vis_frame, (anchor_px, anchor_py), 8, (0, 255, 0), -1)
            cv2.putText(vis_frame, "ESC Anchor", (anchor_px + 12, anchor_py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw grid lines from Esc anchor
            cv2.line(vis_frame, (anchor_px + 40, 0), (anchor_px + 40, h), (255, 255, 255), 1)
            cv2.line(vis_frame, (0, anchor_py + 30), (w, anchor_py + 30), (255, 255, 255), 1)

            small_vis = cv2.resize(vis_frame, (200, 150))
            _, buffer = cv2.imencode('.jpg', small_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            self.current_frame_base64 = base64.b64encode(buffer).decode('utf-8')

            time.sleep(0.04)

    def get_current_frame_base64(self):
        return self.current_frame_base64

    def get_active_quadrant(self):
        return self.active_quadrant

    def get_quadrant_scores(self):
        return self.quadrant_scores
