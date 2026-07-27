import time
import os
import json
import logging
import threading
import numpy as np
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import sounddevice as sd

from backend.camera_engine import CameraEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AudioEngine")

ZONES = ["top_left", "top_right", "bottom_left", "bottom_right"]
ZONE_NAMES_HEBREW = {
    "top_left": "שמאל למעלה",
    "top_right": "ימין למעלה",
    "bottom_left": "שמאל למטה",
    "bottom_right": "ימין למטה"
}

MODEL_FILE = os.path.join(os.path.dirname(__file__), "calibration_model.json")

PROFILES = {
    "work": {
        "name": "💼 עבודה ופגישות",
        "top_left": {"single": {"type": "url", "value": "https://google.com"}, "double": {"type": "shortcut", "value": "volume_up"}},
        "top_right": {"single": {"type": "url", "value": "https://slack.com"}, "double": {"type": "shortcut", "value": "mute"}},
        "bottom_left": {"single": {"type": "app", "value": "Calculator"}, "double": {"type": "shortcut", "value": "sleep"}},
        "bottom_right": {"single": {"type": "app", "value": "Notes"}, "double": {"type": "shortcut", "value": "lock_screen"}}
    },
    "media": {
        "name": "🎵 מדיה וסרטונים",
        "top_left": {"single": {"type": "shortcut", "value": "volume_up"}, "double": {"type": "shortcut", "value": "volume_up"}},
        "top_right": {"single": {"type": "url", "value": "https://youtube.com"}, "double": {"type": "shortcut", "value": "mute"}},
        "bottom_left": {"single": {"type": "url", "value": "https://spotify.com"}, "double": {"type": "shortcut", "value": "sleep"}},
        "bottom_right": {"single": {"type": "shortcut", "value": "mute"}, "double": {"type": "shortcut", "value": "lock_screen"}}
    },
    "gaming": {
        "name": "⚡ גיימינג וקיצורים",
        "top_left": {"single": {"type": "shortcut", "value": "screenshot"}, "double": {"type": "shortcut", "value": "volume_up"}},
        "top_right": {"single": {"type": "shortcut", "value": "mute"}, "double": {"type": "shortcut", "value": "mute"}},
        "bottom_left": {"single": {"type": "shortcut", "value": "lock_screen"}, "double": {"type": "shortcut", "value": "sleep"}},
        "bottom_right": {"single": {"type": "shortcut", "value": "sleep"}, "double": {"type": "shortcut", "value": "lock_screen"}}
    }
}

class AudioEngine:
    def __init__(self, sample_rate=44100, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = 2
        self.stream = None
        self.is_running = False
        self.is_listening_active = False

        self.active_profile_key = "work"
        self.camera_engine = CameraEngine()
        self.is_camera_fusion_enabled = False

        # Raised default sensitivity threshold from 0.02 to 0.045 to ignore random work movements
        self.sensitivity_threshold = 0.045
        self.cooldown_sec = 0.15
        self.last_trigger_time = 0

        self.buffer_len = int(sample_rate * 0.5)
        self.audio_buffer = np.zeros((self.buffer_len, 2), dtype=np.float32)

        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
        self.is_calibrated = False

        self.is_calibrating = False
        self.current_calib_zone = None
        self.calib_samples_target = 8
        self.calib_data = {zone: [] for zone in ZONES}
        self.last_calib_tap_time = 0

        self.last_tap_zone = None
        self.last_tap_time = 0
        self.pending_single_timer = None
        self.double_tap_max_interval = 0.45

        self.on_audio_frame = None
        self.on_tap_detected = None
        self.on_calib_progress = None

        self._load_saved_calibration()

    def set_profile(self, profile_key):
        if profile_key in PROFILES:
            self.active_profile_key = profile_key
            logger.info(f"Switched active action profile to: {profile_key}")
            return True, PROFILES[profile_key]
        return False, None

    def toggle_listening(self):
        self.is_listening_active = not self.is_listening_active
        logger.info(f"Engine detection active state changed: {self.is_listening_active}")
        return self.is_listening_active

    def toggle_camera_fusion(self):
        active = self.camera_engine.toggle_camera()
        self.is_camera_fusion_enabled = active
        logger.info(f"Camera vision fusion enabled: {self.is_camera_fusion_enabled}")
        return self.is_camera_fusion_enabled

    def extract_features_stereo(self, audio_slice_stereo):
        if len(audio_slice_stereo) < 128:
            return None, 0

        if audio_slice_stereo.ndim == 2 and audio_slice_stereo.shape[1] >= 2:
            left_ch = audio_slice_stereo[:, 0]
            right_ch = audio_slice_stereo[:, 1]
        else:
            left_ch = audio_slice_stereo.flatten()
            right_ch = left_ch

        rms_left = float(np.sqrt(np.mean(left_ch**2)))
        rms_right = float(np.sqrt(np.mean(right_ch**2)))
        peak_left = float(np.max(np.abs(left_ch)))
        peak_right = float(np.max(np.abs(right_ch)))
        
        peak_mono = max(peak_left, peak_right)
        if peak_mono < 0.005:
            return None, 0

        lr_balance = float((rms_left - rms_right) / (rms_left + rms_right + 1e-6))

        corr = signal.correlate(left_ch, right_ch, mode='full')
        lags = signal.correlation_lags(len(left_ch), len(right_ch), mode='full')
        best_lag = int(lags[np.argmax(corr)])
        tdoa_ms = (best_lag / self.sample_rate) * 1000.0

        mono_ch = (left_ch + right_ch) / 2.0
        norm_slice = mono_ch / (peak_mono + 1e-6)
        abs_slice = np.abs(norm_slice)
        peak_idx = int(np.argmax(abs_slice))
        rise_samples = max(1, peak_idx)
        attack_velocity = float(peak_mono / (rise_samples / self.sample_rate))

        n_fft = min(1024, len(norm_slice))
        fft_vals = np.abs(np.fft.rfft(norm_slice * np.hanning(len(norm_slice)), n=n_fft))
        total_energy = np.sum(fft_vals**2) + 1e-6
        
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)
        low_band = np.sum(fft_vals[freqs < 500]**2)
        mid_band = np.sum(fft_vals[(freqs >= 500) & (freqs < 2500)]**2)
        high_band = np.sum(fft_vals[freqs >= 2500]**2)

        high_low_ratio = float(high_band / (low_band + 1e-6))
        mid_low_ratio = float(mid_band / (low_band + 1e-6))

        mel_edges = np.logspace(np.log10(40), np.log10(self.sample_rate / 2.2), 9, dtype=int)
        mel_energies = []
        for i in range(len(mel_edges) - 1):
            f_low, f_high = mel_edges[i], mel_edges[i+1]
            idx_low = np.searchsorted(freqs, f_low)
            idx_high = np.searchsorted(freqs, f_high)
            b_energy = float(np.sum(fft_vals[idx_low:idx_high]**2) / total_energy)
            mel_energies.append(b_energy)

        centroid = float(np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-6))
        crest_factor = float(peak_mono / (((rms_left + rms_right)/2.0) + 1e-6))
        zcr = float(np.sum(np.abs(np.diff(np.sign(norm_slice)))) / (2 * len(norm_slice)))

        quality_score = int(min(100, max(50, (crest_factor * 12) + (high_low_ratio * 15))))

        features = [
            lr_balance,
            tdoa_ms,
            attack_velocity,
            high_low_ratio,
            mid_low_ratio,
            centroid,
            crest_factor,
            zcr
        ] + mel_energies

        return np.array(features, dtype=np.float32), quality_score

    def extract_features(self, audio_slice):
        feat, _ = self.extract_features_stereo(audio_slice)
        return feat

    def _fallback_heuristic_zone(self, features):
        lr_balance = features[0]
        tdoa_ms = features[1]
        high_low = features[3]
        centroid = features[5]

        is_left = (lr_balance > 0.0) or (tdoa_ms < 0)
        is_top = (centroid > 1800) or (high_low > 1.1)

        if is_top and is_left:
            return "top_left"
        elif is_top and not is_left:
            return "top_right"
        elif not is_top and is_left:
            return "bottom_left"
        else:
            return "bottom_right"

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Sounddevice status: {status}")

        if indata.ndim == 1:
            samples_stereo = np.column_stack((indata, indata))
        elif indata.shape[1] == 1:
            samples_stereo = np.column_stack((indata[:, 0], indata[:, 0]))
        else:
            samples_stereo = indata[:, :2]

        self.audio_buffer = np.roll(self.audio_buffer, -len(samples_stereo), axis=0)
        self.audio_buffer[-len(samples_stereo):, :] = samples_stereo

        mono_samples = np.mean(samples_stereo, axis=1)
        rms = float(np.sqrt(np.mean(mono_samples**2)))
        peak = float(np.max(np.abs(mono_samples)))

        if self.on_audio_frame:
            vis_samples = mono_samples[::4].tolist() if self.is_listening_active else [0]*len(mono_samples[::4])
            self.on_audio_frame(vis_samples, rms if self.is_listening_active else 0, peak if self.is_listening_active else 0)

        if not self.is_listening_active:
            return

        now = time.time()
        if peak > self.sensitivity_threshold and (now - self.last_trigger_time) > self.cooldown_sec:
            pre_samples = int(self.sample_rate * 0.04)
            post_samples = int(self.sample_rate * 0.12)
            
            slice_end = len(self.audio_buffer)
            slice_start = max(0, slice_end - (pre_samples + post_samples))
            audio_transient_stereo = self.audio_buffer[slice_start:slice_end, :]

            features, quality = self.extract_features_stereo(audio_transient_stereo)
            if features is not None:
                self.last_trigger_time = now
                self._handle_tap(features, quality)

    def _handle_tap(self, features, quality=90):
        now = time.time()

        if self.is_calibrating and self.current_calib_zone:
            if (now - self.last_calib_tap_time) < 0.25:
                return

            self.last_calib_tap_time = now
            self.calib_data[self.current_calib_zone].append(features.tolist())
            count = len(self.calib_data[self.current_calib_zone])
            logger.info(f"Calibrated tap for {self.current_calib_zone}: {count}/{self.calib_samples_target} (Quality={quality}%)")

            if self.on_calib_progress:
                self.on_calib_progress(self.current_calib_zone, count, self.calib_samples_target, quality)

        else:
            try:
                predicted_zone = None
                confidence = 0.70

                if self.is_calibrated:
                    try:
                        scaled_features = self.scaler.transform([features])
                        probs = self.classifier.predict_proba(scaled_features)[0]
                        max_idx = int(np.argmax(probs))
                        confidence = float(probs[max_idx])
                        predicted_zone = str(self.classifier.classes_[max_idx])
                    except Exception as err:
                        logger.warning(f"Calibrated model transform error ({err}). Resetting calibration model.")
                        self.is_calibrated = False
                        if os.path.exists(MODEL_FILE):
                            os.remove(MODEL_FILE)
                        predicted_zone = self._fallback_heuristic_zone(features)
                        confidence = 0.75
                else:
                    predicted_zone = self._fallback_heuristic_zone(features)
                    confidence = 0.75

                cam_quadrant = self.camera_engine.get_active_quadrant() if self.is_camera_fusion_enabled else None
                if cam_quadrant:
                    predicted_zone = cam_quadrant
                    confidence = 0.98

                logger.info(f"Stereo Tap spatial prediction: Zone={predicted_zone}, Conf={confidence:.2f}")

                time_since_last = now - self.last_tap_time

                if (self.last_tap_zone == predicted_zone) and (time_since_last < self.double_tap_max_interval):
                    logger.info(f"⚡ DOUBLE TAP detected on zone: {predicted_zone}")
                    
                    if self.pending_single_timer:
                        self.pending_single_timer.cancel()
                        self.pending_single_timer = None

                    self.last_tap_zone = None
                    self.last_tap_time = 0

                    if self.on_tap_detected:
                        self.on_tap_detected(predicted_zone, confidence, "double")

                else:
                    self.last_tap_zone = predicted_zone
                    self.last_tap_time = now

                    if self.pending_single_timer:
                        self.pending_single_timer.cancel()

                    def trigger_single():
                        logger.info(f"☝️ SINGLE TAP confirmed on zone: {predicted_zone}")
                        if self.on_tap_detected:
                            self.on_tap_detected(predicted_zone, confidence, "single")

                    self.pending_single_timer = threading.Timer(0.30, trigger_single)
                    self.pending_single_timer.start()

            except Exception as e:
                logger.error(f"Prediction error: {e}")

    def start_calibration(self, zone):
        if zone not in ZONES:
            return False, "Invalid zone"
        self.is_calibrating = True
        self.current_calib_zone = zone
        self.calib_data[zone] = []
        self.last_calib_tap_time = 0
        return True, f"Calibration started for {zone}"

    def finish_calibration(self):
        X, y = [], []
        for zone in ZONES:
            samples = self.calib_data.get(zone, [])
            if len(samples) < 2:
                return False, f"Not enough samples for zone {zone}. Need at least 2 taps per zone."
            for feat in samples:
                X.append(feat)
                y.append(zone)

        X = np.array(X)
        y = np.array(y)

        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.classifier.fit(X_scaled, y)
        self.is_calibrated = True
        self.is_calibrating = False
        self.current_calib_zone = None

        self._save_calibration()
        return True, "Calibration successful and saved!"

    def _save_calibration(self):
        data = {
            "calib_data": self.calib_data
        }
        with open(MODEL_FILE, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved calibration data to {MODEL_FILE}")

    def _load_saved_calibration(self):
        if not os.path.exists(MODEL_FILE):
            return
        try:
            with open(MODEL_FILE, "r") as f:
                data = json.load(f)
            self.calib_data = data.get("calib_data", {})
            X, y = [], []
            for zone, samples in self.calib_data.items():
                for feat in samples:
                    X.append(feat)
                    y.append(zone)

            if len(X) >= 8:
                X = np.array(X)
                y = np.array(y)

                self.scaler = StandardScaler()
                self.classifier = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)

                self.scaler.fit(X)
                X_scaled = self.scaler.transform(X)
                self.classifier.fit(X_scaled, y)
                self.is_calibrated = True
                logger.info("Loaded previous calibration model successfully.")
        except Exception as e:
            logger.error(f"Failed to load saved model: {e}")

    def start_stream(self):
        if self.is_running:
            return
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=2,
                dtype='float32',
                callback=self.audio_callback
            )
            self.stream.start()
            self.is_running = True
            logger.info("Stereo audio stream started (channels=2).")
        except Exception as e:
            logger.warning(f"Stereo InputStream failed ({e}), falling back to Mono (channels=1)...")
            try:
                self.stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.chunk_size,
                    channels=1,
                    dtype='float32',
                    callback=self.audio_callback
                )
                self.stream.start()
                self.is_running = True
                logger.info("Mono audio stream started.")
            except Exception as ex:
                logger.error(f"Error starting audio stream: {ex}")
                raise ex

    def stop_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_running = False
        if self.is_camera_fusion_enabled:
            self.camera_engine.stop_camera()
        logger.info("Audio stream stopped.")
