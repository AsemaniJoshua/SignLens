"""SignLens v2 prediction service (no web framework code, so it is easy to test).

Loads the models trained by notebooks 02 and 03 ONCE and serves:
  * predict_image(bytes)   - one photo
  * predict_video(bytes)   - an uploaded clip (static letters by voting over frames,
                             J/Z via the motion model when the hand really moves)
  * StreamSession.push()   - live camera frames over the WebSocket

All landmark processing comes from `signlens_core`, the same code used in
training, and labels come from the *_labels.json files saved with each model.

Model files (copy them from Google Drive: MyDrive/SignLens/models/) go in
model/model_endpoints/saved_models_v2/:
    static_letters_model.tflite  static_letters_labels.json   (required)
    static_digits_model.tflite   static_digits_labels.json    (optional, mode=digits)
    motion_model.tflite          motion_labels.json           (optional, J/Z)
"""
from __future__ import annotations

import base64
import os
import tempfile
import threading
import time
from collections import Counter

import cv2
import numpy as np

from signlens_core import extract
from signlens_core.inference import SignPredictor, load_labels, load_predict_fn
from signlens_core.landmarks import hand_features
from signlens_core.sequences import build_sequence, sequence_features

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODELS_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "saved_models_v2"))

NO_HAND_MSG = "No hand detected. Make sure your whole hand is visible and well lit."
UNSURE_MSG = "Not sure which sign this is. Please hold the sign clearly and try again."
MODEL_NAME = "SignLens-v2"


class ModelsNotReady(RuntimeError):
    pass


def _env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


def decode_image(file_bytes: bytes):
    """Raw image bytes, or a base64 string / data URL (the old API accepted both)."""
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        try:
            text = file_bytes.decode("utf-8")
            if "base64," in text:
                text = text.split("base64,", 1)[1]
            img = cv2.imdecode(np.frombuffer(base64.b64decode(text), np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            img = None
    if img is None:
        raise ValueError("Could not read the image. Send a JPEG/PNG file or a base64 string.")
    return img


class SignLensService:
    """Loads everything once. Thread-safe: model calls are serialized with a lock
    (TFLite interpreters and MediaPipe landmarkers must not be used concurrently)."""

    def __init__(self, models_dir: str | None = None, hand_model_path: str | None = None,
                 min_conf: float | None = None):
        self.models_dir = models_dir or os.getenv("SIGNLENS_MODELS_DIR", DEFAULT_MODELS_DIR)
        self.min_conf = min_conf if min_conf is not None else _env_float("SIGNLENS_MIN_CONFIDENCE", 0.6)
        self.lock = threading.RLock()
        self.static = {}        # mode -> (predict_fn, labels)
        self.motion = None      # (predict_fn, labels) or None

        letters = self._load("static_letters")
        if letters is None:
            raise ModelsNotReady(
                f"static_letters_model.tflite (or .keras) and static_letters_labels.json not found in "
                f"{self.models_dir}. Copy them from Google Drive (MyDrive/SignLens/models/).")
        self.static["letters"] = letters
        digits = self._load("static_digits")
        if digits is not None:
            self.static["digits"] = digits
        self.motion = self._load("motion")

        hand_model = hand_model_path or os.getenv("SIGNLENS_HAND_MODEL",
                                                  os.path.join(self.models_dir, "hand_landmarker.task"))
        self.hand_model = extract.download_model(hand_model)
        self.image_extractor = extract.HandExtractor(self.hand_model, video=False)

    def _load(self, name):
        labels = os.path.join(self.models_dir, f"{name}_labels.json")
        for ext in (".tflite", ".keras"):
            model = os.path.join(self.models_dir, f"{name}_model{ext}")
            if os.path.exists(model) and os.path.exists(labels):
                return load_predict_fn(model), load_labels(labels)
        return None

    # ------------------------------------------------------------------ info
    def status(self) -> dict:
        return {"modes": sorted(self.static), "motion_model": self.motion is not None,
                "motion_labels": [l for l in (self.motion[1] if self.motion else []) if l != "other"],
                "models_dir": self.models_dir, "min_confidence": self.min_conf}

    def _static(self, mode):
        if mode not in self.static:
            raise ValueError(f"Mode '{mode}' is not available. Available: {sorted(self.static)}")
        return self.static[mode]

    @staticmethod
    def _result(prediction, confidence, top3=None, reason=None, model_used=MODEL_NAME, mode="letters"):
        out = {"prediction": prediction, "confidence": float(confidence), "fallback": False,
               "model_used": model_used, "mode": mode, "top3": top3 or []}
        if reason:
            out["reason"] = reason
        return out

    def _from_probs(self, probs, labels, mode, model_used=MODEL_NAME):
        k = int(np.argmax(probs))
        top3 = [[labels[i], round(float(probs[i]), 4)] for i in np.argsort(-probs)[:3]]
        if probs[k] < self.min_conf:
            return self._result(UNSURE_MSG, probs[k], top3, "low confidence", model_used, mode)
        return self._result(labels[k], probs[k], top3, None, model_used, mode)

    # ----------------------------------------------------------------- image
    def predict_image(self, file_bytes: bytes, mode: str = "letters") -> dict:
        fn, labels = self._static(mode)
        img = decode_image(file_bytes)
        h, w = img.shape[:2]
        with self.lock:
            lm, handed, _ = self.image_extractor.from_image(img)
            if lm is None:
                return self._result(NO_HAND_MSG, 0.0, reason="no hand detected", mode=mode)
            probs = fn(hand_features(lm, handed, w / h)[None])[0]
        return self._from_probs(probs, labels, mode)

    # ----------------------------------------------------------------- video
    def predict_video(self, file_bytes: bytes, mode: str = "letters", max_seconds: float = 10.0) -> dict:
        fn, labels = self._static(mode)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(file_bytes)
            path = f.name
        try:
            # read_video_landmarks makes its own landmarker; the lock keeps model use serialized
            with self.lock:
                v = extract.read_video_landmarks(path, self.hand_model, max_seconds)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        frames, handed, aspect = v["frames"], v["handed"], v["aspect"]
        seen = [lm for lm in frames if lm is not None]
        if not frames:
            raise ValueError("Could not read the video file.")
        if len(seen) < max(3, 0.2 * len(frames)):
            return self._result(NO_HAND_MSG, 0.0, reason="no hand detected", mode=mode)

        # 1) a moving sign (J, Z) - only when the hand really travels
        if mode == "letters" and self.motion is not None:
            seq = build_sequence(frames, handed, aspect)
            if seq is not None and np.abs(np.diff(seq["traj"][:, :2], axis=0)).sum() > 1.0:
                m_fn, m_labels = self.motion
                with self.lock:
                    mp = m_fn(sequence_features(seq)[None])[0]
                k = int(np.argmax(mp))
                if m_labels[k] != "other" and mp[k] >= self.min_conf:
                    return self._from_probs(mp, m_labels, mode, model_used=MODEL_NAME + "-motion")

        # 2) a held sign: average the static model over the steady middle of the clip
        a, b = int(len(frames) * 0.15), max(int(len(frames) * 0.85), int(len(frames) * 0.15) + 1)
        middle = [lm for lm in frames[a:b] if lm is not None] or seen
        X = np.stack([hand_features(lm, handed, aspect) for lm in middle])
        with self.lock:
            probs = fn(X).mean(0)
        return self._from_probs(probs, labels, mode)

    # ---------------------------------------------------------------- stream
    def stream(self, mode: str = "letters") -> "StreamSession":
        return StreamSession(self, mode)


class StreamSession:
    """One live camera connection: its own tracker and smoothing, shared models."""

    def __init__(self, service: SignLensService, mode: str = "letters"):
        fn, labels = service._static(mode)
        motion = service.motion if mode == "letters" else None
        self.service, self.mode = service, mode
        self.extractor = extract.HandExtractor(service.hand_model, video=True)
        self.predictor = SignPredictor(fn, labels,
                                       motion_model=motion[0] if motion else None,
                                       motion_labels=motion[1] if motion else None,
                                       min_conf=service.min_conf)
        self.t0 = time.monotonic()
        self.last_ms = -1

    def push(self, frame_bytes: bytes, t_seconds: float | None = None) -> dict:
        img = decode_image(frame_bytes)
        h, w = img.shape[:2]
        t = (time.monotonic() - self.t0) if t_seconds is None else t_seconds
        ms = max(self.last_ms + 1, int(t * 1000))     # MediaPipe needs increasing timestamps
        self.last_ms = ms
        with self.service.lock:
            lm, handed, _ = self.extractor.from_frame(img, ms)
            r = self.predictor.update(lm, handed, ms / 1000.0, aspect=w / h)
        if r["prediction"] is None:
            return {"status": NO_HAND_MSG, "prediction": None, "confidence": 0.0, "stable": False}
        model_used = MODEL_NAME + ("-motion" if r.get("model") == "motion" else "")
        out = {"prediction": r["prediction"], "confidence": float(r["confidence"]),
               "stable": bool(r.get("stable")), "model_used": model_used, "mode": self.mode}
        if r["confidence"] < self.service.min_conf:
            out["status"] = UNSURE_MSG
        return out

    def close(self):
        self.extractor.close()


# Shared, lazily created instance (the API starts even if models are missing,
# and the health check reports what is wrong).
_service = None
_service_error = None
_service_lock = threading.Lock()


def get_service() -> SignLensService:
    global _service, _service_error
    with _service_lock:
        if _service is None:
            try:
                _service = SignLensService()
                _service_error = None
            except Exception as e:  # noqa: BLE001
                _service_error = str(e)
                raise
    return _service


def service_status() -> dict:
    try:
        return {"ready": True, **get_service().status()}
    except Exception:  # noqa: BLE001
        return {"ready": False, "error": _service_error}
