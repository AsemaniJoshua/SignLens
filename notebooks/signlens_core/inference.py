"""One predictor for every consumer: notebooks, local webcam, FastAPI.

Fixes from the audit:
  * labels come from labels.json saved with the model (never hard-coded)
  * features come from the same functions used in training
  * "no hand in view" is a rule, not a guessed class
  * predictions are smoothed over a few frames instead of flickering
  * moving signs (J, Z, digits with motion) go to the sequence model only
    when the hand is actually moving
"""
from __future__ import annotations

import json
import os
from collections import Counter, deque

import numpy as np

from .landmarks import hand_features
from .sequences import build_sequence, sequence_features


def load_labels(path: str) -> list:
    with open(path) as f:
        return json.load(f)["labels"]


def load_predict_fn(model_path: str):
    """Return f(batch_features) -> probabilities, for .keras/.h5 or .tflite."""
    if model_path.endswith(".tflite"):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        it = Interpreter(model_path=model_path)
        it.allocate_tensors()
        inp, out = it.get_input_details()[0], it.get_output_details()[0]

        def f(x):
            x = np.asarray(x, np.float32)
            res = []
            for row in x:  # tflite models are exported with batch size 1
                it.set_tensor(inp["index"], row[None])
                it.invoke()
                res.append(it.get_tensor(out["index"])[0])
            return np.stack(res)
        return f
    import keras
    m = keras.models.load_model(model_path, compile=False)
    return lambda x: m.predict(np.asarray(x, np.float32), verbose=0)


class SignPredictor:
    def __init__(self, static_model: str, static_labels: str,
                 motion_model: str | None = None, motion_labels: str | None = None,
                 min_conf: float = 0.6, smooth: int = 5, motion_window_s: float = 1.5,
                 motion_threshold: float = 1.0, aspect: float = 16 / 9):
        self.static_fn = load_predict_fn(static_model)
        self.static_labels = load_labels(static_labels)
        self.motion_fn = load_predict_fn(motion_model) if motion_model else None
        self.motion_labels = load_labels(motion_labels) if motion_labels else None
        self.min_conf, self.aspect = min_conf, aspect
        self.recent = deque(maxlen=smooth)
        self.buffer = deque()   # (time_s, landmarks or None)
        self.window = motion_window_s
        self.motion_threshold = motion_threshold
        self.handed = None

    # ---- single image --------------------------------------------------
    def predict_landmarks(self, lm, handedness=None, aspect=None):
        if lm is None:
            return {"prediction": None, "confidence": 0.0, "reason": "no hand detected"}
        x = hand_features(lm, handedness, aspect or self.aspect)[None]
        p = self.static_fn(x)[0]
        k = int(np.argmax(p))
        return {"prediction": self.static_labels[k], "confidence": float(p[k]),
                "top3": [(self.static_labels[i], float(p[i])) for i in np.argsort(-p)[:3]]}

    # ---- live stream -----------------------------------------------------
    def update(self, lm, handedness, t_seconds: float, aspect=None):
        """Feed one camera frame's landmarks. Returns the current stable result."""
        aspect = aspect or self.aspect
        if handedness:
            self.handed = handedness
        self.buffer.append((t_seconds, lm))
        while self.buffer and t_seconds - self.buffer[0][0] > self.window:
            self.buffer.popleft()

        if lm is None:
            self.recent.clear()
            return {"prediction": None, "confidence": 0.0, "reason": "no hand detected"}

        result = None
        if self.motion_fn is not None and len(self.buffer) >= 8:
            seq = build_sequence([f for _, f in self.buffer], self.handed, aspect)
            if seq is not None:
                path = np.abs(np.diff(seq["traj"][:, :2], axis=0)).sum()
                if path > self.motion_threshold:
                    p = self.motion_fn(sequence_features(seq)[None])[0]
                    k = int(np.argmax(p))
                    result = {"prediction": self.motion_labels[k], "confidence": float(p[k]), "model": "motion"}
        if result is None:
            result = self.predict_landmarks(lm, handedness, aspect)
            result["model"] = "static"

        if result["confidence"] < self.min_conf:
            self.recent.append(None)
        else:
            self.recent.append(result["prediction"])
        votes = Counter(v for v in self.recent if v is not None)
        if votes:
            label, n = votes.most_common(1)[0]
            if n >= max(2, len(self.recent) // 2 + 1):
                return {**result, "prediction": label, "stable": True}
        return {**result, "stable": False}
