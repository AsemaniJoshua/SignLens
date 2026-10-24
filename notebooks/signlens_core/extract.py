"""MediaPipe hand landmark extraction (Tasks API).

The old code used `mp.solutions.holistic`, which no longer exists in current
MediaPipe releases (0.10.3x ships only the Tasks API), so it cannot run on a
fresh Colab. This module uses `HandLandmarker` instead, and only extracts the
hand: the 132 body-pose numbers the old models relied on described where the
signer stood, not the letter.

Policy: always feed frames exactly as the camera produced them (never flip
them) in training and at inference. Left/right is then reported consistently,
and mirror augmentation covers left-handed signers and selfie cameras.
"""
from __future__ import annotations

import os
import re
import urllib.request

import cv2
import numpy as np

from .data import LETTERS, StaticSet
from .sequences import build_sequence

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp")


def download_model(path: str = "hand_landmarker.task") -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class HandExtractor:
    """Wraps MediaPipe HandLandmarker for still images or video files.

        with HandExtractor("hand_landmarker.task") as ex:
            lm, handed, score = ex.from_image(cv2.imread("a.jpg"))
    """

    def __init__(self, model_path: str, video: bool = False, min_conf: float = 0.5):
        from mediapipe.tasks.python import BaseOptions, vision
        mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path), running_mode=mode,
            num_hands=1, min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf, min_tracking_confidence=min_conf)
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self.video = video

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._lm.close()

    @staticmethod
    def _to_mp(bgr):
        import mediapipe as mp
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    @staticmethod
    def _parse(result):
        if not result.hand_landmarks:
            return None, None, 0.0
        lm = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], np.float32)
        cat = result.handedness[0][0] if result.handedness and result.handedness[0] else None
        return lm, (cat.category_name if cat else None), (float(cat.score) if cat else 0.0)

    def from_image(self, bgr):
        """Returns (lm (21,3) or None, handedness, score)."""
        return self._parse(self._lm.detect(self._to_mp(bgr)))

    def from_frame(self, bgr, timestamp_ms: int):
        """For live/video use (VIDEO mode). Timestamps must increase."""
        return self._parse(self._lm.detect_for_video(self._to_mp(bgr), int(timestamp_ms)))


def read_video_landmarks(path: str, model_path: str, max_seconds: float = 6.0):
    """Run the hand tracker over a video file.

    Returns dict(frames=[(21,3) or None], times=[s], handed, aspect, fps)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frames, times, hands = [], [], []
    with HandExtractor(model_path, video=True) as ex:
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok or i / fps > max_seconds:
                break
            lm, handed, _ = ex.from_frame(bgr, int(1000 * i / fps))
            frames.append(lm); times.append(i / fps)
            if handed:
                hands.append(handed)
            i += 1
    cap.release()
    handed = max(set(hands), key=hands.count) if hands else None
    return {"frames": frames, "times": times, "handed": handed,
            "aspect": (w / h) if w and h else 1.0, "fps": fps}


def crop_hand(bgr, lm, size: int = 160, margin: float = 0.35):
    """Square crop around the detected hand, resized to size x size (RGB).

    The image CNN gets only the hand, not the room behind it. The old CNN saw
    the whole frame, so it could learn the background and lighting of the one
    video it was trained on instead of the handshape."""
    h, w = bgr.shape[:2]
    xs, ys = lm[:, 0] * w, lm[:, 1] * h
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + 2 * margin)
    side = max(side, 16)
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + int(round(side)), y0 + int(round(side))
    pad = max(0, -x0, -y0, x1 - w, y1 - h)
    if pad:
        bgr = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    crop = cv2.resize(bgr[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------
# Folder datasets (Kaggle downloads, your own photos)
# --------------------------------------------------------------------------
def letter_folders(root: str, exts=IMAGE_EXTS):
    """Find every folder named like a class (A..Z, 0..10) anywhere under root.
    Works for most Kaggle layouts (train/A, asl_alphabet_train/A, Train_Alphabet/A ...).
    Returns {label: [folder, ...]}."""
    found = {}
    for dirpath, dirnames, filenames in os.walk(root):
        name = os.path.basename(dirpath).strip()
        lab = name.upper()
        if (lab in LETTERS or name.isdigit()) and any(f.lower().endswith(exts) for f in filenames):
            found.setdefault(lab if lab in LETTERS else name, []).append(dirpath)
    return found


_num = re.compile(r"(\d+)")


def default_group(source: str, label: str, folder: str, filename: str, rank: int, bucket: int):
    """Recording id for a dataset image. Many Kaggle alphabet sets are frames
    of a few videos stored in order, so neighbouring files are near-identical:
    every `bucket` consecutive files (in sorted order) count as one recording."""
    return f"{source}:{label}:{os.path.basename(folder)}:{rank // bucket}"


def extract_image_dataset(root: str, model_path: str, source: str, max_per_class: int = 400,
                          bucket: int = 50, group_fn=default_group, seed: int = 0,
                          progress: bool = True) -> StaticSet:
    """Extract one hand per image from a folder-per-letter dataset.

    max_per_class: sample this many images per letter (spread evenly over the
    sorted files so every recording contributes). 300-500 is plenty for
    landmarks; more mostly repeats the same frames.
    """
    rows, missed = [], {}
    folders = letter_folders(root)
    if not folders:
        raise FileNotFoundError(f"No letter folders with images found under {root}")
    with HandExtractor(model_path, video=False) as ex:
        for label in sorted(folders):
            files = []
            for folder in folders[label]:
                fs = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS))
                fs.sort(key=lambda f: [int(t) if t.isdigit() else t for t in _num.split(f)])
                files += [(folder, f, r) for r, f in enumerate(fs)]
            if len(files) > max_per_class:
                pick = np.linspace(0, len(files) - 1, max_per_class).astype(int)
                files = [files[i] for i in pick]
            ok = 0
            for folder, f, rank in files:
                bgr = cv2.imread(os.path.join(folder, f))
                if bgr is None:
                    continue
                lm, handed, _ = ex.from_image(bgr)
                if lm is None:
                    continue
                h, w = bgr.shape[:2]
                rows.append((lm, handed, w / h, label, group_fn(source, label, folder, f, rank, bucket), source))
                ok += 1
            missed[label] = len(files) - ok
            if progress:
                print(f"  {source} {label}: {ok}/{len(files)} images had a detectable hand")
    if not rows:
        return StaticSet()
    lm, handed, aspect, label, group, src = zip(*rows)
    return StaticSet(np.stack(lm), handed, aspect, label, group, src)


def extract_video_dataset(root: str, model_path: str, motion_labels=("J", "Z"), source: str = "my_videos",
                          static_step: int = 3, trim: float = 0.15, max_seconds: float = 8.0,
                          progress: bool = True):
    """Your own phone/webcam videos, one folder per sign: root/A/*.mp4, root/J/*.mp4 ...

    * motion signs (J, Z, ...) -> one sequence per video for the motion model
    * every other sign        -> frames (every `static_step`-th, skipping the
      first/last `trim` of the clip while the hand moves into place) become
      static samples. The video is the group.
    Returns (StaticSet, sequences list, seq labels, seq groups).
    """
    rows, seqs, seq_labels, seq_groups = [], [], [], []
    folders = letter_folders(root, VIDEO_EXTS)
    for label in sorted(folders):
        for folder in folders[label]:
            for f in sorted(os.listdir(folder)):
                if not f.lower().endswith(VIDEO_EXTS):
                    continue
                v = read_video_landmarks(os.path.join(folder, f), model_path, max_seconds)
                group = f"{source}:{label}/{f}"
                n = len(v["frames"])
                if label in motion_labels:
                    s = build_sequence(v["frames"], v["handed"], v["aspect"])
                    if s is not None:
                        seqs.append(s); seq_labels.append(label); seq_groups.append(group)
                    status = "sequence" if s is not None else "REJECTED (hand not visible enough)"
                else:
                    a, b = int(n * trim), int(n * (1 - trim))
                    k = 0
                    for lm in v["frames"][a:b:static_step]:
                        if lm is not None:
                            rows.append((lm, v["handed"], v["aspect"], label, group, source)); k += 1
                    status = f"{k} static frames"
                if progress:
                    print(f"  {label}/{f}: {n} frames -> {status}")
    if rows:
        lm, handed, aspect, label, group, src = zip(*rows)
        static = StaticSet(np.stack(lm), handed, aspect, label, group, src)
    else:
        static = StaticSet()
    return static, seqs, seq_labels, seq_groups
