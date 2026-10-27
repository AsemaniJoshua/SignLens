"""Dataset loading, the common record format, and leak-free splitting.

Every static sample is stored as one record:
    lm      (21, 3)  raw MediaPipe image landmarks
    handed  "Left" / "Right"
    aspect  image width / height
    label   class name, e.g. "A"
    group   id of the recording it came from (e.g. "yt:A_Front", "cam:B/07")
    source  which dataset it came from (e.g. "youtube", "webcam", "kaggle")

`group` is what makes evaluation honest: consecutive video frames are almost
identical, so ALL frames of one recording must land on the same side of the
train/test split.
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter

import numpy as np

from .landmarks import hand_from_legacy_vector
from .sequences import build_sequence

# Classes are always sorted the same way and saved next to the model.
# (The old API hard-coded a 38-name list while the model had 33 outputs.)
DIGITS = [str(i) for i in range(11)]
LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
MOTION_LETTERS = ["J", "Z"]


def sort_labels(labels) -> list:
    """Letters A-Z first, then digits numerically, then anything else."""
    def key(s):
        if s in LETTERS:
            return (0, LETTERS.index(s), s)
        if s.isdigit():
            return (1, int(s), s)
        return (2, 0, s)
    return sorted(set(labels), key=key)


class StaticSet:
    """A simple column store of static hand samples."""

    def __init__(self, lm=None, handed=None, aspect=None, label=None, group=None, source=None):
        self.lm = np.zeros((0, 21, 3), np.float32) if lm is None else np.asarray(lm, np.float32)
        self.handed = np.asarray([] if handed is None else handed, dtype=object)
        self.aspect = np.asarray([] if aspect is None else aspect, dtype=np.float32)
        self.label = np.asarray([] if label is None else label, dtype=object)
        self.group = np.asarray([] if group is None else group, dtype=object)
        self.source = np.asarray([] if source is None else source, dtype=object)

    def __len__(self):
        return len(self.label)

    def subset(self, mask_or_idx):
        return StaticSet(self.lm[mask_or_idx], self.handed[mask_or_idx], self.aspect[mask_or_idx],
                         self.label[mask_or_idx], self.group[mask_or_idx], self.source[mask_or_idx])

    @staticmethod
    def concat(sets):
        sets = [s for s in sets if len(s)]
        return StaticSet(np.concatenate([s.lm for s in sets]),
                         np.concatenate([s.handed for s in sets]),
                         np.concatenate([s.aspect for s in sets]),
                         np.concatenate([s.label for s in sets]),
                         np.concatenate([s.group for s in sets]),
                         np.concatenate([s.source for s in sets]))

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path, lm=self.lm, handed=self.handed.astype(str), aspect=self.aspect,
                            label=self.label.astype(str), group=self.group.astype(str),
                            source=self.source.astype(str))

    @staticmethod
    def load(path):
        d = np.load(path, allow_pickle=False)
        return StaticSet(d["lm"], d["handed"].astype(object), d["aspect"], d["label"].astype(object),
                         d["group"].astype(object), d["source"].astype(object))

    def normalized(self) -> np.ndarray:
        """(N, 21, 3) position/size/handedness-normalized hands."""
        from .landmarks import normalize_hand
        return np.stack([normalize_hand(l, h, a) for l, h, a in
                         zip(self.lm, self.handed, self.aspect)]) if len(self) else np.zeros((0, 21, 3), np.float32)

    def summary(self):
        """Print samples and number of distinct recordings per class/source.
        The recording count is what matters: 900 frames from 1 clip is 1 example."""
        by = Counter(zip(self.source, self.label))
        recs = Counter((s, l) for s, l, _ in set(zip(self.source, self.label, self.group)))
        srcs = sorted(set(self.source))
        print(f"{len(self)} samples, {len(set(self.group))} recordings, {len(set(self.label))} classes")
        print("class   " + "  ".join(f"{s[:16]:>16s}" for s in srcs))
        for lab in sort_labels(self.label):
            cells = [f"{by[(s, lab)]:>6d} ({recs[(s, lab)]:>3d} rec)" if by[(s, lab)] else f"{'-':>16s}"
                     for s in srcs]
            print(f"{lab:7s} " + "  ".join(cells))


# --------------------------------------------------------------------------
# Loaders for the data that already exists in the SignLens repo
# --------------------------------------------------------------------------
_FRAME_RE = re.compile(r"^(?P<clip>.+?)_frame_\d+$")


def load_repo_youtube_frames(model_dir: str, aspect: float = 16 / 9) -> StaticSet:
    """model/dataset/static_keypoints_optimized: frames cut from the YouTube
    'ASL Manual Alphabet' video (one signer, a Front and a Side clip per
    letter) plus one reference .png per letter."""
    root = os.path.join(model_dir, "dataset", "static_keypoints_optimized")
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "*", "*.npy"))):
        label = os.path.basename(os.path.dirname(path))
        if label == "Neutral":
            continue  # "no sign" is decided by a rule at inference, not learned
        stem = os.path.splitext(os.path.basename(path))[0]
        m = _FRAME_RE.match(stem)
        group, src, asp = (f"yt:{m.group('clip')}", "youtube", aspect) if m else (f"ref:{stem}", "reference", 1.0)
        lm, handed = hand_from_legacy_vector(np.load(path))
        if lm is not None:
            rows.append((lm, handed, asp, label, group, src))
    return _rows_to_set(rows)


def load_repo_webcam_sequences(model_dir: str, aspect: float = 4 / 3, frame_step: int = 1) -> StaticSet:
    """model/initial build/dataset: 30 webcam recordings x 30 frames per
    letter (A-S without J), recorded with the old collect_data.py.
    Each held-sign frame becomes a static sample; the recording is the group."""
    root = os.path.join(model_dir, "initial build", "dataset")
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "*", "*.npy"))):
        label = os.path.basename(os.path.dirname(path))
        if label == "Neutral":
            continue
        seq = np.load(path)
        rec = os.path.splitext(os.path.basename(path))[0]
        for vec in seq[::frame_step]:
            lm, handed = hand_from_legacy_vector(vec)
            if lm is not None:
                rows.append((lm, handed, aspect, label, f"cam:{label}/{int(rec):02d}", "webcam"))
    return _rows_to_set(rows)


def load_repo_video_sequences(model_dir: str, aspect: float = 16 / 9, seq_len: int = 32):
    """model/dataset/video_keypoints_no_face: the ORIGINAL (non '_aug_')
    sequences only. Returns list of (sample_dict, label, group)."""
    root = os.path.join(model_dir, "dataset", "video_keypoints_no_face")
    out = []
    for path in sorted(glob.glob(os.path.join(root, "*", "*.npy"))):
        if "_aug_" in path:
            continue
        label = os.path.basename(os.path.dirname(path))
        if label == "Neutral":
            continue
        frames, handed = [], None
        for vec in np.load(path):
            lm, h = hand_from_legacy_vector(vec)
            frames.append(lm)
            handed = handed or h
        s = build_sequence(frames, handed, aspect, seq_len)
        if s is not None:
            out.append((s, label, f"vid:{label}/{os.path.basename(path)}"))
    return out


def load_repo_video_frames(model_dir: str, labels=None, aspect: float = 16 / 9,
                           frame_step: int = 2) -> StaticSet:
    """Individual frames of the original repo video sequences as static
    samples (e.g. the digit videos 0-10, which have no photos). The video
    is the group, so its frames never straddle the train/test split."""
    root = os.path.join(model_dir, "dataset", "video_keypoints_no_face")
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "*", "*.npy"))):
        if "_aug_" in path:
            continue
        label = os.path.basename(os.path.dirname(path))
        if label == "Neutral" or (labels is not None and label not in labels):
            continue
        for vec in np.load(path)[::frame_step]:
            lm, handed = hand_from_legacy_vector(vec)
            if lm is not None:
                rows.append((lm, handed, aspect, label, f"vid:{label}/{os.path.basename(path)}", "repo_video"))
    return _rows_to_set(rows)


def _rows_to_set(rows) -> StaticSet:
    if not rows:
        return StaticSet()
    lm, handed, aspect, label, group, source = zip(*rows)
    return StaticSet(np.stack(lm), handed, aspect, label, group, source)


# --------------------------------------------------------------------------
# Leak-free splitting
# --------------------------------------------------------------------------
def grouped_split(labels, groups, test_frac=0.2, val_frac=0.1, seed=42):
    """Split indices into train/val/test so no recording is shared.

    Classes whose recordings are too few to cover every split go to train
    only (and are reported), rather than silently leaking.
    """
    labels, groups = np.asarray(labels), np.asarray(groups)
    rng = np.random.default_rng(seed)
    # A recording can contain frames of two letters (transition frames), so
    # each recording is assigned ONCE, using the letter it mostly contains.
    main_label = {}
    for g in sorted(set(groups)):
        main_label[g] = Counter(labels[groups == g]).most_common(1)[0][0]
    assign = {}
    for lab in sort_labels(main_label.values()):
        g = np.array(sorted(k for k, v in main_label.items() if v == lab))
        rng.shuffle(g)
        n = len(g)
        n_te = max(1, int(round(n * test_frac))) if n >= 3 else 0
        n_va = max(1, int(round(n * val_frac))) if n >= 3 else 0
        for k, gid in enumerate(g):
            assign[gid] = "test" if k < n_te else "val" if k < n_te + n_va else "train"
    where = np.array([assign[g] for g in groups])
    return (np.where(where == "train")[0], np.where(where == "val")[0], np.where(where == "test")[0])


def check_no_leak(groups, *index_sets):
    gs = [set(np.asarray(groups)[ix]) for ix in index_sets]
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            shared = gs[i] & gs[j]
            assert not shared, f"split {i} and {j} share recordings: {list(shared)[:5]}"
    return True


def save_labels(path, labels, extra=None):
    meta = {"labels": list(labels)}
    meta.update(extra or {})
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
