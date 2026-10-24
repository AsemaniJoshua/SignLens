"""Turning per-frame hand landmarks into fixed-length sequences.

Two problems in the old video pipeline are fixed here:
  1. Training squeezed a whole clip into 30 frames, while live inference
     used the last 30 camera frames (~1 s). The model saw signs at a
     different speed in production than in training. Now BOTH sides call
     `build_sequence()`, which resamples by time.
  2. Frames with no detected hand were saved as all-zeros and treated as
     real data. Here they are interpolated from neighbours, and a clip is
     rejected if too few frames contain a hand.
"""
from __future__ import annotations

import numpy as np

from .augment import augment_sequence, _rot
from .landmarks import FEATURE_DIM, MIDDLE_MCP, WRIST, hand_features, normalize_hand

SEQ_LEN = 32          # frames per sequence after resampling
SEQ_FEATURE_DIM = FEATURE_DIM + 6   # hand shape + wrist position + velocity


def _interp_fill(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linearly fill invalid rows (along axis 0) from valid neighbours."""
    idx = np.arange(len(arr))
    good = idx[valid]
    flat = arr.reshape(len(arr), -1).copy()
    for j in range(flat.shape[1]):
        flat[~valid, j] = np.interp(idx[~valid], good, flat[valid, j])
    return flat.reshape(arr.shape)


def resample(arr: np.ndarray, n: int) -> np.ndarray:
    """Resample along time (axis 0) to exactly n steps by linear interpolation."""
    t_old = np.linspace(0.0, 1.0, len(arr))
    t_new = np.linspace(0.0, 1.0, n)
    flat = arr.reshape(len(arr), -1)
    out = np.stack([np.interp(t_new, t_old, flat[:, j]) for j in range(flat.shape[1])], axis=1)
    return out.reshape((n,) + arr.shape[1:]).astype(np.float32)


def build_sequence(frames, handedness=None, aspect: float = 1.0,
                   seq_len: int = SEQ_LEN, min_valid: float = 0.5):
    """frames: list of (21, 3) raw landmarks, or None where no hand was seen.

    Returns dict(hands=(seq_len, 21, 3) normalized shapes,
                 traj=(seq_len, 3) wrist path in palm units), or None when
    fewer than `min_valid` of the frames had a hand.
    """
    n = len(frames)
    valid = np.array([f is not None for f in frames])
    if n < 2 or valid.mean() < min_valid or valid.sum() < 2:
        return None
    raw = np.stack([np.asarray(f, np.float32).reshape(21, 3) if f is not None
                    else np.zeros((21, 3), np.float32) for f in frames])
    raw = _interp_fill(raw, valid)

    hands = np.stack([normalize_hand(f, handedness, aspect) for f in raw])

    # Wrist trajectory (this is what carries the motion of J and Z)
    xyz = raw.copy()
    xyz[:, :, 0] *= aspect
    xyz[:, :, 2] *= aspect
    palm = np.median(np.linalg.norm(xyz[:, MIDDLE_MCP, :2] - xyz[:, WRIST, :2], axis=1)) + 1e-6
    traj = (xyz[:, WRIST] - xyz[0, WRIST]) / palm
    traj[:, 2] = 0.0  # MediaPipe wrist depth is unreliable
    if handedness is not None and str(handedness).lower().startswith("l"):
        traj[:, 0] *= -1.0

    return {"hands": resample(hands, seq_len), "traj": resample(traj, seq_len)}


def time_warp(arr: np.ndarray, rng, strength: float = 0.2) -> np.ndarray:
    """Random speed changes within the sign (faster start, slower end, ...)."""
    n = len(arr)
    knots = np.sort(rng.uniform(0, 1, 2))
    ctrl_x = np.r_[0.0, knots, 1.0]
    ctrl_y = np.clip(ctrl_x + rng.uniform(-strength, strength, 4) * [0, 1, 1, 0], 0, 1)
    ctrl_y = np.maximum.accumulate(ctrl_y)
    t_new = np.interp(np.linspace(0, 1, n), ctrl_x, ctrl_y)
    t_old = np.linspace(0, 1, n)
    flat = arr.reshape(n, -1)
    out = np.stack([np.interp(t_new, t_old, flat[:, j]) for j in range(flat.shape[1])], axis=1)
    return out.reshape(arr.shape).astype(np.float32)


def augment_seq_sample(sample: dict, rng) -> dict:
    hands, traj = sample["hands"], sample["traj"].copy()
    hands = augment_sequence(hands, rng, mirror_p=0.0)
    traj = traj @ _rot("z", rng.uniform(-15, 15)).T * rng.uniform(0.8, 1.2)
    if rng.random() < 0.5:                 # mirror hand AND path together
        hands = hands.copy(); hands[:, :, 0] *= -1
        traj[:, 0] *= -1
    stacked = np.concatenate([hands.reshape(len(hands), -1), traj], axis=1)
    stacked = time_warp(stacked, rng)
    return {"hands": stacked[:, :63].reshape(-1, 21, 3), "traj": stacked[:, 63:]}


def synthetic_other_motion(P_static: np.ndarray, rng, n: int, seq_len: int = SEQ_LEN) -> list:
    """Generated 'moving, but not a motion sign' clips for the reject class.

    Without them a J/Z model calls EVERY movement J or Z (e.g. the hand
    moving into position for the next letter). Each clip morphs between
    two real handshapes while the wrist moves along a straight or gently
    curved path, at random speeds and distances.
    """
    out = []
    t = np.linspace(0, 1, seq_len)[:, None]
    for _ in range(n):
        a, b = P_static[rng.integers(len(P_static))], P_static[rng.integers(len(P_static))]
        ease = 0.5 - 0.5 * np.cos(np.pi * np.clip((t - rng.uniform(0, 0.4)) / rng.uniform(0.3, 0.6), 0, 1))
        hands = a[None] * (1 - ease[:, :, None]) + b[None] * ease[:, :, None]
        direction = rng.normal(size=2); direction /= np.linalg.norm(direction) + 1e-8
        dist = rng.uniform(0.0, 3.0)
        bend = rng.normal(0, 0.5) * np.sin(np.pi * t)
        traj = np.zeros((seq_len, 3), np.float32)
        traj[:, :2] = dist * t * direction + bend * np.array([-direction[1], direction[0]])
        out.append({"hands": hands.astype(np.float32), "traj": traj})
    return out


def save_sequences(path, samples, labels, groups, sources):
    np.savez_compressed(path, hands=np.stack([s["hands"] for s in samples]),
                        traj=np.stack([s["traj"] for s in samples]),
                        label=np.asarray(labels, str), group=np.asarray(groups, str),
                        source=np.asarray(sources, str))


def load_sequences(path):
    """Returns (samples, labels, groups, sources)."""
    d = np.load(path, allow_pickle=False)
    samples = [{"hands": h, "traj": t} for h, t in zip(d["hands"], d["traj"])]
    return samples, d["label"].astype(object), d["group"].astype(object), d["source"].astype(object)


def sequence_features(sample: dict) -> np.ndarray:
    """(seq_len, SEQ_FEATURE_DIM) model input for one sequence."""
    hands, traj = sample["hands"], sample["traj"]
    shape = np.stack([hand_features(h, already_normalized=True) for h in hands])
    vel = np.diff(traj, axis=0, prepend=traj[:1])
    return np.concatenate([shape, traj, vel], axis=1).astype(np.float32)
