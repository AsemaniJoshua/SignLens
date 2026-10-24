"""Geometric augmentation on normalized hand landmarks.

Applied ONLY to training samples, on the fly, AFTER the train/val/test split.
(The old project added noise first and split afterwards, so near-copies of
training samples sat in the test set and the reported accuracy was fake.)
"""
from __future__ import annotations

import numpy as np

from .landmarks import FINGERS, N_LANDMARKS, WRIST


def _rot(axis: str, deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float32)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)


def scale_fingers(p: np.ndarray, rng, spread: float = 0.08) -> np.ndarray:
    """Lengthen/shorten each finger separately: different people's hands."""
    p = p.copy()
    for finger in FINGERS:
        k = 1.0 + rng.uniform(-spread, spread)
        base = p[finger[0]].copy()
        p[list(finger)] = base + (p[list(finger)] - base) * k
    return p


def augment_hand(p: np.ndarray, rng: np.random.Generator,
                 rot_z: float = 15.0, rot_xy: float = 20.0,
                 jitter: float = 0.02, mirror_p: float = 0.5,
                 finger_spread: float = 0.08) -> np.ndarray:
    """Random realistic variation of one normalized (21, 3) hand.

    rot_z   : in-plane tilt of the hand, degrees
    rot_xy  : camera viewing-angle change, degrees
    jitter  : landmark detection noise, in palm-length units
    mirror_p: chance of mirroring (left-handed signer / selfie camera)
    """
    q = scale_fingers(p, rng, finger_spread)
    R = (_rot("z", rng.uniform(-rot_z, rot_z))
         @ _rot("y", rng.uniform(-rot_xy, rot_xy))
         @ _rot("x", rng.uniform(-rot_xy, rot_xy)))
    q = (q - q[WRIST]) @ R.T
    q = q * rng.uniform(0.95, 1.05)
    q = q + rng.normal(0.0, jitter, q.shape).astype(np.float32)
    if rng.random() < mirror_p:
        q[:, 0] *= -1.0
    return q.astype(np.float32)


def augment_sequence(seq: np.ndarray, rng: np.random.Generator, **kw) -> np.ndarray:
    """Augment a (T, 21, 3) sequence with ONE consistent random transform for
    all frames (a real signer doesn't change hand size mid-sign), plus small
    per-frame jitter. `seq` holds normalized hands; trajectories are handled
    in sequences.py."""
    jitter = kw.pop("jitter", 0.015)
    sub = np.random.default_rng(rng.integers(1 << 31))
    state = sub.bit_generator.state
    out = []
    for frame in seq:
        sub.bit_generator.state = state  # same transform every frame
        q = augment_hand(frame, sub, jitter=0.0, **kw)
        out.append(q)
    out = np.stack(out)
    out += rng.normal(0.0, jitter, out.shape).astype(np.float32)
    return out.reshape(len(seq), N_LANDMARKS, 3)
