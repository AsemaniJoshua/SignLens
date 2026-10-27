"""Hand landmark normalization and feature engineering.

This is the single source of truth for turning 21 MediaPipe hand landmarks
into model features. Training notebooks AND the inference/API code must both
call `hand_features()` so the model always sees identically processed input.
(The old project computed features separately in five places, and they drifted.)
"""
from __future__ import annotations

import numpy as np

N_LANDMARKS = 21

# MediaPipe hand landmark indices
WRIST = 0
THUMB = (1, 2, 3, 4)
INDEX = (5, 6, 7, 8)
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
FINGERS = (THUMB, INDEX, MIDDLE, RING, PINKY)
TIPS = (4, 8, 12, 16, 20)
MIDDLE_MCP = 9

# Layout of the legacy MediaPipe Holistic vectors saved by the old scripts
#   258  = pose(33*4) + left hand(21*3) + right hand(21*3)
#   1662 = pose(33*4) + face(468*3) + left hand + right hand
_LEGACY_SLICES = {
    258: (slice(132, 195), slice(195, 258)),
    1662: (slice(1536, 1599), slice(1599, 1662)),
}


def hand_from_legacy_vector(vec: np.ndarray):
    """Extract one hand (21, 3) from an old Holistic feature vector.

    Returns (landmarks, handedness) or (None, None) when no hand was detected.
    Prefers the right hand, then the left (the dataset is one-handed).
    """
    vec = np.asarray(vec, dtype=np.float32)
    if vec.shape[-1] not in _LEGACY_SLICES:
        raise ValueError(f"Unknown legacy vector length {vec.shape[-1]}")
    left_sl, right_sl = _LEGACY_SLICES[vec.shape[-1]]
    right, left = vec[right_sl], vec[left_sl]
    if np.any(right):
        return right.reshape(N_LANDMARKS, 3), "Right"
    if np.any(left):
        return left.reshape(N_LANDMARKS, 3), "Left"
    return None, None


def normalize_hand(lm: np.ndarray, handedness: str | None = None,
                   aspect: float = 1.0) -> np.ndarray:
    """Make landmarks independent of position, size and handedness.

    lm         : (21, 3) MediaPipe normalized image landmarks (x, y in 0..1)
    handedness : "Left" / "Right" as reported by MediaPipe. Left hands are
                 mirrored so every sample looks like a right hand.
    aspect     : image width / height. MediaPipe divides x by width and y by
                 height, so x must be stretched by this to keep the hand's
                 true shape (e.g. 16/9 for 1280x720 video, 4/3 for 640x480).

    Returns (21, 3) float32: wrist at origin, palm length (wrist -> middle
    finger knuckle) == 1.  Orientation is intentionally NOT removed, because
    letters like H/U, G/Q and K/P differ mainly by how the hand is rotated.
    """
    p = np.asarray(lm, dtype=np.float32).reshape(N_LANDMARKS, 3).copy()
    p[:, 0] *= aspect
    p[:, 2] *= aspect  # MediaPipe z uses roughly the same scale as x
    p -= p[WRIST]
    if handedness is not None and handedness.lower().startswith("l"):
        p[:, 0] *= -1.0
    palm = np.linalg.norm(p[MIDDLE_MCP, :2])
    if palm < 1e-6:
        palm = np.linalg.norm(p[:, :2], axis=1).max() + 1e-6
    return (p / palm).astype(np.float32)


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def joint_angles(p: np.ndarray) -> np.ndarray:
    """Cosine of the bend at each finger joint (15 values).

    Uses only directions, so it is unaffected by position, scale or camera
    distance. A straight finger gives ~1, a fully curled one goes negative.
    """
    out = []
    for finger in FINGERS:
        chain = (WRIST,) + finger
        for a, b, c in zip(chain[:-2], chain[1:-1], chain[2:]):
            v1, v2 = _unit(p[b] - p[a]), _unit(p[c] - p[b])
            out.append(float(np.dot(v1, v2)))
    return np.asarray(out, dtype=np.float32)


def tip_distances(p: np.ndarray) -> np.ndarray:
    """Distances between all fingertip pairs (10) and from the thumb tip to
    every other landmark (20). Thumb placement is what separates A/S/T/M/N/E."""
    tips = p[list(TIPS)]
    iu = np.triu_indices(5, k=1)
    pair = np.linalg.norm(tips[:, None] - tips[None], axis=-1)[iu]
    others = [i for i in range(N_LANDMARKS) if i != 4]
    thumb = np.linalg.norm(p[others] - p[4], axis=-1)
    return np.concatenate([pair, thumb]).astype(np.float32)


# The wrist is the origin (always 0, 0, 0), so it is left out of the coordinates:
# a feature that never changes carries no information and breaks feature scaling.
FEATURE_DIM = (N_LANDMARKS - 1) * 3 + 15 + 10 + 20  # = 105


def hand_features(lm: np.ndarray, handedness: str | None = None,
                  aspect: float = 1.0, already_normalized: bool = False) -> np.ndarray:
    """Full feature vector (105,) for one hand. Use this everywhere."""
    p = lm if already_normalized else normalize_hand(lm, handedness, aspect)
    return np.concatenate([p[1:].reshape(-1), joint_angles(p), tip_distances(p)]).astype(np.float32)


def scaling_stats(X: np.ndarray, min_std: float = 1e-2):
    """Mean and variance for the model's input scaling layer, with a floor on
    the spread. Without the floor, a feature that barely varies in the data
    gets divided by ~0 and produces enormous inputs (the v2.0.0 training bug)."""
    X = np.asarray(X, np.float64).reshape(-1, np.shape(X)[-1])
    return X.mean(0).astype(np.float32), np.maximum(X.var(0), min_std ** 2).astype(np.float32)
