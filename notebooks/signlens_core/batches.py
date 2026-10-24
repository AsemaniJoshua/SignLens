"""Feature batches for training (framework-independent, so it is testable).

A Keras PyDataset in the notebooks just calls these. Augmentation happens
per batch, so every epoch sees fresh variations of the training samples.
"""
from __future__ import annotations

import numpy as np

from .augment import augment_hand
from .landmarks import hand_features
from .sequences import augment_seq_sample, sequence_features


def static_batch(P: np.ndarray, idx, rng=None, augment: bool = False) -> np.ndarray:
    """P: (N, 21, 3) normalized hands. Returns (len(idx), FEATURE_DIM)."""
    if augment:
        return np.stack([hand_features(augment_hand(P[i], rng), already_normalized=True) for i in idx])
    return np.stack([hand_features(P[i], already_normalized=True) for i in idx])


def sequence_batch(samples: list, idx, rng=None, augment: bool = False) -> np.ndarray:
    """samples: list of build_sequence() dicts. Returns (len(idx), SEQ_LEN, SEQ_FEATURE_DIM)."""
    if augment:
        return np.stack([sequence_features(augment_seq_sample(samples[i], rng)) for i in idx])
    return np.stack([sequence_features(samples[i]) for i in idx])


def balanced_order(y_idx: np.ndarray, rng, per_class: int | None = None) -> np.ndarray:
    """An epoch's sample order where every class appears equally often.
    Classes with few samples are repeated (with augmentation, so not copies)
    instead of letting big classes dominate."""
    classes = np.unique(y_idx)
    per_class = per_class or int(np.median([np.sum(y_idx == c) for c in classes]))
    order = np.concatenate([rng.choice(np.where(y_idx == c)[0], per_class,
                                       replace=np.sum(y_idx == c) < per_class) for c in classes])
    rng.shuffle(order)
    return order
