"""Honest evaluation helpers (framework-independent)."""
from __future__ import annotations

from collections import Counter

import numpy as np


def report(y_true, y_pred, labels=None, title: str = "", show_classes: bool = True) -> dict:
    """Accuracy, macro-F1 and per-class recall/precision."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    acc = float(np.mean(y_true == y_pred)) if len(y_true) else float("nan")
    rows, f1s = [], []
    for lab in labels:
        tp = np.sum((y_true == lab) & (y_pred == lab))
        n_true, n_pred = np.sum(y_true == lab), np.sum(y_pred == lab)
        if n_true == 0:
            continue
        rec = tp / n_true
        prec = tp / n_pred if n_pred else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        rows.append((lab, n_true, rec, prec))
    macro_f1 = float(np.mean(f1s)) if f1s else float("nan")
    if title:
        print(f"== {title}")
    print(f"accuracy {acc:.1%}   macro-F1 {macro_f1:.3f}   ({len(y_true)} samples, {len(rows)} classes)")
    if show_classes and rows:
        weak = sorted(rows, key=lambda r: r[2])[:6]
        print("weakest classes (recall):", ", ".join(f"{l} {r:.0%}" for l, _, r, _ in weak))
        conf = Counter((t, p) for t, p in zip(y_true, y_pred) if t != p)
        if conf:
            print("most common mistakes (true->pred):",
                  ", ".join(f"{t}->{p} x{n}" for (t, p), n in conf.most_common(6)))
    return {"accuracy": acc, "macro_f1": macro_f1,
            "per_class_recall": {l: float(r) for l, _, r, _ in rows}}


def confusion(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[t], idx[p]] += 1
    return m


def plot_confusion(y_true, y_pred, labels, title="Confusion matrix (rows = true, normalized)"):
    import matplotlib.pyplot as plt
    m = confusion(y_true, y_pred, labels).astype(float)
    m = m / np.maximum(m.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(0.35 * len(labels) + 3, 0.35 * len(labels) + 2.5))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    return fig


def leave_one_source_out(sources, labels, fit_predict, min_classes: int = 3,
                         min_samples: int = 30, skip=()) -> dict:
    """The real-world test: train on every source except one, test on that one.

    `fit_predict(train_idx, test_idx) -> predicted labels for test_idx`.
    A source is a different signer/dataset/camera, so this measures how well
    the model works for a NEW person, which is what the app actually faces.
    Only classes that also exist in the training sources are scored.
    """
    sources, labels = np.asarray(sources), np.asarray(labels)
    results = {}
    for src in sorted(set(sources)):
        if src in skip:
            continue
        te = np.where(sources == src)[0]
        tr = np.where(sources != src)[0]
        known = set(labels[tr])
        te = te[np.isin(labels[te], list(known))]
        if len(te) < min_samples or len(set(labels[te])) < min_classes:
            print(f"-- skip {src}: too few samples/classes shared with the other sources")
            continue
        pred = np.asarray(fit_predict(tr, te))
        results[src] = report(labels[te], pred, title=f"held-out source: {src}", show_classes=True)
    if results:
        mean = np.mean([r["accuracy"] for r in results.values()])
        print(f"\nMean accuracy on unseen sources: {mean:.1%}")
    return results
