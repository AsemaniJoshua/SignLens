"""Run:  python notebooks/tests/test_core.py   (or pytest notebooks/tests)

Tests use only numpy (+ the repo's existing keypoint files when present),
so they also run on Colab before any training.
"""
import os
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import signlens_core as sc  # noqa: E402
from signlens_core import inference, sequences  # noqa: E402

RNG = np.random.default_rng(0)
REPO_MODEL_DIR = os.environ.get("SIGNLENS_MODEL_DIR",
                                os.path.join(os.path.dirname(os.path.dirname(HERE)), "model"))


def fake_hand(seed=0):
    r = np.random.default_rng(seed)
    base = np.zeros((21, 3), np.float32)
    base[1:] = r.uniform(-0.1, 0.1, (20, 3))
    base[9] = [r.uniform(-0.03, 0.03), -0.12, r.uniform(-0.02, 0.02)]  # middle knuckle above the wrist
    return base + [0.5, 0.6, 0.0]


def test_translation_and_scale_invariance():
    h = fake_hand()
    f1 = sc.hand_features(h, "Right", 1.0)
    moved = (h - h[0]) * 1.7 + h[0] + [0.2, -0.1, 0.0]  # hand closer to camera + elsewhere in frame
    f2 = sc.hand_features(moved, "Right", 1.0)
    assert np.allclose(f1, f2, atol=1e-4), "features must not depend on hand position/size"


def test_left_hand_is_mirrored_to_right():
    h = fake_hand()
    mirrored = h.copy(); mirrored[:, 0] = 1.0 - mirrored[:, 0]
    assert np.allclose(sc.hand_features(h, "Right"), sc.hand_features(mirrored, "Left"), atol=1e-4)


def test_aspect_correction():
    h = fake_hand()
    # MediaPipe divides x by image width; its z uses the same scale as x
    squashed = h.copy(); squashed[:, 0] = 0.5 + (squashed[:, 0] - 0.5) * (9 / 16)
    squashed[:, 2] *= 9 / 16
    assert np.allclose(sc.normalize_hand(h, "Right", 1.0),
                       sc.normalize_hand(squashed, "Right", 16 / 9), atol=1e-4)


def test_feature_shape_and_palm_unit():
    p = sc.normalize_hand(fake_hand())
    assert p.shape == (21, 3) and np.allclose(p[0], 0)
    assert abs(np.linalg.norm(p[9, :2]) - 1) < 1e-4
    assert sc.hand_features(fake_hand()).shape == (sc.FEATURE_DIM,)


def test_no_constant_features_and_scaled_inputs_stay_bounded():
    """Regression test for the v2.0.0 bug: the wrist coordinates were always 0,
    Keras divided augmented values by ~0, and the network saw inputs of ~770,000."""
    P = np.stack([sc.normalize_hand(fake_hand(i)) for i in range(60)])
    clean = sc.batches.static_batch(P, range(len(P)))
    aug = sc.batches.static_batch(P, range(len(P)), RNG, augment=True)
    assert clean.shape[1] == sc.FEATURE_DIM
    assert clean.var(0).min() > 1e-8, "a feature never changes"
    keras_like = (aug - clean.mean(0)) / np.maximum(clean.std(0), 1e-7)
    assert np.abs(keras_like).max() < 1000, np.abs(keras_like).max()
    mean, var = sc.scaling_stats(clean)
    assert np.all(var >= 1e-4)
    assert np.allclose(sc.augment_hand(P[0], RNG)[0], 0), "wrist must stay at the origin"


def test_augmentation_changes_but_stays_valid():
    p = sc.normalize_hand(fake_hand())
    outs = [sc.augment_hand(p, RNG) for _ in range(50)]
    assert all(np.isfinite(o).all() and o.shape == (21, 3) for o in outs)
    spread = np.std([o[8] for o in outs], axis=0).mean()
    assert 0.01 < spread < 0.5, f"augmentation strength looks wrong: {spread}"


def test_sequence_build_resample_and_gaps():
    frames = [fake_hand(i % 3) + [0.01 * i, 0, 0] for i in range(40)]
    frames[5] = frames[6] = None             # tracker lost the hand briefly
    s = sc.build_sequence(frames, "Right", 1.0)
    assert s["hands"].shape == (sc.SEQ_LEN, 21, 3) and s["traj"].shape == (sc.SEQ_LEN, 3)
    assert s["traj"][-1, 0] > s["traj"][0, 0], "rightward motion must be kept in the trajectory"
    assert sc.sequence_features(s).shape == (sc.SEQ_LEN, sc.SEQ_FEATURE_DIM)
    assert sc.build_sequence([None] * 30, "Right") is None
    a = sc.augment_seq_sample(s, RNG)
    assert sc.sequence_features(a).shape == (sc.SEQ_LEN, sc.SEQ_FEATURE_DIM)


def test_legacy_vector_decoding():
    v = np.zeros(258, np.float32); v[195:258] = fake_hand().ravel()
    lm, h = sc.hand_from_legacy_vector(v); assert h == "Right" and np.allclose(lm, fake_hand())
    v = np.zeros(1662, np.float32); v[1536:1599] = fake_hand().ravel()
    lm, h = sc.hand_from_legacy_vector(v); assert h == "Left"
    assert sc.hand_from_legacy_vector(np.zeros(258))[0] is None


def test_grouped_split_has_no_leak_even_with_shared_clips():
    labels = np.array(["A"] * 40 + ["B"] * 40)
    groups = np.array([f"g{i // 4}" for i in range(80)])
    groups[38:42] = "transition"             # one clip holds frames of A and B
    tr, va, te = sc.grouped_split(labels, groups, 0.2, 0.1)
    assert sc.check_no_leak(groups, tr, va, te)
    assert len(tr) + len(va) + len(te) == 80


def test_label_order_is_stable():
    assert sc.sort_labels(["10", "B", "2", "A", "Z", "1"]) == ["A", "B", "Z", "1", "2", "10"]


def test_extract_parsing_with_mock_mediapipe():
    from signlens_core.extract import HandExtractor
    P = types.SimpleNamespace
    res = P(hand_landmarks=[[P(x=0.1 * i % 1, y=0.2, z=0.0) for i in range(21)]],
            handedness=[[P(category_name="Left", score=0.97)]])
    lm, handed, score = HandExtractor._parse(res)
    assert lm.shape == (21, 3) and handed == "Left" and score > 0.9
    assert HandExtractor._parse(P(hand_landmarks=[], handedness=[]))[0] is None


class _FakeExtractor:
    """Stands in for MediaPipe: 'detects' a hand in every image whose top-left pixel isn't black."""
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *e): pass
    def close(self): pass
    def from_image(self, bgr):
        return (fake_hand(), "Right", 0.9) if bgr[0, 0].any() else (None, None, 0.0)
    def from_frame(self, bgr, ts):
        return self.from_image(bgr)


def test_crop_hand_inside_and_at_edges():
    import cv2  # noqa: F401
    from signlens_core.extract import crop_hand
    img = np.zeros((480, 640, 3), np.uint8)
    lm = fake_hand(); lm[:, :2] = np.clip(lm[:, :2], 0, 1)
    assert crop_hand(img, lm, 96).shape == (96, 96, 3)
    edge = lm.copy(); edge[:, 0] = edge[:, 0] - 0.45   # hand partly outside the frame
    assert crop_hand(img, edge, 96).shape == (96, 96, 3)


def test_image_dataset_folder_walk_with_fake_mediapipe():
    import cv2
    from signlens_core import extract
    orig = extract.HandExtractor
    extract.HandExtractor = _FakeExtractor
    try:
        with tempfile.TemporaryDirectory() as d:
            for lab, n in (("A", 12), ("b", 7), ("del", 3), ("nothing", 2)):   # Kaggle-style extras
                os.makedirs(os.path.join(d, "asl_train", lab))
                for i in range(n):
                    img = np.full((60, 80, 3), 200, np.uint8)
                    if i == 0:
                        img[:] = 0               # "no hand" image
                    cv2.imwrite(os.path.join(d, "asl_train", lab, f"{lab}{i}.jpg"), img)
            s = extract.extract_image_dataset(d, "unused.task", "kaggle_x", max_per_class=10,
                                              bucket=4, progress=False)
            assert set(s.label) == {"A", "B"}, set(s.label)          # 'del'/'nothing' ignored, 'b' -> 'B'
            assert np.sum(s.label == "A") == 9 and np.sum(s.label == "B") == 6
            assert abs(s.aspect[0] - 80 / 60) < 1e-6
            assert len(set(s.group[s.label == "A"])) == 3            # 10 sampled files / bucket 4
    finally:
        extract.HandExtractor = orig


def test_video_dataset_with_fake_mediapipe():
    import cv2
    from signlens_core import extract
    orig = extract.HandExtractor
    extract.HandExtractor = _FakeExtractor
    try:
        with tempfile.TemporaryDirectory() as d:
            for lab in ("A", "J"):
                os.makedirs(os.path.join(d, lab))
                path = os.path.join(d, lab, f"ama_01.mp4")
                vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
                for _ in range(60):
                    vw.write(np.full((48, 64, 3), 180, np.uint8))
                vw.release()
            stat, seqs, labs, grps = extract.extract_video_dataset(d, "unused.task", ("J",), progress=False)
            assert set(stat.label) == {"A"} and len(stat) > 10
            assert labs == ["J"] and seqs[0]["hands"].shape == (sc.SEQ_LEN, 21, 3)
            assert all(g.startswith("my_videos:A/") for g in stat.group)
    finally:
        extract.HandExtractor = orig


def test_synthetic_other_motion():
    from signlens_core.sequences import synthetic_other_motion
    P = np.stack([sc.normalize_hand(fake_hand(i)) for i in range(5)])
    clips = synthetic_other_motion(P, RNG, 10)
    assert len(clips) == 10
    assert all(np.isfinite(sc.sequence_features(c)).all() for c in clips)


def test_sequence_save_load_roundtrip():
    s = sc.build_sequence([fake_hand(i % 2) for i in range(20)], "Right")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "seq.npz")
        sc.save_sequences(p, [s, s], ["J", "Z"], ["g1", "g2"], ["src", "src"])
        samples, labels, groups, sources = sc.load_sequences(p)
        assert list(labels) == ["J", "Z"] and np.allclose(samples[1]["traj"], s["traj"])


def test_static_set_save_load_roundtrip():
    S = sc.StaticSet(np.stack([fake_hand(), fake_hand(1)]), ["Right", "Left"], [1.0, 1.5],
                     ["A", "B"], ["g1", "g2"], ["src", "src"])
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.npz"); S.save(p); T = sc.StaticSet.load(p)
        assert list(T.label) == ["A", "B"] and np.allclose(T.normalized(), S.normalized())


def test_predictor_with_dummy_model():
    with tempfile.TemporaryDirectory() as d:
        lab = os.path.join(d, "labels.json")
        sc.save_labels(lab, ["A", "B"])
        inference.load_predict_fn = lambda path: (lambda x: np.tile([0.9, 0.1], (len(x), 1)))
        pred = inference.SignPredictor("x.keras", lab, min_conf=0.6, smooth=3)
        assert pred.predict_landmarks(None)["prediction"] is None
        out = [pred.update(fake_hand(), "Right", t / 30) for t in range(5)]
        assert out[-1]["prediction"] == "A" and out[-1]["stable"]


def test_repo_data_if_available():
    if not os.path.isdir(os.path.join(REPO_MODEL_DIR, "dataset")):
        print("   (repo dataset not found, skipped)")
        return
    yt = sc.load_repo_youtube_frames(REPO_MODEL_DIR)
    assert len(yt) > 500 and "Neutral" not in set(yt.label)
    feats = np.stack([sc.hand_features(l, h, a) for l, h, a in zip(yt.lm[:50], yt.handed[:50], yt.aspect[:50])])
    assert np.isfinite(feats).all()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
