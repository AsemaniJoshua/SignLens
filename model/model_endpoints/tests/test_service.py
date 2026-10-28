"""Tests for the v2 model server.   Run from model/model_endpoints:

    python tests/test_service.py

MediaPipe's hand model needs a download, so the tests replace only the
"find the hand" step: images/frames are mapped to REAL hand landmarks from the
repo dataset. A small classifier trained on those landmarks stands in for the
.tflite file. Everything after that (features, voting, motion switch, stream
smoothing, API responses) runs for real. FastAPI is replaced by a tiny stub
if it isn't installed, so the route functions themselves are exercised too.
"""
import asyncio
import filecmp
import json
import os
import sys
import tempfile
import types

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(SERVER, "..", ".."))
sys.path.insert(0, SERVER)
MODEL_DIR = os.environ.get("SIGNLENS_MODEL_DIR", os.path.join(REPO, "model"))

import signlens_core as sc  # noqa: E402
from signlens_core import extract  # noqa: E402


# --------------------------------------------------------------------- fakes
def _install_fastapi_stub():
    try:
        import fastapi  # noqa: F401
        return False
    except ImportError:
        pass
    m = types.ModuleType("fastapi")

    class APIRouter:
        def post(self, *a, **k): return lambda f: f
        def websocket(self, *a, **k): return lambda f: f

    class HTTPException(Exception):
        def __init__(self, status_code, detail=None):
            super().__init__(detail); self.status_code, self.detail = status_code, detail

    class WebSocketDisconnect(Exception):
        pass

    m.APIRouter, m.HTTPException, m.WebSocketDisconnect = APIRouter, HTTPException, WebSocketDisconnect
    m.File = m.Query = lambda default=None, **k: default
    m.UploadFile = m.WebSocket = object
    m.status = types.SimpleNamespace(HTTP_503_SERVICE_UNAVAILABLE=503, WS_1008_POLICY_VIOLATION=1008,
                                     WS_1011_INTERNAL_ERROR=1011)
    sys.modules["fastapi"] = m
    return True


def load_real_landmarks():
    yt = sc.load_repo_youtube_frames(MODEL_DIR)
    yt = yt.subset(yt.source == "youtube")
    cam = sc.load_repo_webcam_sequences(MODEL_DIR, frame_step=3)
    return sc.StaticSet.concat([yt, cam])


def train_fake_tflite(S):
    """A real classifier on real landmarks, wrapped like load_predict_fn returns."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    labels = sc.sort_labels(S.label)
    y = np.array([labels.index(l) for l in S.label])
    X = sc.batches.static_batch(S.normalized(), range(len(S)))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=400)).fit(X, y)

    def fn(x):
        p = np.zeros((len(x), len(labels)), np.float32)
        p[:, clf.classes_] = clf.predict_proba(np.asarray(x, np.float32))
        return p
    return fn, labels


class FakeExtractor:
    """Image pixel (0,0) encodes which real hand to return; black = no hand."""
    hands = None   # set by the tests: list of (lm, handed)

    def __init__(self, *a, **k): pass
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *e): pass

    def from_image(self, bgr):
        code = int(bgr[0, 0, 0]) + 256 * int(bgr[0, 0, 1])
        if code == 0:
            return None, None, 0.0
        lm, handed = FakeExtractor.hands[code - 1]
        return lm, handed, 0.95

    def from_frame(self, bgr, ts):
        return self.from_image(bgr)


def image_bytes(code, size=(48, 64)):
    img = np.full(size + (3,), 127, np.uint8)
    img[0, 0] = [code % 256, code // 256, 0]
    ok, buf = cv2.imencode(".png", img)    # PNG: lossless, so the code survives
    return buf.tobytes()


# --------------------------------------------------------------------- setup
def build_service(tmp):
    from app.models import signlens_service as svc
    S = load_real_landmarks()
    static_fn, labels = train_fake_tflite(S)
    # motion stand-in: says "Z" for clips whose wrist really travels, else "other"
    motion_labels = ["J", "Z", "other"]

    def motion_fn(x):
        path = np.abs(x[0, :, -2:]).sum()          # summed wrist velocity features
        return np.array([[0.02, 0.95, 0.03]] if path > 1.0 else [[0.05, 0.05, 0.9]], np.float32)

    for name, labs in (("static_letters", labels), ("motion", motion_labels)):
        open(os.path.join(tmp, f"{name}_model.tflite"), "wb").write(b"x")
        sc.save_labels(os.path.join(tmp, f"{name}_labels.json"), labs)
    fns = {"static_letters": static_fn, "motion": motion_fn}
    svc.load_predict_fn = lambda path: fns[os.path.basename(path).replace("_model.tflite", "")]
    svc.extract.download_model = lambda path: path
    svc.extract.HandExtractor = FakeExtractor
    hands = list(zip(S.lm, S.handed))
    FakeExtractor.hands = hands
    service = svc.SignLensService(models_dir=tmp, hand_model_path=os.path.join(tmp, "hand.task"))
    return svc, service, S


# --------------------------------------------------------------------- tests
def test_vendored_core_matches_notebooks():
    a, b = os.path.join(REPO, "notebooks", "signlens_core"), os.path.join(SERVER, "signlens_core")
    if not os.path.isdir(a):
        print("   (notebooks/ not present, skipped)")
        return
    files = sorted(f for f in os.listdir(a) if f.endswith(".py"))
    match, mismatch, errors = filecmp.cmpfiles(a, b, files, shallow=False)
    assert not mismatch and not errors, f"run sync_signlens_core.py: {mismatch + errors}"


def test_missing_models_give_clear_error():
    from app.models import signlens_service as svc
    with tempfile.TemporaryDirectory() as d:
        try:
            svc.SignLensService(models_dir=d, hand_model_path="x")
            raise AssertionError("should fail")
        except svc.ModelsNotReady as e:
            assert "static_letters" in str(e)


def test_image_prediction_matches_true_letter():
    with tempfile.TemporaryDirectory() as d:
        svc, service, S = build_service(d)
        rng = np.random.default_rng(0)
        picks = rng.choice(len(S), 60, replace=False)
        right = 0
        for i in picks:
            r = service.predict_image(image_bytes(int(i) + 1))
            assert set(r) >= {"prediction", "confidence", "fallback", "model_used"}
            right += r["prediction"] == S.label[i]
        assert right / len(picks) > 0.85, right / len(picks)
        r = service.predict_image(image_bytes(0))
        assert r["prediction"] == svc.NO_HAND_MSG and r["confidence"] == 0.0
        b64 = ("data:image/png;base64," + __import__("base64").b64encode(image_bytes(int(picks[0]) + 1)).decode()).encode()
        assert service.predict_image(b64)["prediction"] == S.label[picks[0]]
        try:
            service.predict_image(b"not an image")
            raise AssertionError("should fail")
        except ValueError:
            pass
        try:
            service.predict_image(image_bytes(1), mode="digits")
            raise AssertionError("digits model is not loaded in this test")
        except ValueError as e:
            assert "digits" in str(e)


def _fake_video(frames, handed="Right", aspect=4 / 3):
    return lambda path, model, max_s=10.0: {"frames": frames, "times": [], "handed": handed,
                                             "aspect": aspect, "fps": 30.0}


def test_video_held_sign_uses_static_vote():
    with tempfile.TemporaryDirectory() as d:
        svc, service, S = build_service(d)
        right = total = 0
        for lab in ("A", "B", "L", "O", "R"):
            rec = [g for g in sorted(set(S.group)) if g.startswith(f"cam:{lab}/")][0]
            frames = [lm for lm, g in zip(S.lm, S.group) if g == rec]
            svc.extract.read_video_landmarks = _fake_video(frames)
            r = service.predict_video(b"fake mp4 bytes")
            total += 1
            right += r["prediction"] == lab
            assert r["model_used"] == "SignLens-v2", "a held sign must not go to the motion model"
        assert right >= 4, f"{right}/{total}"
        svc.extract.read_video_landmarks = _fake_video([None] * 30)
        assert service.predict_video(b"x")["prediction"] == svc.NO_HAND_MSG


def test_video_moving_sign_uses_motion_model():
    with tempfile.TemporaryDirectory() as d:
        svc, service, S = build_service(d)
        base = S.lm[0]
        frames = [base + [0.012 * t, 0.006 * np.sin(t / 3), 0] for t in range(40)]  # hand travels
        svc.extract.read_video_landmarks = _fake_video(frames)
        r = service.predict_video(b"x")
        assert r["prediction"] == "Z" and r["model_used"] == "SignLens-v2-motion", r
        # digits mode never uses the motion model
        service.static["digits"] = service.static["letters"]
        assert service.predict_video(b"x", mode="digits")["model_used"] == "SignLens-v2"


def test_stream_becomes_stable_on_held_sign():
    with tempfile.TemporaryDirectory() as d:
        svc, service, S = build_service(d)
        rec = [g for g in sorted(set(S.group)) if g.startswith("cam:B/")][0]
        idx = [i for i, g in enumerate(S.group) if g == rec]
        session = service.stream("letters")
        out = [session.push(image_bytes(i + 1), t_seconds=k / 15) for k, i in enumerate(idx * 2)]
        assert out[-1]["prediction"] == "B" and out[-1]["stable"], out[-1]
        r = session.push(image_bytes(0), t_seconds=99)
        assert r["prediction"] is None and "No hand" in r["status"]
        session.close()


def test_api_routes():
    stubbed = _install_fastapi_stub()
    with tempfile.TemporaryDirectory() as d:
        svc, service, S = build_service(d)
        svc._service = service
        from app.api_v1 import endpoints
        from app.api_v1.schemas import InputType, Mode

        upload = types.SimpleNamespace(file=types.SimpleNamespace(read=lambda: image_bytes(5)))
        r = endpoints.predict_media(InputType.image, upload, Mode.letters)
        assert r["prediction"] == S.label[4] and r["fallback"] is False
        if stubbed:
            try:
                endpoints.predict_media(InputType.stream, upload, Mode.letters)
                raise AssertionError("stream over POST must be rejected")
            except sys.modules["fastapi"].HTTPException as e:
                assert e.status_code == 400

        # Gemini fallback stays off unless configured
        os.environ.pop("GEMINI_FALLBACK_BELOW", None)
        none = types.SimpleNamespace(file=types.SimpleNamespace(read=lambda: image_bytes(0)))
        assert endpoints.predict_media(InputType.image, none, Mode.letters)["fallback"] is False
        os.environ["GEMINI_FALLBACK_BELOW"] = "0.5"
        endpoints.predict_with_gemini = lambda b, m: "A"
        r = endpoints.predict_media(InputType.image, none, Mode.letters)
        assert r["fallback"] is True and r["prediction"] == "A"
        os.environ.pop("GEMINI_FALLBACK_BELOW")

        # WebSocket: 3 frames then disconnect
        WSD = sys.modules["fastapi"].WebSocketDisconnect
        frames = [image_bytes(5), image_bytes(5), image_bytes(0)]

        class WS:
            query_params = {"api_key": "k", "type": "stream"}
            sent, closed = [], None
            async def accept(self): pass
            async def receive_bytes(self):
                if frames:
                    return frames.pop(0)
                raise WSD()
            async def send_json(self, x): WS.sent.append(x)
            async def close(self, code=None): WS.closed = code
        asyncio.run(endpoints.predict_stream(WS()))
        assert len(WS.sent) == 3 and WS.sent[0]["prediction"] == S.label[4]
        assert WS.sent[2]["prediction"] is None

        WS.query_params, WS.sent = {"type": "stream"}, []
        asyncio.run(endpoints.predict_stream(WS()))
        assert WS.closed == 1008, "missing api_key must be refused"
        svc._service = None


if __name__ == "__main__":
    if not os.path.isdir(os.path.join(MODEL_DIR, "initial build", "dataset")):
        sys.exit(f"Repo dataset not found at {MODEL_DIR} (set SIGNLENS_MODEL_DIR)")
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
