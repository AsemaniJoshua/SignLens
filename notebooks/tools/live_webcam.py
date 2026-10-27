"""Live test of the trained models on your laptop webcam.

    pip install mediapipe opencv-python ai-edge-litert
    python notebooks/tools/live_webcam.py --models path/to/SignLens/models

Download the `models/` folder from Google Drive (MyDrive/SignLens/models)
first. Press q to quit, m to switch letters/numbers mode.
"""
import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from signlens_core import extract  # noqa: E402
from signlens_core.inference import SignPredictor  # noqa: E402


def make_predictor(models, mode):
    def f(name):
        for ext in (".tflite", ".keras"):
            p = os.path.join(models, f"{name}_model{ext}")
            if os.path.exists(p):
                return p
        return None
    static = f(f"static_{mode}")
    if static is None:
        sys.exit(f"No static_{mode}_model.tflite/.keras in {models}")
    motion = f("motion") if mode == "letters" else None
    return SignPredictor(static, os.path.join(models, f"static_{mode}_labels.json"),
                         motion_model=motion,
                         motion_labels=os.path.join(models, "motion_labels.json") if motion else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, help="folder with *_model.tflite and *_labels.json")
    ap.add_argument("--hand-model", default="hand_landmarker.task", help="MediaPipe model (downloaded if missing)")
    ap.add_argument("--camera", default="0", help="camera index or a video file path")
    args = ap.parse_args()

    hand_model = extract.download_model(args.hand_model)
    modes = ["letters"] + (["digits"] if os.path.exists(os.path.join(args.models, "static_digits_labels.json")) else [])
    mode = 0
    predictor = make_predictor(args.models, modes[mode])
    cap = cv2.VideoCapture(int(args.camera) if args.camera.isdigit() else args.camera)
    t0 = time.time()
    with extract.HandExtractor(hand_model, video=True) as ex:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = time.time() - t0
            h, w = frame.shape[:2]
            lm, handed, _ = ex.from_frame(frame, int(t * 1000))   # raw frame, never flipped (same as training)
            r = predictor.update(lm, handed, t, aspect=w / h)
            shown = cv2.flip(frame, 1)                            # mirror only for display
            text = (f"{r['prediction']}  {r['confidence']:.0%}  [{r.get('model', '')}]"
                    if r["prediction"] else "show your hand")
            color = (0, 200, 0) if r.get("stable") else (0, 165, 255)
            cv2.putText(shown, text, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
            cv2.putText(shown, f"mode: {modes[mode]} (m to switch, q to quit)", (16, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("SignLens", shown)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m") and len(modes) > 1:
                mode = (mode + 1) % len(modes)
                predictor = make_predictor(args.models, modes[mode])
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
