# SignLens v2: training notebooks

A rebuilt training pipeline for SignLens fingerspelling recognition, designed to run on **Google Colab**. Read `../AUDIT.md` first: it explains why the old models failed, with numbers.

## What's here

```
notebooks/
├── 00_diagnose_old_models.ipynb     reproduces why the old models failed (CPU, ~1 min)
├── 01_build_landmark_dataset.ipynb  repo data + Kaggle + your photos/videos -> landmark dataset
├── 02_train_static_model.ipynb      letters model (and digits model): main model
├── 03_train_motion_model.ipynb      J and Z (movement signs)
├── 04_image_cnn_baseline.ipynb      optional pixel-based CNN, for comparison in the paper
├── signlens_core/                   shared code used by notebooks AND the API
│   ├── landmarks.py   hand normalization + 108 features  <- the single source of truth
│   ├── augment.py     geometric augmentation (training only)
│   ├── sequences.py   motion clips: resampling, wrist path, generated "other" class
│   ├── data.py        loaders, dataset format, leak-free grouped split
│   ├── extract.py     MediaPipe Tasks HandLandmarker, hand crops, folder datasets
│   ├── batches.py     training batches with on-the-fly augmentation
│   ├── evaluate.py    reports, confusion matrix, leave-one-source-out test
│   └── inference.py   SignPredictor: one predictor for API, webcam and notebooks
├── tests/test_core.py 18 tests (python notebooks/tests/test_core.py)
└── tools/live_webcam.py  live test on your laptop camera
```

## 1. Put it in the repo (once)

1. Copy the `notebooks/` folder and `AUDIT.md` into the root of the SignLens repo.
2. Add a root `.gitignore` so the repo stops growing:
   ```
   node_modules/
   **/node_modules/
   __pycache__/
   *.pyc
   .venv*/
   work/
   ```
   Then remove what's already committed (files stay on your disk): `git rm -r --cached admin/node_modules`
3. Commit and push. The notebooks clone the repo from GitHub, so they must be on `main`.

## 2. Run order on Colab

Open each notebook with its *Open in Colab* badge, or go to colab.research.google.com → GitHub → `AsemaniJoshua/SignLens`.

| step | notebook | runtime | time |
|---|---|---|---|
| 1 | `00_diagnose_old_models` | CPU | ~1 min |
| 2 | `01_build_landmark_dataset` | CPU | 10–20 min the first time (Kaggle download + extraction, cached in Drive afterwards) |
| 3 | `02_train_static_model` | CPU or T4 GPU | 5–15 min |
| 4 | `02_train_static_model` again with `CLASS_SET = "digits"` | | optional |
| 5 | `03_train_motion_model` | CPU | ~5 min |
| 6 | `04_image_cnn_baseline` | **T4 GPU** | ~30 min, optional |

Everything is saved to **`MyDrive/SignLens/`** (`data/`, `models/`), so nothing is lost when Colab disconnects.

**Kaggle (for notebook 01 and 04):** kaggle.com → Settings → *Create New Token*. In Colab: 🔑 *Secrets* → add `KAGGLE_USERNAME` and `KAGGLE_KEY` → enable notebook access.

## 3. The two numbers to report

* **Held-out recordings** (notebook 02, section 3): no frame of a test recording was seen in training.
* **Leave-one-source-out** (section 4): the model never saw that person or dataset at all. This is the honest "new user" number; report this one.

Many published fingerspelling results (99%+) are random splits of frames from one dataset, which is the same trap as the old SignLens. A properly signer-independent evaluation is itself a contribution worth stating in your paper.

## 4. Collecting more data (this is the biggest lever now)

The pipeline is fixed; accuracy for new users now depends mainly on **how many different people** are in the data.

* **Who:** any 5–10 people (friends, classmates, family). You don't need to be a fluent signer for fingerspelling. Learn each handshape from a GSL/ASL reference and check with a deaf signer when possible.
* **How:** phone video, 3–5 s per clip, hold the letter while moving the hand slightly (closer, further, small tilt). One sign per clip. Different rooms and lighting.
* **Where:** `MyDrive/SignLens/my_videos/<SIGN>/<name>_<nn>.mp4`, e.g. `my_videos/A/ama_01.mp4`, `my_videos/J/kofi_03.mp4`. The name before `_` becomes the "person", so you can test on someone the model never saw.
* **How much:** 2 clips per letter per person is enough to start. 10 people × 2 clips × 26 letters ≈ 520 short clips, about 1 hour of recording in total.
* **Privacy:** only landmarks (21 hand points) are stored for training, not faces or video. Tell participants that, and keep the raw videos private in Drive (not in the public repo).

Notebook 01's last cell prints exactly which letters still need data.

## 5. Use the new models in the API

The exported files (from `MyDrive/SignLens/models/`):

```
static_letters_model.tflite   static_letters_labels.json
static_digits_model.tflite    static_digits_labels.json     (if trained)
motion_model.tflite           motion_labels.json
```

Replace the prediction code in `model/model_endpoints` with the shared predictor, so the API processes landmarks **exactly** like training did and reads labels from the JSON files:

```python
# model/model_endpoints/app/models/signlens.py
import os, sys
import cv2, numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "notebooks"))
from signlens_core import extract
from signlens_core.inference import SignPredictor

MODELS = os.path.join(os.path.dirname(__file__), "..", "..", "saved_models_v2")
HAND_MODEL = extract.download_model(os.path.join(MODELS, "hand_landmarker.task"))
LETTERS = SignPredictor(os.path.join(MODELS, "static_letters_model.tflite"),
                        os.path.join(MODELS, "static_letters_labels.json"))

def image_predict(file_bytes: bytes) -> dict:
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    with extract.HandExtractor(HAND_MODEL) as ex:          # do NOT flip the image
        lm, handed, _ = ex.from_image(img)
    h, w = img.shape[:2]
    return LETTERS.predict_landmarks(lm, handed, aspect=w / h)
    # -> {"prediction": "A", "confidence": 0.97, "top3": [...]}  or prediction None if no hand
```

For the WebSocket stream, create **one `SignPredictor` (with the motion model) and one `HandExtractor(video=True)` per connection**, and for every frame call `ex.from_frame(img, timestamp_ms)` then `predictor.update(lm, handed, t_seconds, aspect=w / h)`. It smooths over a few frames and switches to the motion model only when the hand is actually moving.

Update `model_endpoints/requirements.txt` (TensorFlow is no longer needed to serve `.tflite` models):

```
fastapi
uvicorn
python-multipart
pydantic
numpy
opencv-python-headless
mediapipe>=0.10.20
ai-edge-litert
```

Consider keeping the Gemini fallback only as an explicit, logged option. With the old 0.9 threshold, most answers came from Gemini rather than SignLens.

## 6. Test locally

```bash
pip install mediapipe opencv-python ai-edge-litert numpy
python notebooks/tests/test_core.py
python notebooks/tools/live_webcam.py --models path/to/downloaded/models
```

## What was verified before handing this over

* `signlens_core`: 18 tests pass (normalization invariances, augmentation, grouped split without leaks, sequence building, dataset loaders on the real repo data, extraction logic with a mocked MediaPipe, predictor smoothing).
* Notebook 00 ran end to end on the real repo data. Notebook 01 ran end to end with the repo data; its Kaggle and own-photo/video paths were tested through the core tests with a mocked MediaPipe.
* Notebooks 02 and 03 ran end to end with a scikit-learn stand-in in place of the Keras training cells (the test environment had no TensorFlow). Data loading, splitting, augmentation, evaluation and reporting were executed for real.
* **Not executed:** the Keras/TFLite cells and notebook 04, which were only checked for syntax and undefined names, and the Kaggle downloads and MediaPipe model download (network-restricted). Check the first Colab run of notebook 02 carefully, especially the "TFLite vs Keras" check at the end of the export cell.
