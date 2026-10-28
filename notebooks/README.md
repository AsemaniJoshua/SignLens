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
│   ├── landmarks.py   hand normalization + 105 features  <- the single source of truth
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

The model server (`model/model_endpoints`) already uses the v2 models. Same URLs and response fields as before, so the mobile app keeps working.

**Set up (once):**

1. Copy these from Google Drive (`MyDrive/SignLens/models/`) into `model/model_endpoints/saved_models_v2/`:
   `static_letters_model.tflite`, `static_letters_labels.json`, and optionally the `static_digits_*` and `motion_*` files.
2. Install and start, from `model/model_endpoints`:
   ```bash
   pip install -r requirements.txt
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
3. Open `http://localhost:8000/`. It should say `"status": "ok"` and list the loaded modes.

**What changed for the app:**

| | before | now |
|---|---|---|
| `POST /api/v1/predict?type=image\|video` | old `.h5` models, wrong-letter bug | v2 models; optional `&mode=digits` |
| no hand in the picture | a guessed letter | `"No hand detected. ..."` with confidence 0 |
| unsure | sent to Gemini below 90% | `"Not sure ..."`; Gemini only if you turn it on |
| live stream (WebSocket) | flickering raw guesses | smoothed; show a letter when `"stable": true` |
| extra fields | | `mode`, `top3`, `reason` (the app can ignore them) |

**Gemini fallback** is off by default. To turn it on, set `GEMINI_API_KEY` and `GEMINI_FALLBACK_BELOW=0.5`; answers from Gemini are marked `"fallback": true`.

**Shared code:** the server has its own copy of `signlens_core` (Docker only sees `model/model_endpoints`). After changing `notebooks/signlens_core`, run `python model/model_endpoints/sync_signlens_core.py`. The server tests check that the copies match:

```bash
cd model/model_endpoints && python tests/test_service.py
```

## Re-export a model without retraining

`.tflite` files made before the export fix (commit `50cb148`) are about **10 KB** and output NaN; the predictor refuses them. The `.keras` files from those runs are fine. To rebuild the `.tflite` files from them, run this in any Colab notebook after its Setup cell (no retraining, about 1 minute):

```python
import tempfile, numpy as np, tensorflow as tf, keras
from signlens_core import inference
for name in ("static_letters", "static_digits", "motion"):
    k_path, t_path = f"{OUT_DIR}/{name}_model.keras", f"{OUT_DIR}/{name}_model.tflite"
    if not os.path.exists(k_path):
        continue
    model = keras.models.load_model(k_path)
    d = tempfile.mkdtemp() + "/saved_model"
    model.export(d, format="tf_saved_model")
    open(t_path, "wb").write(tf.lite.TFLiteConverter.from_saved_model(d).convert())
    X = np.random.default_rng(0).normal(size=(50,) + tuple(model.input_shape[1:])).astype("float32")
    k, l = model.predict(X, verbose=0), inference.load_predict_fn(t_path)(X)
    print(f"{name}: {os.path.getsize(t_path)/1024:.0f} KB | max difference {np.abs(k - l).max():.1e} | "
          f"same answer {np.mean(k.argmax(1) == l.argmax(1)):.0%}")
```

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
