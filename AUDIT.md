# SignLens model audit

Audit of `AsemaniJoshua/SignLens` at commit `33d86e4` (main), October 2026.
Scope: everything under `model/` (data, training, inference, API). Every number below was measured on the data in the repo and can be reproduced with `notebooks/00_diagnose_old_models.ipynb`.

## Summary

The models did not fail because the network architecture was wrong. They failed because of the **data**, the **features**, and **how accuracy was measured**. On top of that, a **label bug in the API** reported the wrong sign even when the video model was right.

| measurement (same data, same network size) | accuracy |
|---|---|
| Old static pipeline, measured the old way (augment, then random split) | **89.3%** |
| Old features, tested on a **different signer** (YouTube → webcam) | **12.0%** (chance is 5.6%) |
| Old features, tested on a different signer (webcam → YouTube) | **5.8%** |
| New normalized hand features + geometric augmentation, YouTube → webcam | **85.5%** |
| New pipeline, letters only, train without the webcam signer, test on them | **87.6%** |

The 89% the old code printed was memorization of near-duplicate frames. Facing a new person, the old model was guessing.

## Root causes, most important first

### 1. Accuracy was measured on copies of the training data
- `model/static_keypoints_optimized/scripts/train_model_static.py:45-62`: noise copies are added to every sample **before** shuffling and splitting, so each test sample has a near-twin in training.
- `model/video_lstm_optimized/scripts/augment_keypoints.py:29-43` writes 30 augmented copies of each clip to disk. `train_model_video.py:12-25` then loads them together with the originals, and `:49-58` splits randomly. The test set is mostly augmented copies of training clips.
- `model/image_cnn/scripts/train_model_image.py:15-41`: `validation_split` on frames cut from the same 1–2 second clips, from the same augmenting generator (including `horizontal_flip`) used for training.
- On top of that, all static images are consecutive video frames (`FRAME_STEP = 3` in `automation_scripts/generate_dataset.py:24`), so even without augmentation, random splits put near-identical frames on both sides.

### 2. The features described where the person was, not the handshape
- Every extractor (`extract_keypoints.py:14-24`, `extract_keypoints_from_videos.py:15-25`, `endpoints.py:119-136`) feeds raw MediaPipe coordinates: 132 body-pose values plus hand points in **screen position**, unnormalized for hand size, distance, left/right hand or image aspect ratio.
- The model can separate training letters by where the hand sits in the frame and how the body is posed. That works on the one video it saw and collapses for anyone else (cause of the 12% / 5.8% above).

### 3. The API mapped video predictions to the wrong letters
- `model/model_endpoints/app/api_v1/endpoints.py:88` hard-codes `VIDEO_CLASSES` with **38** names.
- The video model was trained on `dataset/video_keypoints_no_face`, which has **33** classes, because H, K, L, W and X were never extracted (see cause 5).
- Every model output after `G` was reported as the wrong sign: **15 of 33 classes**, e.g. model "I" → API "H", model "M" → API "J".
- `video_lstm_optimized/scripts/run_model_video.py:35-36` has the same problem: it reads labels from `video_raw`, not from the training folder.

### 4. Very little real variation in the data
- **YouTube source:** all A–Z images and videos come from **one YouTube video** ("ASL Manual Alphabet | Start ASL"): one signer, one front and one side clip per letter, 1–2 s each. Each class folder also holds transition frames of the neighbouring letter (e.g. `image/J/I_Front_frame_015.jpg`).
- **Video sequences:** 1 real recording per class (2–3 for a few) plus 30 noise copies each.
- **Unused webcam data:** the most varied data in the repo, `model/initial build/dataset` (30 webcam recordings × 30 frames for A–S), was **never used** by the newer pipelines.
- **ASL, not GSL:** the data is ASL. GSL is strongly ASL-influenced, but confirm GSL handshapes with a GSL signer or dictionary before describing the model as GSL.

### 5. Video pipeline: missing classes and a time-scale mismatch
- `extract_keypoints_from_videos.py:45-47` silently skips clips shorter than 30 frames. All clips for H, K, L, W and X were that short, so those letters have **no** video data.
- **Training vs inference speed:** training samples 30 frames spread over the **whole clip** (`:49`). The API (`endpoints.py:191-235`) and live script use the **first/last 30 consecutive frames** (~1 s), so the model sees signs at a different speed than it was trained on.
- **Zero padding:** short API inputs are padded with zero vectors (`endpoints.py:231-232`) the model never saw.

### 6. Frames without a hand were trained as letters
- When MediaPipe found no hand, the old code saved zeros and kept the label. The webcam set has **2,533** frames labeled as a letter with no hand in them (Q: 551 of 900, E: 433, G: 381). "Neutral" is 15/16 empty frames.
- The models were partly trained to call "nothing" a letter.

### 7. Why Gemini "worked"
- `endpoints.py:146` and `:252` send any prediction below 90% confidence to Gemini, and Gemini's answer is returned instead.
- In practice most answers in the app came from Gemini, not SignLens.

### 8. The old code no longer runs on current MediaPipe
- Every script uses `mp.solutions.holistic`. MediaPipe 0.10.3x (current) ships only the Tasks API; `mediapipe.solutions` no longer exists, so a fresh Colab install fails immediately.
- `model_endpoints/requirements.txt` pins TensorFlow 2.13 and MediaPipe 0.10.5 to keep it alive.

### 9. Letters and numbers in one model
- In ASL, **O/0, F/9, V/2 and W/6 use the same handshape** (D/1 is very close). A single static model with letters and digits can only guess between them.
- In the held-out test, 437 webcam "O"s were classified as "0".
- The new notebooks train letters and numbers as separate models and leave the choice to an app mode.

## Repository hygiene (affects publishing)
- **`node_modules` committed:** `admin/node_modules` is in git (**20,732 files**), and there is no root `.gitignore`.
- **Large files in git:** datasets, `.MOV` and `.mp4` files are committed directly; `.h5`, `.jpg` and `.png` are in Git LFS. This is why the repo is ~7 GB.
- **Copyright:** the full YouTube video and its trimmed clips are committed (`model/automation_scripts/*.mp4`, `trimmed_clips/`). Remove them from the public repo and cite the source; publishing third-party video needs permission.
- **Personal paths and secrets:** `predict_recorded_video.py:6` contains a personal file path. API-key checks are commented out (`endpoints.py:282`, `:312-316`).

## Per-class data (distinct recordings, which is what matters)

| class | YouTube clips | webcam recordings | video sequences (real) |
|---|---|---|---|
| A–I (no J) | 2–4 | 29–30 | 1–2 (H: 0) |
| J | 4 (incl. transitions) | 0 | 1 |
| K, L | 4 | 30 | 0 |
| M–S | 4 | 30 | 1–3 |
| T–Y | 4 | 0 | 0–2 (W, X: 0) |
| Z | 4 | 0 | 2 |
| digits 0–10 | – | – | 1 each |

## Keep / rewrite / discard

| file(s) | verdict | note |
|---|---|---|
| `model/initial build/dataset/*` | **keep** | best real data in the repo |
| `model/dataset/static_keypoints_optimized`, `video_keypoints_no_face/*/[0-9].npy` | keep | usable after hand-only normalization |
| `video_keypoints_*/**/*_aug_*.npy` (≈1,200 files) | discard | noise copies; augmentation now happens during training |
| `dataset/static_keypoints/` | discard | byte-for-byte duplicate (120/120 files) of A, B, C, Neutral in `initial build/dataset` |
| all `extract_*.py`, `train_*.py`, `run_*.py`, `model_*.py` | rewrite | replaced by `notebooks/signlens_core` + notebooks 01–04 |
| `model_endpoints/app/api_v1/endpoints.py` | rewrite prediction parts | use `signlens_core.inference.SignPredictor` and the exported `labels.json` |
| `automation_scripts/generate_dataset.py`, `sync_videos_to_images.py`, `trim_video.py` | retire | tied to the single YouTube video |
| `model_fusion/` | retire for now | revisit once the individual models generalize |
| YouTube `.mp4` files, `admin/node_modules` | remove from git | copyright, repo size |

## Open questions
1. Who signed the `initial build` webcam recordings? If it was the deaf signer, they are your most valuable data, so record that in the paper.
2. Should the target be GSL, or ASL first? This decides which public datasets count as in-domain.
3. Are the digit `.MOV` files portrait (phone)? The new loader assumes 16:9; set `aspect` accordingly.
