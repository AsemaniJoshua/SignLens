# Put the trained models here

Copy these files from Google Drive (`MyDrive/SignLens/models/`) into this folder:

| file | needed? |
|---|---|
| `static_letters_model.tflite` + `static_letters_labels.json` | **required** (letters) |
| `static_digits_model.tflite` + `static_digits_labels.json` | optional (`mode=digits`) |
| `motion_model.tflite` + `motion_labels.json` | optional (J and Z) |

The `.keras` files are not needed by the server.

`hand_landmarker.task` (MediaPipe's hand model, about 8 MB) is downloaded here
automatically the first time the server starts. If the server has no internet
access, download it once from
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
and put it in this folder.

Check that everything loaded: open http://localhost:8000/ and look for
`"status": "ok"` and the list of modes.
