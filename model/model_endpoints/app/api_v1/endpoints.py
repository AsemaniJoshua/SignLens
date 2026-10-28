"""SignLens API v1 routes, served by the v2 models.

Same URLs and response fields as before, so the mobile app keeps working:
    POST /api/v1/predict?type=image|video[&mode=letters|digits]   (multipart field "file")
    WS   /api/v1/predict-stream?api_key=...&type=stream|video[&mode=letters|digits]
Responses: {"prediction", "confidence", "fallback", "model_used", ...}

All the sign-recognition logic lives in app/models/signlens_service.py.
"""
import asyncio
import base64
import json
import os
import urllib.error
import urllib.request

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect, status

from app.models.signlens_service import NO_HAND_MSG, UNSURE_MSG, ModelsNotReady, get_service
from .schemas import InputType, Mode, PredictResponse

router = APIRouter()

# ---------------------------------------------------------------------------
# Optional Gemini fallback. OFF by default: in the old API every answer below
# 90% confidence went to Gemini, so most results were not SignLens at all.
# Enable with GEMINI_API_KEY and GEMINI_FALLBACK_BELOW=0.5 (for example).
# Answers that came from Gemini are always marked fallback=true.
# ---------------------------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")


def _gemini_threshold() -> float:
    try:
        return float(os.getenv("GEMINI_FALLBACK_BELOW", "0"))
    except ValueError:
        return 0.0


def predict_with_gemini(file_bytes: bytes, mime_type: str):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [
        {"text": ("Identify the American Sign Language (ASL) alphabet letter (A-Z) or digit (0-10) "
                  "shown in this media. Output ONLY the single letter or number, or 'None' if no sign is shown.")},
        {"inlineData": {"mimeType": mime_type, "data": base64.b64encode(file_bytes).decode()}},
    ]}]}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            parts = json.loads(resp.read())["candidates"][0]["content"]["parts"]
            return parts[0]["text"].strip().replace(".", "")
    except Exception as e:  # noqa: BLE001
        print(f"[Gemini fallback failed] {e}")
        return None


def _maybe_fallback(result: dict, file_bytes: bytes, mime_type: str) -> dict:
    threshold = _gemini_threshold()
    unsure = result["prediction"] in (NO_HAND_MSG, UNSURE_MSG)
    if threshold > 0 and (unsure or result["confidence"] < threshold):
        label = predict_with_gemini(file_bytes, mime_type)
        if label:
            return {**result, "prediction": label, "fallback": True, "model_used": GEMINI_MODEL}
    return result


def _service_or_503():
    try:
        return get_service()
    except ModelsNotReady as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))


# ---------------------------------------------------------------------------
@router.post("/predict", response_model=PredictResponse)
def predict_media(type: InputType, file: UploadFile = File(...), mode: Mode = Query(Mode.letters)):
    if type not in (InputType.image, InputType.video):
        raise HTTPException(status_code=400, detail="This endpoint only supports type=image or type=video.")
    service = _service_or_503()
    file_bytes = file.file.read()
    try:
        if type == InputType.image:
            result = service.predict_image(file_bytes, mode.value)
            result = _maybe_fallback(result, file_bytes, "image/jpeg")
        else:
            result = service.predict_video(file_bytes, mode.value)
            result = _maybe_fallback(result, file_bytes, "video/mp4")
    except ValueError as e:  # unreadable file, unknown mode
        raise HTTPException(status_code=400, detail=str(e))
    print(f"[predict {type.value}/{mode.value}] {result['prediction']} ({result['confidence']:.2f}) "
          f"via {result['model_used']}")
    return result


@router.websocket("/predict-stream")
async def predict_stream(websocket: WebSocket):
    """Send JPEG/PNG frame bytes; receive one JSON message per frame:
    {"prediction": "A", "confidence": 0.93, "stable": true, ...}
    or {"status": "No hand detected...", "prediction": null, ...}.
    Show a letter to the user only when "stable" is true."""
    await websocket.accept()
    params = websocket.query_params
    type_ = params.get("type")
    mode = params.get("mode", "letters")
    if not params.get("api_key") or type_ not in (InputType.video.value, InputType.stream.value):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    try:
        session = get_service().stream(mode)
    except (ModelsNotReady, ValueError) as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return
    try:
        while True:
            frame = await websocket.receive_bytes()
            try:
                result = await asyncio.to_thread(session.push, frame)
            except ValueError as e:
                await websocket.send_json({"error": str(e)})
                continue
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        print(f"[stream error] {e}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:  # noqa: BLE001
            pass
    finally:
        session.close()
