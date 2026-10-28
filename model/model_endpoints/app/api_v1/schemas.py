from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel


class InputType(str, Enum):
    image = "image"
    video = "video"
    stream = "stream"


class Mode(str, Enum):
    """Letters and numbers are separate models: in ASL, O/0, V/2, W/6 and F/9
    use the same handshape, so the app must say which one it expects."""
    letters = "letters"
    digits = "digits"


class PredictRequest(BaseModel):
    type: InputType


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    fallback: bool = False
    model_used: str = "SignLens-v2"
    mode: str = "letters"
    top3: List[Tuple[str, float]] = []
    reason: Optional[str] = None
