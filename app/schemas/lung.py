from typing import Literal
from uuid import UUID

from pydantic import BaseModel, HttpUrl


class LungSoundAnalyzeRequest(BaseModel):
    recording_id: UUID
    audio_url: HttpUrl


class RespiratoryCyclePrediction(BaseModel):
    index: int
    start: float
    end: float
    duration: float
    status: Literal["processed", "unprocessable"]
    label: Literal["normal", "crackle", "wheeze", "both"] | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    n_frames: int | None = None
    reason: str | None = None


class LungSoundAnalyzeResponse(BaseModel):
    recording_id: UUID
    boundaries: list[float]
    cycles: list[RespiratoryCyclePrediction]
