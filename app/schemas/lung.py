from uuid import UUID

from pydantic import BaseModel, HttpUrl


class LungSoundAnalyzeRequest(BaseModel):
    recording_id: UUID
    audio_url: HttpUrl


class RespiratoryCyclePrediction(BaseModel):
    start: float
    end: float
    label: str
    confidence: float


class LungSoundAnalyzeResponse(BaseModel):
    recording_id: UUID
    cycles: list[RespiratoryCyclePrediction]
