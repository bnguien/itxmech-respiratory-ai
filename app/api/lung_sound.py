from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from fastapi import APIRouter, HTTPException, Request

from app.schemas.lung import LungSoundAnalyzeRequest, LungSoundAnalyzeResponse

router = APIRouter(prefix="/lung-sound", tags=["lung-sound"])


@router.post("/analyze", response_model=LungSoundAnalyzeResponse)
def analyze_lung_sound(
    payload: LungSoundAnalyzeRequest,
    request: Request,
) -> LungSoundAnalyzeResponse:
    with TemporaryDirectory() as temporary_directory:
        wav_path = Path(temporary_directory) / "recording.wav"

        try:
            with httpx.Client(follow_redirects=True) as client:
                with client.stream("GET", str(payload.audio_url)) as response:
                    response.raise_for_status()
                    with wav_path.open("wb") as wav_file:
                        for chunk in response.iter_bytes():
                            wav_file.write(chunk)
        except httpx.HTTPError:
            raise HTTPException(
                status_code=502,
                detail="Unable to download audio.",
            ) from None

        try:
            result = request.app.state.lung_pipeline.analyze_file(wav_path)
            return LungSoundAnalyzeResponse(
                recording_id=payload.recording_id,
                **result,
            )
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="Unable to analyze audio.",
            ) from None
