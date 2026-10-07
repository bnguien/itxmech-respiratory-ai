from pathlib import Path

import soundfile as sf

from app.core.config import get_settings
from app.models.lung_sound.classifier import (
    CycleTooShortError,
    LungSoundClassifier,
)
from app.models.lung_sound.segmenter import RespiratoryCycleSegmenter


class LungSoundPipeline:
    def __init__(self) -> None:
        settings = get_settings()

        segmenter_dir = Path(settings.lung_segmenter_weights_path).parent

        self.segmenter = RespiratoryCycleSegmenter(
            artifact_dir=segmenter_dir,
        )

        self.classifier = LungSoundClassifier(
            model_path=settings.lung_classifier_model_path,
            config_path=settings.lung_classifier_config_path,
        )

    def analyze_file(self, wav_path: str | Path) -> dict:
        wav_path = Path(wav_path)

        if not wav_path.exists():
            raise FileNotFoundError(f"WAV not found: {wav_path}")

        # 1. TCN: tìm boundaries và respiratory cycles
        segmentation = self.segmenter.segment_file(wav_path)

        # 2. Đọc WAV gốc một lần
        audio, original_sr = sf.read(
            wav_path,
            dtype="float32",
            always_2d=False,
        )

        results = []

        # 3. Phân loại từng cycle bằng BEATs
        for index, cycle in enumerate(segmentation["cycles"]):
            start = float(cycle["start"])
            end = float(cycle["end"])

            start_sample = int(start * original_sr)
            end_sample = int(end * original_sr)

            cycle_audio = audio[start_sample:end_sample]

            try:
                prediction = self.classifier.predict(
                    cycle_audio,
                    original_sr,
                )

                results.append(
                    {
                        "index": index,
                        "start": start,
                        "end": end,
                        "duration": end - start,
                        "status": "processed",
                        **prediction,
                    }
                )

            except CycleTooShortError as exc:
                results.append(
                    {
                        "index": index,
                        "start": start,
                        "end": end,
                        "duration": end - start,
                        "status": "unprocessable",
                        "reason": str(exc),
                        "label": None,
                        "confidence": None,
                        "probabilities": None,
                    }
                )

        return {
            "boundaries": segmentation["boundaries"],
            "cycles": results,
        }
