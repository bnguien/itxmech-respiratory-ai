from pathlib import Path
from pprint import pprint

from app.core.config import get_settings
from app.models.lung_sound.classifier import LungSoundClassifier


def main() -> None:
    settings = get_settings()

    wav_path = Path("/Users/beong/Downloads/test-16k-mono.wav")

    # Lấy cycle đầu tiên từ kết quả TCN trước đó
    start = 0.07500000298023224
    end = 1.184999942779541

    print("=== LOAD BEATS CLASSIFIER ===")

    classifier = LungSoundClassifier(
        model_path=settings.lung_classifier_model_path,
        config_path=settings.lung_classifier_config_path,
    )

    print("Model loaded successfully ")
    print(f"WAV: {wav_path}")
    print(f"Cycle: {start:.3f}s -> {end:.3f}s")

    print("\n=== CLASSIFY CYCLE ===")

    result = classifier.predict_file_cycle(
        wav_path=wav_path,
        start=start,
        end=end,
    )

    print("\n=== RESULT ===")
    pprint(result)


if __name__ == "__main__":
    main()
