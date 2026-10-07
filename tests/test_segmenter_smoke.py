import argparse
from pathlib import Path
from pprint import pprint

from app.core.config import get_settings
from app.models.lung_sound.segmenter import RespiratoryCycleSegmenter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path", type=Path)
    args = parser.parse_args()

    if not args.wav_path.exists():
        raise FileNotFoundError(f"WAV not found: {args.wav_path}")

    settings = get_settings()

    artifact_dir = Path(settings.lung_segmenter_weights_path).parent

    print("=== LOAD SEGMENTER ===")
    print(f"Artifact dir: {artifact_dir}")

    segmenter = RespiratoryCycleSegmenter(
        artifact_dir=artifact_dir,
    )

    print(f"Device: {segmenter.device}")
    print("Model loaded successfully ✅")

    print("\n=== SEGMENT WAV ===")
    print(f"WAV: {args.wav_path}")

    result = segmenter.segment_file(args.wav_path)

    print("\n=== RESULT ===")
    pprint(result)


if __name__ == "__main__":
    main()