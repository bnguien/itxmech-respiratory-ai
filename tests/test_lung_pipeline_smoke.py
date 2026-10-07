from pprint import pprint

from app.pipelines.lung_pipeline import LungSoundPipeline


def main() -> None:
    wav_path = "/Users/beong/Downloads/test-16k-mono.wav"

    print("=== LOAD FULL LUNG SOUND PIPELINE ===")

    pipeline = LungSoundPipeline()

    print("Pipeline loaded successfully ✅")

    print("\n=== ANALYZE RECORDING ===")

    result = pipeline.analyze_file(wav_path)

    print("\n=== RESULT ===")
    pprint(result)


if __name__ == "__main__":
    main()
