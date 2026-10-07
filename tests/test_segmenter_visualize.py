from pathlib import Path
import sys

import librosa
import matplotlib.pyplot as plt
import numpy as np

# Đổi import này theo đúng project của bạn
from app.models.lung_sound.segmenter import RespiratoryCycleSegmenter

# hoặc copy đúng dòng import từ test_segmenter_smoke.py


def main(wav_path: str):
    wav_file = Path(wav_path).expanduser().resolve()
    if not wav_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {wav_file}")

    print("=== LOAD SEGMENTER ===")
    segmenter = RespiratoryCycleSegmenter(
        artifact_dir="model_artifacts/lung_sound/segmenter"
    )

    print("=== RUN SEGMENTATION ===")
    result = segmenter.segment_file(wav_file)

    boundaries = result["boundaries"]
    cycles = result["cycles"]

    print(f"Boundaries: {len(boundaries)}")
    print(f"Cycles: {len(cycles)}")

    print("=== LOAD AUDIO FOR PLOT ===")
    y, sr = librosa.load(wav_file, sr=None, mono=True)
    times = np.arange(len(y)) / sr

    print("=== DRAW FIGURE ===")
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(times, y, linewidth=0.8)
    ax.set_title("Respiratory Cycle Segmentation")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    # Tô từng chu kỳ
    for i, cycle in enumerate(cycles):
        start = cycle["start"]
        end = cycle["end"]
        ax.axvspan(start, end, alpha=0.15)
        ax.text(
            (start + end) / 2,
            ax.get_ylim()[1] * 0.85,
            f"C{i+1}",
            ha="center",
            va="center",
            fontsize=8,
        )

    # Vẽ boundaries
    for b in boundaries:
        ax.axvline(b, linestyle="--", linewidth=1)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = wav_file.with_name(wav_file.stem + "_segmentation.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            'Usage: PYTHONPATH=. python tests/test_segmenter_visualize.py "/path/to/file.wav"'
        )
        sys.exit(1)

    main(sys.argv[1])
