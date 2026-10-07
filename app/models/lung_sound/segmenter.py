from __future__ import annotations

import json
from numbers import Real
from pathlib import Path
from typing import Any, TypedDict

import librosa
import numpy as np
from numpy.typing import ArrayLike, NDArray
import soundfile as sf
from scipy.signal import butter, find_peaks, sosfiltfilt
import torch
from torch import nn


_DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts/final_segmenter"


class CycleInterval(TypedDict):
    """Interval between two consecutive predicted boundaries, in seconds."""

    start: float
    end: float


class SegmentationResult(TypedDict):
    """JSON-compatible boundary timestamps and candidate cycle intervals."""

    boundaries: list[float]
    cycles: list[CycleInterval]


class Chomp1d(nn.Module):
    """Remove the right padding of a causal convolution, as in Step 8."""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.size] if self.size > 0 else x


class TemporalBlock(nn.Module):
    """The unchanged two-convolution residual block used during training."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=pad, dilation=dilation
        )
        self.chomp1 = Chomp1d(pad)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=pad, dilation=dilation
        )
        self.chomp2 = Chomp1d(pad)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.chomp1(self.conv1(x)))
        out = self.dropout(out)
        out = torch.relu(self.chomp2(self.conv2(out)))
        out = self.dropout(out)
        return torch.relu(out + self.residual(x))


class BoundaryTCN(nn.Module):
    """Reconstruct the training architecture with identical state-dict keys."""

    def __init__(
        self,
        in_features: int,
        channels: list[int],
        kernel: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks = []
        c_in = in_features
        for i, c_out in enumerate(channels):
            blocks.append(
                TemporalBlock(c_in, c_out, kernel, dilation=2**i, dropout=dropout)
            )
            c_in = c_out
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Conv1d(c_in, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.tcn(x)).squeeze(1)


class RespiratoryCycleSegmenter:
    """Load once and reuse the frozen final segmenter for whole recordings.

    Args:
        artifact_dir: Directory containing boundary_tcn.pt, scaler.npz and
            training_config.json. The default is relative to this source file's
            repository, independent of the working directory. An explicitly
            supplied relative path is resolved against the working directory.

    CUDA is selected when available, otherwise CPU. The saved model, scaler,
    and configuration are read only during initialization. Invalid artifacts
    raise an error rather than substituting defaults or changing parameters.
    """

    def __init__(self, artifact_dir: str | Path = _DEFAULT_ARTIFACT_DIR) -> None:
        self.artifact_dir = Path(artifact_dir).expanduser().resolve()
        with (self.artifact_dir / "training_config.json").open(encoding="utf-8") as f:
            training_config = json.load(f)
        self._validate_config(training_config)
        frozen = training_config["frozen_config"]
        preprocessing = frozen["preprocessing"]
        tcn = frozen["tcn"]
        post = frozen["post_processing"]

        self._sr = preprocessing["sr_target"]
        self._bandpass_low = preprocessing["bandpass_low_hz"]
        self._bandpass_high = preprocessing["bandpass_high_hz"]
        self._bandpass_order = preprocessing["bandpass_order"]
        self._frame_length_s = preprocessing["frame_length_s"]
        self._hop_length_s = preprocessing["hop_length_s"]
        self._peak_height = post["peak_height"]
        self._min_peak_distance_s = post["min_peak_distance_s"]
        self._prominence = post["prominence"]

        with np.load(self.artifact_dir / "scaler.npz", allow_pickle=False) as scaler:
            self._mean = scaler["mean"].copy()
            self._std = scaler["std"].copy()
        for name, values in (("mean", self._mean), ("std", self._std)):
            if values.shape != (4,) or values.dtype != np.dtype(np.float32):
                raise ValueError(f"Saved scaler {name} must contain four float32 values.")
            if not np.isfinite(values).all():
                raise ValueError(f"Saved scaler {name} contains non-finite values.")
        if not (self._std > 0).all():
            raise ValueError("Saved scaler std must be strictly positive.")
        self._mean.setflags(write=False)
        self._std.setflags(write=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = BoundaryTCN(
            len(preprocessing["features"]), tcn["channels"], tcn["kernel"], tcn["dropout"]
        ).to(self.device)
        state = torch.load(
            self.artifact_dir / "boundary_tcn.pt",
            map_location=self.device,
            weights_only=True,
        )
        self._model.load_state_dict(state, strict=True)
        if not all(torch.isfinite(value).all() for value in self._model.state_dict().values()):
            raise ValueError("Model checkpoint contains non-finite weights.")
        self._model.eval()
        self._model.requires_grad_(False)

    @staticmethod
    def _validate_config(training: dict[str, Any]) -> None:
        """Reject configuration drift without overriding any saved value."""
        try:
            frozen = training["frozen_config"]
            if frozen["model"] != "BoundaryTCN":
                raise ValueError("Expected the frozen BoundaryTCN model.")
            expected_preprocessing = {
                "sr_target": 4000,
                "bandpass_low_hz": 80,
                "bandpass_high_hz": 1800,
                "bandpass_order": 4,
                "frame_length_s": 0.05,
                "hop_length_s": 0.01,
                "features": ["log_rms", "zcr", "spectral_centroid", "log1p_spectral_flux"],
            }
            expected_tcn = {
                "channels": [24, 24, 24, 24],
                "kernel": 5,
                "dropout": 0.1,
                "final_train_epochs": 9,
            }
            for section, expected in (
                ("preprocessing", expected_preprocessing),
                ("tcn", expected_tcn),
                ("post_processing", {
                    "peak_height": 0.75,
                    "min_peak_distance_s": 0.8,
                    "prominence": None,
                }),
            ):
                for key, value in expected.items():
                    if frozen[section][key] != value:
                        raise ValueError(f"Frozen configuration mismatch: {section}.{key}")
            if training["checkpoint_epoch"] != 9 or training["epochs_completed"] != 9:
                raise ValueError("Expected the final epoch-9 checkpoint.")
            if training["feature_order"] != expected_preprocessing["features"]:
                raise ValueError("Saved feature order differs from training.")
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed or incomplete training_config.json.") from exc

    def _prepare_audio(self, audio: ArrayLike, sr: Real) -> NDArray[np.float32]:
        """Validate, downmix, resample, filter and peak-normalize as in Step 8."""
        if isinstance(sr, (bool, np.bool_)) or not isinstance(sr, Real):
            raise ValueError("sr must be a finite, positive sample rate in Hz.")
        if not np.isfinite(sr) or sr <= 0:
            raise ValueError("sr must be a finite, positive sample rate in Hz.")
        try:
            values = np.asarray(audio)
        except (TypeError, ValueError) as exc:
            raise ValueError("audio must be a rectangular real-valued waveform.") from exc
        if values.ndim not in (1, 2) or values.size == 0:
            raise ValueError("audio must be nonempty with shape (samples,) or (samples, channels).")
        if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise ValueError("audio must contain only finite real numeric samples.")
        wav = np.asarray(values, dtype=np.float32)
        if not np.isfinite(wav).all():
            raise ValueError("audio samples must be representable as float32.")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self._sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self._sr).astype(np.float32)

        nyq = self._sr / 2.0
        low = max(self._bandpass_low / nyq, 1e-4)
        high = min(self._bandpass_high / nyq, 0.999)
        sos = butter(self._bandpass_order, [low, high], btype="bandpass", output="sos")
        # Preserve the training pipeline's exact short-waveform behavior.
        wav = sosfiltfilt(sos, wav).astype(np.float32) if len(wav) >= 32 else wav.astype(np.float32)
        if not len(wav) or not np.isfinite(wav).all():
            raise ValueError("Audio preprocessing produced empty or non-finite samples.")
        peak = np.max(np.abs(wav)) if len(wav) else 0.0
        if peak > 1e-9:
            wav = wav / peak
        return wav

    def _frame_features(
        self, wav: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Compute the four training features and original-relative frame centers."""
        sr = self._sr
        frame_length = max(8, int(round(self._frame_length_s * sr)))
        hop_length = max(1, int(round(self._hop_length_s * sr)))
        if len(wav) < frame_length:
            wav = np.pad(wav, (0, frame_length - len(wav)))

        frames = librosa.util.frame(wav, frame_length=frame_length, hop_length=hop_length).T.copy()
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
        zcr = np.mean((frames[:, 1:] * frames[:, :-1]) < 0, axis=1)
        window = np.hanning(frame_length).astype(np.float32)
        spec = np.abs(np.fft.rfft(frames * window[None, :], axis=1))
        spec_norm = spec / (spec.sum(axis=1, keepdims=True) + 1e-9)
        freqs = np.fft.rfftfreq(frame_length, d=1.0 / sr)
        centroid = np.sum(spec_norm * freqs[None, :], axis=1) / (sr / 2.0)
        flux = np.zeros(len(frames), dtype=np.float64)
        if len(frames) > 1:
            flux[1:] = np.sum(np.diff(spec_norm, axis=0) ** 2, axis=1)
        feats = np.stack(
            [np.log(rms + 1e-6), zcr, centroid, np.log1p(flux)], axis=1
        ).astype(np.float32)
        times = (np.arange(len(frames)) * hop_length + frame_length / 2.0) / sr
        return feats, times.astype(np.float32)

    @torch.no_grad()
    def segment(self, audio: ArrayLike, sr: Real) -> SegmentationResult:
        """Predict boundaries and intervals from an entire input recording.

        Args:
            audio: Finite numeric samples, shaped (samples,) for mono or
                (samples, channels) for multichannel audio. Channels are averaged
                after conversion to float32, exactly as for training files.
                Channels-first input must be transposed by the caller.
            sr: Original positive sample rate in Hz; resampling is automatic.

        Returns:
            Boundary timestamps in seconds and intervals between consecutive
            boundaries. Fewer than two boundaries gives an empty cycles list.
            No artificial endpoints, trimming, or additional filtering is added.

        Raises:
            ValueError: Invalid/empty waveform or invalid sample rate.
            RuntimeError: Non-finite features or model scores.
        """
        wav = self._prepare_audio(audio, sr)
        features, frame_times = self._frame_features(wav)
        features_norm = ((features - self._mean) / self._std).astype(np.float32)
        if not np.isfinite(features_norm).all():
            raise RuntimeError("Feature extraction produced non-finite values.")
        x = torch.from_numpy(features_norm.T[None]).float().to(self.device)
        self._model.eval()
        probs = torch.sigmoid(self._model(x))[0].cpu().numpy()
        if not np.isfinite(probs).all():
            raise RuntimeError("Model produced non-finite boundary scores.")
        distance_frames = max(1, int(round(self._min_peak_distance_s / self._hop_length_s)))
        peak_indices, _ = find_peaks(
            probs,
            height=self._peak_height,
            distance=distance_frames,
            prominence=self._prominence,
        )
        boundaries = frame_times[peak_indices].astype(float).tolist()
        cycles: list[CycleInterval] = [
            {"start": start, "end": end}
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        return {"boundaries": boundaries, "cycles": cycles}

    def segment_file(self, path: str | Path) -> SegmentationResult:
        """Read an audio file with soundfile and segment it with cached artifacts.

        File decoding errors propagate from soundfile. This method reads only
        the supplied audio file; it does not load metadata or annotations.
        """
        audio, sr = sf.read(Path(path).expanduser(), always_2d=False)
        return self.segment(audio, sr)


if __name__ == "__main__":
    # Explicit usage: python -m src.segmentation.inference example.wav
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Audio recording to segment")
    parser.add_argument("--artifact-dir", type=Path, default=_DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()
    segmenter = RespiratoryCycleSegmenter(artifact_dir=args.artifact_dir)
    result = segmenter.segment_file(args.audio)
    print(json.dumps(result, indent=2))
