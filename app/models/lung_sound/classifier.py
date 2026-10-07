import json
import math
from math import gcd
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio
from scipy.signal import resample_poly


class CycleTooShortError(ValueError):
    pass


class LungSoundClassifier:
    def __init__(
        self,
        model_path: str | Path,
        config_path: str | Path,
    ) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)

        with self.config_path.open(encoding="utf-8") as f:
            config = json.load(f)

        preprocessing = config["preprocessing"]

        self.sr = int(preprocessing["sr"])
        self.bins = int(preprocessing["bins"])
        self.frame_length = float(preprocessing["frame_length"])
        self.frame_shift = float(preprocessing["frame_shift"])
        self.scale = float(preprocessing["scale"])
        self.mean = float(preprocessing["mean"])
        self.std = float(preprocessing["std"])

        self.patch_size = int(config["patch_size"])
        self.labels = list(config["labels"])

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            return audio.astype(np.float32)

        if audio.ndim == 2:
            return audio.mean(axis=1, dtype=np.float32)

        raise ValueError("Audio must have shape [samples] or [samples, channels].")

    def _resample(
        self,
        audio: np.ndarray,
        original_sr: int,
    ) -> np.ndarray:
        if original_sr == self.sr:
            return audio.astype(np.float32)

        divisor = gcd(int(original_sr), self.sr)

        up = self.sr // divisor
        down = int(original_sr) // divisor

        return resample_poly(audio, up, down).astype(np.float32)

    def _official_fbank(self, audio: np.ndarray) -> np.ndarray:
        waveform = torch.from_numpy(audio).float().unsqueeze(0)

        # BEATs official preprocessing
        waveform = waveform * self.scale

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            num_mel_bins=self.bins,
            sample_frequency=self.sr,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
        )

        fbank = (fbank - self.mean) / (2 * self.std)

        return fbank.numpy().astype(np.float32)

    def _prepare_inputs(
        self,
        audio: np.ndarray,
        original_sr: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        audio = self._to_mono(audio)
        audio = self._resample(audio, original_sr)

        fbank = self._official_fbank(audio)

        real_frames = fbank.shape[0]

        if real_frames < self.patch_size:
            raise CycleTooShortError(
                f"Cycle has only {real_frames} Fbank frames; "
                f"minimum is {self.patch_size}."
            )

        padded_frames = math.ceil(real_frames / self.patch_size) * self.patch_size

        padded = np.zeros(
            (padded_frames, self.bins),
            dtype=np.float32,
        )

        padded[:real_frames] = fbank

        fbank_input = padded[np.newaxis, ...]
        lengths = np.array([real_frames], dtype=np.int64)

        return fbank_input, lengths

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - np.max(logits, axis=-1, keepdims=True)

        exp = np.exp(logits)

        return exp / np.sum(exp, axis=-1, keepdims=True)

    def predict(
        self,
        audio: np.ndarray,
        original_sr: int,
    ) -> dict:
        fbank, lengths = self._prepare_inputs(
            audio,
            original_sr,
        )

        logits = self.session.run(
            ["logits"],
            {
                "fbank": fbank,
                "lengths": lengths,
            },
        )[0]

        probabilities = self._softmax(logits)[0]

        class_index = int(np.argmax(probabilities))

        return {
            "label": self.labels[class_index],
            "confidence": float(probabilities[class_index]),
            "probabilities": {
                label: float(probabilities[index])
                for index, label in enumerate(self.labels)
            },
            "n_frames": int(lengths[0]),
        }

    def predict_file_cycle(
        self,
        wav_path: str | Path,
        start: float,
        end: float,
    ) -> dict:
        audio, original_sr = sf.read(
            wav_path,
            dtype="float32",
            always_2d=False,
        )

        start_sample = int(start * original_sr)
        end_sample = int(end * original_sr)

        cycle = audio[start_sample:end_sample]

        if cycle.size == 0:
            raise ValueError("Cycle contains no audio samples.")

        return self.predict(
            cycle,
            original_sr,
        )
