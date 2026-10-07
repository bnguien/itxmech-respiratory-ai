import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path("model_artifacts/lung_sound")

SEGMENTER_DIR = ROOT / "segmenter"
CLASSIFIER_DIR = ROOT / "classifier"


# =========================================================
# TCN SEGMENTER
# =========================================================

print("=== TCN SEGMENTER ===")

segmenter_config_path = SEGMENTER_DIR / "training_config.json"
scaler_path = SEGMENTER_DIR / "scaler.npz"
weights_path = SEGMENTER_DIR / "boundary_tcn.pt"

with open(segmenter_config_path, "r", encoding="utf-8") as f:
    segmenter_config = json.load(f)

print("Config:")
print(json.dumps(segmenter_config, indent=2))

scaler = np.load(scaler_path)

print("\nScaler keys:")
print(scaler.files)

for key in scaler.files:
    print(key, scaler[key].shape)

checkpoint = torch.load(
    weights_path,
    map_location="cpu",
    weights_only=True,
)

print("\nCheckpoint type:")
print(type(checkpoint))

if isinstance(checkpoint, dict):
    print("Checkpoint keys:")
    print(checkpoint.keys())


# =========================================================
# BEATS CLASSIFIER
# =========================================================

print("\n=== BEATS CLASSIFIER ===")

classifier_config_path = CLASSIFIER_DIR / "pipeline_config.json"
classifier_model_path = CLASSIFIER_DIR / "beats_mean_max_dynamic.onnx"

with open(classifier_config_path, "r", encoding="utf-8") as f:
    classifier_config = json.load(f)

print("Pipeline config:")
print(json.dumps(classifier_config, indent=2))

session = ort.InferenceSession(
    str(classifier_model_path),
    providers=["CPUExecutionProvider"],
)

print("\nONNX inputs:")

for item in session.get_inputs():
    print(
        {
            "name": item.name,
            "shape": item.shape,
            "type": item.type,
        }
    )

print("\nONNX outputs:")

for item in session.get_outputs():
    print(
        {
            "name": item.name,
            "shape": item.shape,
            "type": item.type,
        }
    )

print("\nProviders:")
print(session.get_providers())
