<div align="center">

# RespiratoryCare AI Server

AI inference service for **RespiCare**, an AIoT respiratory healthcare system.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-005CED?style=flat&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Librosa](https://img.shields.io/badge/Librosa-FF6F00?style=flat&logoColor=white)](https://librosa.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

</div>

## Overview

RespiCare AI Server provides APIs for analyzing respiratory data collected from IoT medical devices.

The current development focuses on **lung sound analysis**.

Planned AI capabilities include:

- Lung sound respiratory cycle analysis
- Lung sound classification
- SpO₂ analysis and prediction

## Architecture

```text
Electronic Stethoscope
        │
        │ WAV
        ▼
Supabase Storage
        │
        │ Signed URL
        ▼
Next.js Backend
        │
        │ HTTP
        ▼
RespiratoryCare AI Server
        │
        ├── Audio preprocessing
        ├── Respiratory cycle segmentation
        ├── Feature extraction
        ├── AI inference
        └── Post-processing
                │
                ▼
        Classification Result
                │
                ▼
        Next.js Backend
                │
                ▼
        PostgreSQL / Web UI
```

The AI server does not directly manage patients, visits, authentication, or application data.

## Lung Sound Pipeline

The lung sound pipeline processes one complete WAV recording from a clinical visit.

```text
WAV Recording
    ↓
Audio Validation
    ↓
Preprocessing
    ↓
Respiratory Cycle Segmentation
    ↓
Feature Extraction
    ↓
AI Model
    ↓
Cycle Classification
    ↓
Prediction Results
```

Each respiratory cycle can be classified as:

- Normal
- Crackles
- Wheezes
- Crackles + Wheezes

The model returns results for individual respiratory cycles rather than making a clinical diagnosis.

## API

Initial endpoints:

```http
GET /health
POST /api/v1/lung-sound/analyze
```

Future endpoint:

```http
POST /api/v1/spo2/analyze
```

## Example Lung Sound Response

```json
{
  "recording_id": "uuid",
  "cycles": [
    {
      "start": 0.52,
      "end": 3.1,
      "label": "crackles",
      "confidence": 0.91
    },
    {
      "start": 3.2,
      "end": 6.05,
      "label": "normal",
      "confidence": 0.95
    }
  ]
}
```

## Tech Stack

- Python
- FastAPI
- PyTorch
- ONNX Runtime
- NumPy
- Librosa
- Docker

Additional libraries may be introduced depending on the final inference pipeline.

## Project Structure

```text
itxmech-respiratory-ai/
├── app/
│   ├── main.py
│   ├── api/
│   │   └── lung_sound.py
│   ├── models/
│   │   └── lung_model.py
│   ├── pipelines/
│   │   └── lung_pipeline.py
│   ├── preprocessing/
│   │   └── audio.py
│   ├── schemas/
│   │   └── lung.py
│   └── core/
│       └── config.py
│
├── model_artifacts/
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Model Artifacts

Large model files should not be committed directly to Git.

Examples:

```text
*.pt
*.pth
*.onnx
*.ckpt
```

Model artifacts should be stored separately and loaded by the AI service during deployment.

## Development

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI development server:

```bash
uvicorn app.main:app --reload
```

Default local URL:

```text
http://localhost:8000
```

API documentation:

```text
http://localhost:8000/docs
```

## Docker

Build the image:

```bash
docker build -t respiratorycare-ai .
```

Run:

```bash
docker run -p 8000:8000 respiratorycare-ai
```

Or with Docker Compose:

```bash
docker compose up -d --build
```

## Deployment

During development, the AI server can run locally using Docker.

```text
MacBook / Development Machine
└── Docker
    └── RespiratoryCare AI Server
```

For production, the same Docker image can later be deployed to a VPS or other cloud infrastructure.

```text
VPS
└── Docker
    └── RespiratoryCare AI Server
```

## Related Project

RespiratoryCare Web:

```text
itxmech-respiratory-web
```

The web application handles:

- Authentication
- Patient management
- Visit management
- Lung recording management
- Supabase integration
- Doctor-facing UI

The AI repository is intentionally kept separate from the web application.

## Disclaimer

The AI models in RespiCare are designed to support respiratory sound analysis and clinical decision support.

They are **not intended to independently diagnose medical conditions**.
