from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RespiratoryCare AI Server"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"

    # Lung sound - TCN Segmenter
    lung_segmenter_weights_path: str = (
        "model_artifacts/lung_sound/segmenter/boundary_tcn.pt"
    )
    lung_segmenter_scaler_path: str = "model_artifacts/lung_sound/segmenter/scaler.npz"
    lung_segmenter_config_path: str = (
        "model_artifacts/lung_sound/segmenter/training_config.json"
    )

    # Lung sound - BEATs Classifier
    lung_classifier_model_path: str = (
        "model_artifacts/lung_sound/classifier/beats_mean_max_dynamic.onnx"
    )
    lung_classifier_config_path: str = (
        "model_artifacts/lung_sound/classifier/pipeline_config.json"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
