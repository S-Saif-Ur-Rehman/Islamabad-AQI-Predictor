"""Local and Hopsworks model registry support."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import re
import hashlib

from config import config


class LocalModelRegistry:
    def __init__(self):
        config.LOCAL_MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    def _horizon_dir(self, horizon_label: str) -> Path:
        if not re.fullmatch(r"[1-9][0-9]*d", horizon_label):
            raise ValueError("Invalid horizon label")
        d = config.LOCAL_MODEL_REGISTRY_DIR / horizon_label
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_model(
        self,
        horizon_label: str,
        model: Any,
        model_type: str,
        metrics: dict,
        feature_columns: list[str],
        is_keras: bool = False,
        scaler: Any = None,
    ) -> Path:
        d = self._horizon_dir(horizon_label)
        if is_keras:
            model_path = d / "model.keras"
            model.save(model_path)
        else:
            model_path = d / "model.joblib"
            joblib.dump(model, model_path)

        has_scaler = scaler is not None
        if has_scaler:
            joblib.dump(scaler, d / "scaler.joblib")

        metadata = {
            "horizon_label": horizon_label,
            "model_type": model_type,
            "metrics": metrics,
            "feature_columns": feature_columns,
            "is_keras": is_keras,
            "has_scaler": has_scaler,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        }
        with open(d / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        return model_path

    def load_model(self, horizon_label: str):
        d = self._horizon_dir(horizon_label)
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No trained model found for horizon '{horizon_label}'. Run the training "
                f"pipeline first: python -m src.pipelines.training_pipeline"
            )
        with open(meta_path) as f:
            metadata = json.load(f)

        if metadata["is_keras"]:
            import tensorflow as tf

            model = tf.keras.models.load_model(d / "model.keras")
        else:
            expected = metadata.get("model_sha256")
            if expected and hashlib.sha256((d / "model.joblib").read_bytes()).hexdigest() != expected:
                raise ValueError("Model artifact integrity check failed")
            model = joblib.load(d / "model.joblib")

        scaler = joblib.load(d / "scaler.joblib") if metadata.get("has_scaler") else None
        return model, metadata, scaler


class HopsworksModelRegistry:
    """Small Hopsworks model registry wrapper."""

    def __init__(self):
        import tempfile
        tempfile.tempdir = str(config.PROJECT_ROOT / ".hopsworks_tmp")
        (config.PROJECT_ROOT / ".hopsworks_tmp").mkdir(parents=True, exist_ok=True)
        import hopsworks

        if not config.HOPSWORKS_API_KEY or not config.HOPSWORKS_PROJECT_NAME:
            raise ValueError(
                "HOPSWORKS_API_KEY and HOPSWORKS_PROJECT_NAME must both be set "
                "in .env when USE_HOPSWORKS=true. Mirrors the same guard "
                "HopsworksFeatureStore already has — without it, hopsworks.login() "
                "can fail with a confusing error or, in a non-interactive CI "
                "runner, hang waiting for interactive input."
            )
        self.project = hopsworks.login(
            api_key_value=config.HOPSWORKS_API_KEY,
            project=config.HOPSWORKS_PROJECT_NAME,
        )
        self.mr = self.project.get_model_registry()

    def save_model(
        self,
        horizon_label: str,
        model: Any,
        model_type: str,
        metrics: dict,
        feature_columns: list[str],
        is_keras: bool = False,
        scaler: Any = None,
    ):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_sha256 = None
            if is_keras:
                model.save(tmp_path / "model.keras")
            else:
                model_path = tmp_path / "model.joblib"
                joblib.dump(model, model_path)
                # Save a hash so we can spot a corrupt model file.
                model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()

            has_scaler = scaler is not None
            if has_scaler:
                joblib.dump(scaler, tmp_path / "scaler.joblib")

            with open(tmp_path / "metadata.json", "w") as f:
                json.dump(
                    {
                        "model_type": model_type,
                        "feature_columns": feature_columns,
                        "is_keras": is_keras,
                        "has_scaler": has_scaler,
                        "model_sha256": model_sha256,
                    },
                    f,
                )

            hw_model = self.mr.python.create_model(
                name=f"aqi_{horizon_label}_model",
                metrics=metrics,
                description=f"AQI forecast model, horizon={horizon_label}, type={model_type}",
            )
            hw_model.save(str(tmp_path))
        return hw_model

    def load_model(self, horizon_label: str):
        hw_model = self.mr.get_best_model(f"aqi_{horizon_label}_model", "rmse", "min")
        model_dir = Path(hw_model.download())
        with open(model_dir / "metadata.json") as f:
            metadata = json.load(f)
        metadata["metrics"] = hw_model.training_metrics

        if metadata["is_keras"]:
            import tensorflow as tf

            model = tf.keras.models.load_model(model_dir / "model.keras")
        else:
            expected = metadata.get("model_sha256")
            actual = hashlib.sha256((model_dir / "model.joblib").read_bytes()).hexdigest()
            if expected and actual != expected:
                raise ValueError(
                    f"Model artifact integrity check failed for horizon "
                    f"'{horizon_label}' (downloaded from Hopsworks Model "
                    "Registry) — its hash doesn't match what was recorded "
                    "at training time."
                )
            model = joblib.load(model_dir / "model.joblib")

        scaler = joblib.load(model_dir / "scaler.joblib") if metadata.get("has_scaler") else None
        return model, metadata, scaler


def get_model_registry():
    if config.USE_HOPSWORKS:
        return HopsworksModelRegistry()
    return LocalModelRegistry()
