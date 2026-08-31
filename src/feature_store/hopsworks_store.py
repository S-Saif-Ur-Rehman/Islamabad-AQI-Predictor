"""Hopsworks feature store implementation."""
from __future__ import annotations

import pandas as pd

from config import config
from src.feature_store.base import FeatureStore


class HopsworksFeatureStore(FeatureStore):
    def __init__(self):
        import tempfile
        # Use a local temp folder so Windows paths stay valid.
        tempfile.tempdir = str(config.PROJECT_ROOT / ".hopsworks_tmp")
        (config.PROJECT_ROOT / ".hopsworks_tmp").mkdir(parents=True, exist_ok=True)
        import hopsworks  # local import: optional dependency

        if not config.HOPSWORKS_API_KEY or not config.HOPSWORKS_PROJECT_NAME:
            raise ValueError(
                "HOPSWORKS_API_KEY and HOPSWORKS_PROJECT_NAME must be set in .env "
                "when USE_HOPSWORKS=true."
            )
        self.project = hopsworks.login(
            api_key_value=config.HOPSWORKS_API_KEY,
            project=config.HOPSWORKS_PROJECT_NAME,
        )
        self.fs = self.project.get_feature_store()

    def _get_or_create_fg(self, feature_group_name: str, df: pd.DataFrame, primary_key: list[str]):
        event_time = "timestamp" if "timestamp" in df.columns else (
            "date" if "date" in df.columns else None
        )
        return self.fs.get_or_create_feature_group(
            name=feature_group_name,
            version=1,
            description=f"AQI predictor feature group: {feature_group_name}",
            primary_key=primary_key,
            event_time=event_time,
            online_enabled=True,
            statistics_config={"enabled": False},
            # Use HUDI for a lighter client setup.
            time_travel_format="HUDI",
        )

    def insert(self, feature_group_name: str, df: pd.DataFrame, primary_key: list[str]) -> None:
        fg = self._get_or_create_fg(feature_group_name, df, primary_key)
        # Start materialization ourselves so reads are ready right away.
        fg.insert(
            self._align_to_schema(df, fg),
            write_options={"wait_for_job": True, "start_offline_materialization": False},
        )
        self._ensure_materialized(fg)

    @staticmethod
    def _ensure_materialized(fg) -> None:
        """Run the offline materialization job and wait for it to finish."""
        execution = fg.materialization_job.run(await_termination=True)
        if execution is not None and not getattr(execution, "success", True):
            raise RuntimeError(
                f"Offline materialization job for feature group '{fg.name}' "
                "did not succeed — check the job's execution logs in the "
                "Hopsworks UI before retrying reads or training."
            )

    @staticmethod
    def _align_to_schema(df: pd.DataFrame, feature_group: object) -> pd.DataFrame:
        """Make pandas dtypes match the Hopsworks schema."""
        aligned = df.copy()
        for column in feature_group.columns:
            name, data_type = column.name, column.type.lower()
            if name not in aligned.columns:
                continue
            if data_type == "double":
                aligned[name] = pd.to_numeric(aligned[name], errors="raise").astype("float64")
            elif data_type == "bigint":
                if aligned[name].isna().any():
                    raise ValueError(f"{name} cannot be null; Hopsworks expects bigint")
                aligned[name] = pd.to_numeric(aligned[name], errors="raise").astype("int64")
            elif data_type == "int":
                if aligned[name].isna().any():
                    raise ValueError(f"{name} cannot be null; Hopsworks expects int")
                aligned[name] = pd.to_numeric(aligned[name], errors="raise").astype("int32")
            elif data_type == "timestamp":
                aligned[name] = pd.to_datetime(aligned[name], utc=True)
        return aligned

    def read(self, feature_group_name: str, online: bool = True) -> pd.DataFrame:
        fg = self.fs.get_feature_group(name=feature_group_name, version=1)
        try:
            return fg.read(online=online)
        except Exception:
            return fg.read()

    def get_or_create_feature_view(self, feature_group_name: str):
        """Create a feature view for training and reading data."""
        fg = self.fs.get_feature_group(name=feature_group_name, version=1)
        labels = (
            [f"aqi_target_{days}d" for days in config.FORECAST_HORIZONS_DAYS]
            if feature_group_name == config.DAILY_FEATURES_FG
            else None
        )
        return self.fs.get_or_create_feature_view(
            name=f"{feature_group_name}_view",
            version=1,
            query=fg.select_all(),
            labels=labels,
            description=f"AQI Predictor Feature View for {feature_group_name}",
        )

    def get_training_data(self, feature_group_name: str) -> pd.DataFrame:
        """Read training data from the feature view."""
        fv = self.get_or_create_feature_view(feature_group_name)
        try:
            X, y = fv.training_data()
            if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
                return pd.concat([X, y], axis=1)
        except Exception:
            pass
        return self.read(feature_group_name)
