"""Generate the forecast from the latest daily data."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import config
from src.features.aqi_calculator import aqi_category
from src.feature_store.base import get_feature_store
from src.models.explain import explain_prediction, top_feature_contributions
from src.models.model_registry import get_model_registry
from src.utils.alerts import check_alert
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _predict_one(model, metadata: dict, scaler, row_df: pd.DataFrame) -> float:
    feature_cols = metadata["feature_columns"]
    X = row_df[feature_cols]
    if metadata["is_keras"]:
        X_scaled = scaler.transform(X) if scaler is not None else X.values
        pred = model.predict(X_scaled, verbose=0).flatten()[0]
    else:
        pred = model.predict(X)[0]
    return float(pred)


def run() -> dict:
    store = get_feature_store()
    daily_df = store.read(config.DAILY_FEATURES_FG)
    if daily_df.empty:
        raise RuntimeError(
            "No daily features available. Run backfill_pipeline.py then daily_aggregation.py first."
        )

    daily_df = daily_df.sort_values("date").reset_index(drop=True)
    latest_row = daily_df.iloc[[-1]]
    current_aqi = float(latest_row["aqi"].iloc[0])
    latest_date = pd.to_datetime(latest_row["date"].iloc[0])

    registry = get_model_registry()

    forecast = {}
    explanations = {}

    for horizon in config.FORECAST_HORIZONS_DAYS:
        horizon_label = f"{horizon}d"
        try:
            model, metadata, scaler = registry.load_model(horizon_label)
        except FileNotFoundError as exc:
            logger.warning(str(exc))
            continue

        feature_cols = metadata["feature_columns"]
        missing = [c for c in feature_cols if c not in latest_row.columns or pd.isna(latest_row[c].iloc[0])]
        if missing:
            logger.warning("Skipping %s: missing required features %s", horizon_label, missing)
            continue
        row_df = latest_row[feature_cols].reset_index(drop=True)
        pred = _predict_one(model, metadata, scaler, row_df)
        forecast[horizon_label] = round(pred, 1)

        try:
            # Remove rows with missing lag or rolling values before SHAP.
            background = daily_df.iloc[:-1][feature_cols].dropna()
            shap_values, base_value = explain_prediction(
                model, metadata["model_type"], background, row_df
            )
            explanations[horizon_label] = {
                "base_value": float(base_value),
                "top_contributions": [
                    {"feature": f, "impact": float(v)}
                    for f, v in top_feature_contributions(feature_cols, shap_values)
                ],
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("SHAP explanation failed for horizon %s (%s)", horizon_label, exc)

    alert = check_alert(current_aqi, {k: v for k, v in forecast.items()})

    result = {
        "city": config.CITY_NAME,
        "as_of_date": latest_date.date().isoformat(),
        "current_aqi": current_aqi,
        "current_category": aqi_category(current_aqi),
        "forecast": forecast,
        "forecast_categories": {k: aqi_category(v) for k, v in forecast.items()},
        "explanations": explanations,
        "alert": alert,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Persist the prediction row for historical tracking / EDA (predicted vs actual)
    pred_row = pd.DataFrame(
        [
            {
                "date": latest_date,
                "generated_at": datetime.now(timezone.utc),
                "current_aqi": current_aqi,
                **{f"pred_{k}": v for k, v in forecast.items()},
            }
        ]
    )
    try:
        store.insert(config.PREDICTIONS_FG, pred_row, primary_key=["date"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not persist prediction row (%s) — continuing anyway.", exc)

    return result


if __name__ == "__main__":
    setup_logging()
    import json

    print(json.dumps(run(), indent=2, default=str))
