"""Train the AQI forecast models for each horizon."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from config import config
from src.features.feature_engineering import select_available_features
from src.feature_store.base import get_feature_store
from src.models.baseline import PersistenceBaseline
from src.models.model_registry import get_model_registry
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

MIN_ROWS_TO_TRAIN = 20  # below this, splits get too noisy to trust
TEST_FRACTION = 0.2


def _time_ordered_split(df: pd.DataFrame, test_fraction: float):
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def _evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def _build_mlp(input_dim: int):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_horizon(daily_df: pd.DataFrame, horizon_days: int) -> dict:
    horizon_label = f"{horizon_days}d"
    target_col = f"aqi_target_{horizon_days}d"
    feature_cols = select_available_features(daily_df)

    data = daily_df.dropna(subset=[target_col, *feature_cols]).reset_index(drop=True)
    if len(data) < MIN_ROWS_TO_TRAIN:
        logger.warning(
            "Only %d usable rows for horizon %s (need >= %d). Skipping — run the backfill "
            "pipeline for more history, or wait for more daily data to accumulate.",
            len(data), horizon_label, MIN_ROWS_TO_TRAIN,
        )
        return {}

    train_df, test_df = _time_ordered_split(data, TEST_FRACTION)
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    results = {}

    # Baseline: today's AQI is the next-day guess.
    persistence = PersistenceBaseline().fit(X_train, y_train)
    results["persistence_baseline"] = {
        "model": persistence,
        "metrics": _evaluate(y_test, persistence.predict(X_test)),
        "is_keras": False,
    }

    # Random forest
    rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["random_forest"] = {"model": rf, "metrics": _evaluate(y_test, rf.predict(X_test)), "is_keras": False}

    # Ridge regression with scaling
    from sklearn.pipeline import make_pipeline

    ridge_pipeline = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X_train, y_train)
    results["ridge"] = {
        "model": ridge_pipeline,
        "metrics": _evaluate(y_test, ridge_pipeline.predict(X_test)),
        "is_keras": False,
    }

    # TensorFlow MLP
    try:
        mlp_scaler = StandardScaler().fit(X_train)
        mlp = _build_mlp(input_dim=len(feature_cols))
        mlp.fit(
            mlp_scaler.transform(X_train), y_train,
            validation_split=0.1, epochs=100, batch_size=8, verbose=0,
            callbacks=[__import__("tensorflow").keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        )
        y_pred_mlp = mlp.predict(mlp_scaler.transform(X_test), verbose=0).flatten()
        # Save the scaler separately so it can be restored at inference time.
        results["tensorflow_mlp"] = {
            "model": mlp, "metrics": _evaluate(y_test, y_pred_mlp), "is_keras": True, "scaler": mlp_scaler,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("TensorFlow MLP training failed for horizon %s (%s) — skipping.", horizon_label, exc)

    # Pick the best model by RMSE
    best_name = min(results, key=lambda k: results[k]["metrics"]["rmse"])
    best = results[best_name]
    logger.info(
        "Horizon %s — best model: %s | RMSE=%.2f MAE=%.2f R2=%.3f",
        horizon_label, best_name, best["metrics"]["rmse"], best["metrics"]["mae"], best["metrics"]["r2"],
    )

    registry = get_model_registry()
    registry.save_model(
        horizon_label=horizon_label,
        model=best["model"],
        model_type=best_name,
        metrics=best["metrics"],
        feature_columns=feature_cols,
        is_keras=best.get("is_keras", False),
        scaler=best.get("scaler"),
    )

    return {
        "horizon": horizon_label,
        "best_model": best_name,
        "metrics": {name: r["metrics"] for name, r in results.items()},
    }


def run():
    store = get_feature_store()
    daily_df = store.get_training_data(config.DAILY_FEATURES_FG)
    if daily_df.empty:
        logger.warning("Daily feature table is empty — run daily_aggregation.py first.")
        return []

    daily_df = daily_df.sort_values("date").reset_index(drop=True)
    all_results = []
    for horizon in config.FORECAST_HORIZONS_DAYS:
        result = train_horizon(daily_df, horizon)
        if result:
            all_results.append(result)

    # Save all model scores so the dashboard can compare them.
    if all_results:
        config.LOCAL_MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        comparison_path = config.LOCAL_MODEL_REGISTRY_DIR / "comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(
                {"city": config.CITY_NAME, "generated_at": datetime.now(timezone.utc).isoformat(), "results": all_results},
                f,
                indent=2,
            )
        logger.info("Wrote model comparison to %s", comparison_path)

    return all_results


if __name__ == "__main__":
    setup_logging()
    results = run()
    for r in results:
        logger.info("Horizon %s comparison:", r["horizon"])
        for name, metrics in r["metrics"].items():
            marker = " <- deployed" if name == r["best_model"] else ""
            logger.info(
                "   %-20s RMSE=%.2f MAE=%.2f R2=%.3f%s",
                name, metrics["rmse"], metrics["mae"], metrics["r2"], marker,
            )
