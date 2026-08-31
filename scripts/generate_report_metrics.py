"""Print the latest model metrics into a simple markdown table."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config


def main():
    rows = []
    comp_path = config.LOCAL_MODEL_REGISTRY_DIR / "comparison.json"
    if comp_path.exists():
        with open(comp_path) as f:
            comp = json.load(f)
        for r in comp.get("results", []):
            horizon = r["horizon"]
            best_model = r["best_model"]
            metrics = r["metrics"][best_model]
            rows.append((horizon, best_model, metrics, comp.get("generated_at", "")))
    else:
        for horizon_days in config.FORECAST_HORIZONS_DAYS:
            meta_path = config.LOCAL_MODEL_REGISTRY_DIR / f"{horizon_days}d" / "metadata.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            rows.append((f"{horizon_days}d", meta["model_type"], meta["metrics"], meta["trained_at"]))

    if not rows:
        print("No trained models or comparison data found yet — run training_pipeline.py first.")
        return

    print(f"City: {config.CITY_NAME}\n")
    print("| Horizon | Deployed model | RMSE | MAE | R² | Trained at (UTC) |")
    print("|---|---|---|---|---|---|")
    for horizon, model_type, metrics, trained_at in rows:
        print(
            f"| {horizon} | {model_type} | {metrics['rmse']:.2f} | "
            f"{metrics['mae']:.2f} | {metrics['r2']:.3f} | {trained_at} |"
        )


if __name__ == "__main__":
    main()
