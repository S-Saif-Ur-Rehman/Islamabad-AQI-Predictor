#!/usr/bin/env bash
# Run the full local pipeline once.
# Usage: ./scripts/run_all_local.sh [backfill_days]
set -euo pipefail
cd "$(dirname "$0")/.."

BACKFILL_DAYS="${1:-730}"

echo "== 1/4: Backfill ${BACKFILL_DAYS} days =="
python -m src.pipelines.backfill_pipeline --days "$BACKFILL_DAYS"

echo "== 2/4: Grab one live reading =="
python -m src.pipelines.feature_pipeline

echo "== 3/4: Build daily features =="
python -m src.pipelines.daily_aggregation

echo "== 4/4: Train models =="
python -m src.pipelines.training_pipeline

echo
echo "Done. Start the dashboard with:"
echo "  streamlit run dashboard/app.py"
