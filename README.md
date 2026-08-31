# Pearls AQI Predictor

See **[REPORT.md](REPORT.md)** for the detailed project report (architecture, methodology,
results, limitations) — that's the submission deliverable; this README is the setup/usage guide.

Serverless, end-to-end AQI forecasting: hourly feature collection, daily
model training, and a Streamlit dashboard that predicts your city's AQI for
the next 3 days — with SHAP explanations and hazard alerts.

## Architecture

```
                 ┌────────────────────┐        ┌──────────────────────┐
   AQICN API ──▶ │  Feature Pipeline  │  ──▶   │   Feature Store       │
OpenWeather API ▶│  (hourly, GH       │        │ (local parquet, or    │
                 │   Actions cron)    │        │  Hopsworks — your     │
                 └────────────────────┘        │  choice via .env)     │
                                                └──────────┬───────────┘
                                                            │
                                                 ┌──────────▼───────────┐
                                                 │ Daily Aggregation     │
                                                 │ (rebuilds daily       │
                                                 │  feature table +      │
                                                 │  1d/2d/3d targets)    │
                                                 └──────────┬───────────┘
                                                            │
                                                 ┌──────────▼───────────┐
                                                 │  Training Pipeline    │
                                                 │  (daily, GH Actions)  │
                                                 │  RF / Ridge / TF MLP  │
                                                 │  → Model Registry     │
                                                 └──────────┬───────────┘
                                                            │
                                                 ┌──────────▼───────────┐
                                                 │  Inference Pipeline   │
                                                 │  + SHAP explanations  │
                                                 └──────────┬───────────┘
                                                            │
                                                 ┌──────────▼───────────┐
                                                 │  Streamlit Dashboard  │
                                                 └───────────────────────┘
```

Every "feature store" and "model registry" call goes through a small
interface (`src/feature_store/base.py`, `src/models/model_registry.py`)
with two implementations each:

| | No account needed (default) | Hopsworks |
|---|---|---|
| Feature store | `data/local_store/*.parquet` | Hopsworks Feature Groups |
| Model registry | `models_registry/*` (joblib/keras + JSON metadata) | Hopsworks Model Registry |

Flip `USE_HOPSWORKS=true` in `.env` (and fill in `HOPSWORKS_API_KEY` /
`HOPSWORKS_PROJECT_NAME`) whenever you're ready — no other code changes
needed. Get a free account at https://app.hopsworks.ai.

## Data sources

- **AQICN** (https://aqicn.org) — live, ground-truth AQI reading for your
  city (US EPA 0-500 scale) plus pollutant sub-indices. Free token:
  https://aqicn.org/data-platform/token/
- **OpenWeather** (https://openweathermap.org) — current + forecast
  weather, current + *historical* pollutant concentrations. Free key:
  https://home.openweathermap.org/api_keys
- **Open-Meteo** (https://open-meteo.com) — free historical weather
  archive, **no API key required**. Used only during backfill, to fill the
  gap OpenWeather's free tier leaves for historical weather.

## Why the backfilled AQI is an *estimate*

AQICN's free tier has no bulk-historical endpoint, so days before you
started running the hourly pipeline don't have an exact AQICN reading.
`backfill_pipeline.py` reconstructs an approximate AQI from OpenWeather's
historical PM2.5/PM10 concentrations using the standard EPA breakpoint
formula (`src/features/aqi_calculator.py`). This is a genuine EPA
methodology, but it only considers PM2.5/PM10 (see that file's docstring
for why). Live rows collected by the hourly pipeline use AQICN's real,
full AQI directly and are strictly more trustworthy — the backfill just
gets you enough training history to not have to wait weeks before your
first model.

## Getting started

```bash
git clone <your-fork-url>
cd aqi-predictor
./scripts/setup_local_dev.sh        # creates .venv, installs deps, copies .env.example -> .env
source .venv/bin/activate
```

Edit `.env`:
- `CITY_NAME` / `LATITUDE` / `LONGITUDE` / `AQICN_CITY_SLUG` already default to
  **Islamabad, Pakistan** (AQICN station: `pakistan/islamabad/us-embassy`).
  Change these if you want a different city.
- Add your `AQICN_API_TOKEN` and `OPENWEATHER_API_KEY`.
- Leave `USE_HOPSWORKS=false` for now — come back to this later.

Run everything once, end to end:

```bash
./scripts/run_all_local.sh 730      # backfill 2 years, aggregate, train
streamlit run dashboard/app.py
```

Or run each stage yourself:

```bash
python -m src.pipelines.backfill_pipeline --days 730
python -m src.pipelines.feature_pipeline        # one live reading
python -m src.pipelines.daily_aggregation
python -m src.pipelines.training_pipeline
python -m src.pipelines.inference_pipeline      # prints the forecast as JSON
```

## Tests

```bash
pytest tests/ -v
```

Covers the EPA AQI calculator and the feature engineering functions
(pure functions, no API calls needed — safe to run anywhere, anytime).

## Flask API

Alongside the Streamlit dashboard, a small Flask API exposes the same forecast as JSON:

```bash
python -m api.app
curl http://localhost:5000/health
curl http://localhost:5000/forecast
```

## Filling in the report's results table

After `training_pipeline.py` has run at least once:

```bash
python -m scripts.generate_report_metrics
```

Paste its output into the Results section of `REPORT.md`.

## Automating with GitHub Actions

1. Push this repo to GitHub.
2. Repo **Settings → Secrets and variables → Actions**:
   - **Secrets**: `AQICN_API_TOKEN`, `OPENWEATHER_API_KEY`, and (once you
     have them) `HOPSWORKS_API_KEY`.
   - **Variables**: `CITY_NAME`, `LATITUDE`, `LONGITUDE`, `AQICN_CITY_SLUG`,
     `USE_HOPSWORKS` (`true`/`false`), `HOPSWORKS_PROJECT_NAME`.
3. That's it — `.github/workflows/feature_pipeline.yml` runs hourly and
   `.github/workflows/training_pipeline.yml` runs daily. Both can also be
   triggered manually from the **Actions** tab (`workflow_dispatch`).

While `USE_HOPSWORKS=false`, both workflows commit the updated
`data/local_store/` / `models_registry/` files back into the repo after
each run — that's how state survives between otherwise-stateless Actions
runs. Once Hopsworks is enabled, storage lives there instead and those
commit-back steps simply become no-ops.

## Deploying the dashboard

Push to GitHub, then deploy for free on
[Streamlit Community Cloud](https://streamlit.io/cloud): point it at
`dashboard/app.py` and paste the same `.env` values into the app's
**Secrets** panel (Streamlit reads `st.secrets`, which `python-dotenv`
falls back to automatically once you also set them as real environment
variables in the deploy config — or simply add
`os.environ.update(st.secrets)` at the top of `dashboard/app.py` if you'd
rather not duplicate values).

## Project structure

```
REPORT.md                          # detailed project report (submission deliverable)
config/config.py                   # all settings, read from env vars
src/clients/                       # AQICN, OpenWeather, Open-Meteo API clients
src/features/                      # feature engineering + EPA AQI calculator
src/feature_store/                 # local + Hopsworks feature store backends
src/models/                        # model registry, SHAP explainability, persistence baseline
src/pipelines/
  feature_pipeline.py               # hourly: fetch + write one raw row
  backfill_pipeline.py              # historical backfill (argparse: --days)
  daily_aggregation.py              # raw hourly -> daily features + targets
  training_pipeline.py              # baseline / Ridge / RF / TF MLP per horizon, picks best
  inference_pipeline.py             # latest features -> 3-day forecast + SHAP
dashboard/app.py                   # Streamlit UI
api/app.py                         # Flask REST API (/health, /forecast)
.github/workflows/                 # hourly + daily automation
tests/                             # unit tests for pure-function modules
scripts/                           # local dev convenience scripts + report metrics generator
```

## Extending this

- **More models**: add another branch in `train_horizon()` in
  `training_pipeline.py` — it already picks whichever model scores lowest
  RMSE, so a new candidate just needs to return `{"model", "metrics",
  "is_keras"}` in the same `results` dict.
- **Alerts via email/Slack**: `src/utils/alerts.py` already computes
  whether a hazard threshold is breached — wire its output into a
  notification call at the end of `inference_pipeline.run()`.
- **Flask/FastAPI API layer**: if you want predictions served over HTTP
  (rather than only inside the Streamlit process), wrap
  `inference_pipeline.run()` in a single Flask route — `flask` is already
  in `requirements.txt` for this.
