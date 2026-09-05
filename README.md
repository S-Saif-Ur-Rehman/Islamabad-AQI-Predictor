# Pearls AQI Predictor

Serverless, end-to-end AQI forecasting: hourly feature collection, daily model training, and a
Streamlit dashboard that predicts your city's AQI for the next 3 days, with SHAP explanations
and hazard alerts.

**Live app:** https://islamabad-aqipredictor.streamlit.app/, currently running the single-tier
`streamlit-only-deployment` branch described below. The two-tier `master` branch, with a Flask
API on Render, is built and tested but not deployed yet.

See **[REPORT.md](REPORT.md)** for the detailed project report (architecture, methodology,
results, limitations). That's the submission deliverable; this README is the setup/usage guide.

This README is identical on every branch of this repo. It describes both deployment
architectures the project ships; see [Two deployment branches](#two-deployment-branches) for
which one you're looking at right now and how the two relate.

## Contents

- [Architecture](#architecture)
- [Two deployment branches](#two-deployment-branches)
- [Data sources](#data-sources)
- [Why the backfilled AQI is an estimate](#why-the-backfilled-aqi-is-an-estimate)
- [Getting started](#getting-started)
- [Tests](#tests)
- [Filling in the report's results table](#filling-in-the-reports-results-table)
- [Automating with GitHub Actions](#automating-with-github-actions)
- [Deploying this](#deploying-this)
- [Project structure](#project-structure)
- [Extending this](#extending-this)

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
                                                 │  Baseline / Ridge /   │
                                                 │  RF / TF MLP → via    │
                                                 │  Feature View →       │
                                                 │  Model Registry       │
                                                 └──────────┬───────────┘
                                                            │
                                                 ┌──────────▼───────────┐
                                                 │  Inference Pipeline   │
                                                 │  + SHAP explanations  │
                                                 └──────────┬───────────┘
                                                            │
                                                 ┌──────────▼───────────┐
                                                 │  Streamlit Dashboard  │
                                                 │  (Flask API too, on   │
                                                 │  the two-tier branch) │
                                                 └───────────────────────┘
```

Everything from the feature pipeline through SHAP explanations is identical on every branch. The
only thing that differs is the last box: how the web layer is deployed. That's covered in full
below.

Every "feature store" and "model registry" call goes through a small interface
(`src/feature_store/base.py`, `src/models/model_registry.py`) with two implementations each:

| | No account needed (default) | Hopsworks |
|---|---|---|
| Feature store | `data/local_store/*.parquet` | Hopsworks Feature Groups + a Feature View for training |
| Model registry | `models_registry/*` (joblib/keras + JSON metadata) | Hopsworks Model Registry |

This project actually runs against Hopsworks. `USE_HOPSWORKS=true` in the real `.env`, feature
groups populated with two years of Islamabad data, models trained and registered there. The
local backend still works fine on its own if you'd rather not set up an account before poking
around: flip `USE_HOPSWORKS=false` and everything runs against parquet files on disk instead, no
code changes needed either way. Hopsworks itself is free at https://app.hopsworks.ai.

## Two deployment branches

This repo ships two working deployment architectures, each on its own branch, rather than
picking just one:

| Branch | Architecture | Web layer |
|---|---|---|
| **`master`** | Two-tier | `api/app.py` (Flask) serves the forecast as JSON; `dashboard/app.py` is a thin client that calls it over HTTP. Built to deploy via `render.yaml` (Render) + Streamlit Community Cloud; the Render side isn't live yet. |
| **`streamlit-only-deployment`** | Single-tier | `dashboard/app.py` runs inference in-process: reads Hopsworks directly, loads the model, computes SHAP. No `api/app.py`, no `render.yaml`, nothing else to host. |

#### Why this exists at all

The project brief names Flask as a required technology alongside Streamlit. The two-tier branch
(`master`) satisfies that literally: Flask runs as its own hosted service, not just imported and
left idle. The single-tier branch exists because a separately-hosted API is one more moving part
than a forecasting dashboard strictly needs, and because it's a fair question whether a grader or
a future maintainer would rather see the leaner version. Building both, on two branches, means
neither answer had to be picked in advance.

#### Why two branches instead of one config toggle

Early on, `dashboard/app.py` supported both modes in a single file, switching on whether an
`API_URL` secret was set. That worked, but every checkout carried Flask, `render.yaml`, and the
HTTP-client code path regardless of which way it was actually being deployed. That's dead weight
on whichever branch never uses it, and a runtime flag standing in for what's really an
architectural decision. Splitting them into two real branches makes each one a complete, minimal
reflection of the architecture it deploys: everything in `master` is either running or directly
deployment config for the two-tier setup, and the same is true of `streamlit-only-deployment`
for the single-tier one.

**How to treat them:**
- `master` is the default branch, and the one this README and `REPORT.md` primarily describe.
- Check out `streamlit-only-deployment` for the simpler, single-service alternative: same
  pipelines, same models, same dashboard UI, just without a separately-hosted API.
- Everything upstream of the web layer (feature pipeline, backfill, daily aggregation, training,
  inference, SHAP, GitHub Actions automation) is identical on both branches. Changes there should
  land on one branch and get merged or cherry-picked into the other, so the two don't drift apart
  on anything but the web layer.
- `git branch --show-current` shows which one you have checked out.
  `git diff master streamlit-only-deployment` shows exactly what differs in practice:
  `api/app.py`, `render.yaml`, `tests/test_api.py`, a handful of lines in `dashboard/app.py`
  (the `API_URL` branch), and the Flask/`requests` lines in the two `requirements*.txt` files.

## Data sources

- **AQICN** (https://aqicn.org): live, ground-truth AQI reading for your city (US EPA 0-500
  scale) plus pollutant sub-indices. Free token: https://aqicn.org/data-platform/token/
- **OpenWeather** (https://openweathermap.org): current + forecast weather, current +
  *historical* pollutant concentrations. Free key: https://home.openweathermap.org/api_keys
- **Open-Meteo** (https://open-meteo.com): free historical weather archive, no API key required.
  Used only during backfill, to fill the gap OpenWeather's free tier leaves for historical
  weather.

## Why the backfilled AQI is an *estimate*

AQICN's free tier has no bulk-historical endpoint, so days before you started running the hourly
pipeline don't have an exact AQICN reading. `backfill_pipeline.py` reconstructs an approximate
AQI from OpenWeather's historical PM2.5/PM10 concentrations using the standard EPA breakpoint
formula (`src/features/aqi_calculator.py`). This is a genuine EPA methodology, but it only
considers PM2.5/PM10 (see that file's docstring for why). Live rows collected by the hourly
pipeline use AQICN's real, full AQI directly and are strictly more trustworthy. The backfill just
gets you enough training history so you don't have to wait weeks before your first model.

## Getting started

```bash
git clone <your-fork-url>
cd aqi-predictor
./scripts/setup_local_dev.sh        # creates .venv, installs deps, copies .env.example -> .env
source .venv/bin/activate
```

Works the same on either branch. `setup_local_dev.sh` installs both `requirements.txt` and
`dashboard/requirements.txt` plus pytest. The project's dependencies are split by deployment
target (see below), so full local dev needs both files together.

Edit `.env`:
- `CITY_NAME` / `LATITUDE` / `LONGITUDE` / `AQICN_CITY_SLUG` already default to
  **Islamabad, Pakistan** (AQICN station: `pakistan/islamabad/us-embassy`).
  Change these if you want a different city.
- Add your `AQICN_API_TOKEN` and `OPENWEATHER_API_KEY`.
- Leave `USE_HOPSWORKS=false` if you're just trying things out locally.

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

On `master`, you can also run the Flask API locally:

```bash
python -m api.app
curl http://localhost:5000/health
curl http://localhost:5000/forecast
curl http://localhost:5000/history
```

Set `API_AUTH_TOKEN` in `.env` to require an `X-API-Key` header on `/forecast` and `/history`.
Leave it unset and the API stays open, which is fine for local use but worth turning on before
exposing this anywhere public. `streamlit-only-deployment` doesn't have `api/app.py` at all; the
dashboard calls `inference_pipeline.run()` directly instead.

## Tests

```bash
pytest tests/ -v
```

Covers the EPA AQI calculator and the feature engineering functions on both branches: pure
functions, no live AQICN/OpenWeather calls required. `master` additionally includes
`tests/test_api.py` for the Flask endpoints; `streamlit-only-deployment` doesn't have it, since
there's no API to test there.

## Filling in the report's results table

After `training_pipeline.py` has run at least once:

```bash
python -m scripts.generate_report_metrics
```

Paste its output into the Results section of `REPORT.md`.

## Automating with GitHub Actions

1. Push this repo to GitHub.
2. Repo **Settings → Secrets and variables → Actions**:
   - **Secrets**: `AQICN_API_TOKEN`, `OPENWEATHER_API_KEY`, and `HOPSWORKS_API_KEY`.
   - **Variables**: `CITY_NAME`, `LATITUDE`, `LONGITUDE`, `AQICN_CITY_SLUG`,
     `USE_HOPSWORKS` (`true`/`false`), `HOPSWORKS_PROJECT_NAME`.
3. That's it: `.github/workflows/feature_pipeline.yml` runs hourly and
   `.github/workflows/training_pipeline.yml` runs daily. Both can also be
   triggered manually from the **Actions** tab (`workflow_dispatch`). These workflows are
   identical on both branches; they only touch the pipeline, never the web layer.

While `USE_HOPSWORKS=false`, both workflows commit the updated
`data/local_store/` / `models_registry/` files back into the repo after
each run, since GitHub Actions runners start from a clean checkout every time and would
otherwise "forget" all prior runs. With Hopsworks enabled (the actual setup here), storage lives
there instead and those commit-back steps are no-ops.

## Deploying this

### Two-tier (`master`): Flask on Render + Streamlit dashboard

1. Push `master` to GitHub.
2. On [Render](https://render.com): **New → Blueprint**, point it at this repo on the `master`
   branch. It reads `render.yaml` and builds the Flask API as a web service
   (`gunicorn api.app:app`). Fill in the secrets marked `sync: false`
   (`HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT_NAME`, `AQICN_API_TOKEN`, `OPENWEATHER_API_KEY`,
   `API_AUTH_TOKEN`) in Render's Environment tab after the first deploy; they're never read
   from `render.yaml` itself.
3. On [Streamlit Community Cloud](https://streamlit.io/cloud): new app pointed at
   `dashboard/app.py` on `master`. Paste the values from `.streamlit/secrets.toml.example` into
   its Secrets panel, including `API_URL` set to the Render service's URL and a matching
   `API_AUTH_TOKEN`.

Render is a natural host for the Flask side: an always-on process with no execution-time limit,
which matches `gunicorn` and `/forecast`'s ~20-second live SHAP computation cleanly. Some
account setups on Render's free tier ask for a card before a web service will deploy. That's the
tradeoff this branch takes on in exchange for demonstrating Flask as a real, separately-hosted
service rather than just imported code.

### Single-tier (`streamlit-only-deployment`): Streamlit only

1. Push the `streamlit-only-deployment` branch to GitHub.
2. On Streamlit Community Cloud: new app pointed at `dashboard/app.py` on
   `streamlit-only-deployment`. Paste the values from `.streamlit/secrets.toml.example`; no
   `API_URL` needed, since there's nothing to call.
3. `dashboard/requirements.txt` (co-located with the entrypoint, so Streamlit Cloud picks it up
   automatically instead of the root `requirements.txt`) has everything this branch's dashboard
   needs: no Flask, no `requests`, no gunicorn.

This branch sidesteps the Render tradeoff entirely, at the cost of not having Flask running
anywhere as its own service.

## Project structure

```
REPORT.md                          # detailed project report (submission deliverable)
config/config.py                   # all settings, read from env vars
src/clients/                       # AQICN, OpenWeather, Open-Meteo API clients
src/features/                      # feature engineering + EPA AQI calculator
src/feature_store/                 # local + Hopsworks feature store backends (Hopsworks incl. Feature View)
src/models/                        # model registry, SHAP explainability, persistence baseline
src/utils/                         # alerts, logging, error sanitization
src/pipelines/
  feature_pipeline.py               # hourly: fetch + write one raw row
  backfill_pipeline.py              # historical backfill (argparse: --days)
  daily_aggregation.py              # raw hourly -> daily features + targets
  training_pipeline.py              # baseline / Ridge / RF / TF MLP per horizon, picks best
  inference_pipeline.py             # latest features -> 3-day forecast + SHAP
dashboard/app.py                   # Streamlit UI
dashboard/requirements.txt         # dashboard-only deps, picked up by Streamlit Cloud automatically
.streamlit/secrets.toml.example    # secrets template, content differs per branch (see above)
.github/workflows/                 # hourly + daily automation (identical on both branches)
tests/                             # unit tests
scripts/                           # local dev convenience scripts + report metrics generator
```

`master` only, on top of the above:

```
api/app.py                         # Flask REST API (/health, /forecast, /history)
render.yaml                        # Render Blueprint for the Flask API
tests/test_api.py                  # Flask endpoint tests
```

## Extending this

- **More models**: add another branch in `train_horizon()` in
  `training_pipeline.py`. It already picks whichever model scores lowest
  RMSE, so a new candidate just needs to return `{"model", "metrics",
  "is_keras"}` in the same `results` dict.
- **Alerts via email/Slack**: `src/utils/alerts.py` already computes
  whether a hazard threshold is breached; wire its output into a
  notification call at the end of `inference_pipeline.run()`.
- **Faster `/forecast` on `master`**: precompute SHAP once during the daily training run and
  store it alongside the model, instead of computing it live on every request. `/forecast`
  currently takes ~20 seconds because of live SHAP computation. Fine on Render (no
  execution-time limit), but it would rule out stricter-timeout serverless hosts.
