#!/usr/bin/env bash
# One-time local setup for the project.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

pip install --upgrade pip
# Install app, dashboard, and test packages for local work.
pip install -r requirements.txt -r dashboard/requirements.txt
pip install pytest>=8.0.0

if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "Created .env from .env.example."
else
  echo ".env already exists."
fi

echo "Done. Activate with: source .venv/bin/activate"
