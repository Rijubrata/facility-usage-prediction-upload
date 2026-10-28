#!/usr/bin/env bash
# End-to-end reproduction: data -> train (tune on val) -> evaluate on holdout -> live predictions -> tests.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
export PYTHONPATH=.
.venv/bin/python -m fup.generate_data   # ~1 s   -> data/
.venv/bin/python -m fup.train           # ~4 min -> models/
.venv/bin/python -m fup.evaluate        # ~10 s  -> reports/
.venv/bin/python -m fup.predict         # ~5 s   -> reports/next_booking_predictions.csv
.venv/bin/python -m pytest -q
