# Facility Usage Prediction System

This prototype predicts four things about a resident's **next** booking of a shared community
facility, such as the gym, pool, courts or clubhouse:

1. **Facility**
2. **Usage day**
3. **Usage hour**
4. **When to notify them**

It covers synthetic data generation, leakage-safe features, a chained gradient-boosting model,
evaluation on a chronologically later holdout, and a spreadsheet and HTML table comparing predictions with actuals.

**Holdout results** (Nov–Dec 2025, 3,843 bookings, never used for fitting or tuning):

| | Model | Habit baseline |
|---|---|---|
| Facility accuracy (top-3) | 57.8% (86.4%) | 57.9% (76.9%) |
| Usage day – exact date | 39.2% | 23.7% |
| Usage hour – exact hour (±1 h) | 47.4% (71.9%) | 46.8% (71.7%) |
| Nudge within ±60 min of actual booking | 35.0% | 18.9% |
| **Mean output accuracy (avg "x of 4" / 4)** | **44.8%** | 36.8% |
| All 4 outputs correct | 13.5% | 8.0% |

## Deliverables

| Deliverable | Location |
|---|---|
| 01 Dataset | `data/bookings.csv` (+ `facilities.csv`, `residents.csv`) |
| 01 Runnable prototype | `fup/` package, `run_all.sh` |
| Model artifact | `models/chain_model.joblib`, `models/training_metadata.json` |
| 02 Technical documentation | [`docs/TECHNICAL_DOCUMENTATION.md`](docs/TECHNICAL_DOCUMENTATION.md) |
| 03 Prediction review (spreadsheet) | `reports/prediction_review.xlsx`. Sheets: Summary · Predictions · Error analysis · Facility confusion |
| 03 Prediction review (UI table) | `reports/prediction_review.html`. Open in any browser; works offline, with search and filter. |
| Flat predictions / metrics | `reports/test_predictions.csv`, `reports/metrics.json` |
| Live next-booking predictions | `reports/next_booking_predictions.csv` |

## Quick start

Requires Python ≥ 3.11 and no internet once the dependencies are installed.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./run_all.sh
```

Or run the steps one by one:

```bash
export PYTHONPATH=.
.venv/bin/python -m fup.generate_data   # 1 s    regenerate data/ (deterministic, seed 42)
.venv/bin/python -m fup.train           # ~4 min build features, tune on Sep–Oct, refit, save models/
.venv/bin/python -m fup.evaluate        # ~10 s  score Nov–Dec holdout, write reports/
.venv/bin/python -m fup.predict         # ~5 s   next booking + nudge time for every resident
.venv/bin/python -m pytest -q           # ~8 s   leakage / split / data / metric tests
```

Optional flags:

* `python -m fup.train --no-tune` skips the grid search and finishes in about 1.5 min.
* `python -m fup.predict --resident R-104 --as-of "2025-11-15 12:00"` makes a point-in-time prediction
  for one resident, using only bookings created up to that moment.

The data, model and reports are included. A reviewer can open
`reports/prediction_review.xlsx` or `.html` straight away, or re-run everything with `./run_all.sh`,
which reproduces them exactly.

## Drive upload and GitHub

Keep the source code, `requirements.txt`, data, models, reports and documentation together.
The upload copy excludes `.venv/`, Python and pytest caches, macOS metadata and local Git
history. After downloading and extracting it, follow the quick start above to recreate the
environment. `./run_all.sh` can also create the environment and install dependencies automatically;
the initial dependency installation requires internet access.

The `.gitignore` excludes local environments, caches and `.env` secrets from future Git commits.
To publish the extracted upload copy to GitHub later, run:

```bash
git init -b main
git add .
git commit -m "Initial import"
git remote add origin <your-github-repository-url>
git push -u origin main
```

## How it works

```
bookings.csv ──► point-in-time features at t0 (the moment the previous booking was created)
                 │   only rows with booking_ts ≤ t0 are visible
                 ▼
     facility model ──► day-offset model ──► hour model ──► booking-time regressor
      P(f|hist)          P(d|hist,f)          P(h|hist,f,d)    median booking minute → nudge
```

* **Split**: train = targets booked before 1 Sep 2025 · validation (tuning only) = Sep–Oct ·
  **test = Nov–Dec 2025**. After tuning, the model is refit on train + validation.
* **Match rules**: facility exact · day = exact calendar date · hour = exact start hour ·
  nudge = within ±60 min of when the resident actually booked.

See the technical documentation for features, leakage controls, the alternatives considered,
error analysis and limitations.

## Repository layout

```
fup/
  config.py         paths, seed, split dates, facility catalogue
  generate_data.py  synthetic community simulator
  features.py       leakage-safe point-in-time feature builder
  model.py          chained model + baselines
  split.py          chronological split
  train.py          tuning on validation, refit, save artifact
  metrics.py        per-output and overall metrics
  evaluate.py       holdout evaluation, Excel review, error analysis
  report.py         predicted-vs-actual review table
  html_report.py    offline HTML review UI
  predict.py        live next-booking + nudge prediction
tests/test_pipeline.py
data/  models/  reports/  docs/
```
