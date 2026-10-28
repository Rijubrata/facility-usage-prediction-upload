"""Production-style inference: predict each resident's NEXT booking and when to nudge them.

Uses the saved model and all bookings visible at ``--as-of`` (default: end of the dataset).

Run:  python -m fup.predict [--as-of "2025-12-31 23:59"] [--resident R-104] [--out reports/next_booking_predictions.csv]
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import features as Fe


def predict_next(df: pd.DataFrame, model, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    if as_of is not None:
        df = df[df.booking_ts <= as_of]  # nothing created after the cut-off is visible
    s = Fe.build_state(df, live=True)
    p = model.predict(s)
    now = as_of if as_of is not None else df.booking_ts.max()
    now_m = int(Fe.to_min(pd.Series([now]))[0])
    use_m = (s.t0_day + p["off"]) * Fe.DAY + p["hour"] * 60
    # The model predicts relative to the resident's last booking. If that predicted slot is already
    # in the past (resident inactive since), roll it forward by whole weeks -- keeps weekday + hour.
    week = 7 * Fe.DAY
    roll = np.where(use_m <= now_m + 30, np.ceil((now_m + 30 - use_m) / week).astype(int), 0) * week
    use = Fe.from_min(use_m + roll)
    nudge_m = p["nudge"] + roll
    # a nudge that is already due is sent right away (still before the predicted usage)
    nudge = Fe.from_min(np.maximum(nudge_m, now_m + 1))
    days_idle = (now_m - s.t0) / Fe.DAY
    return pd.DataFrame({
        "resident_id": s.meta.resident_id,
        "last_booking_id": s.meta.anchor_id,
        "history_n": s.base.n_hist.to_numpy(),
        "pred_facility": [C.FACILITY_LABEL[C.FACILITY_IDS[f]] for f in p["fac"]],
        "pred_facility_id": [C.FACILITY_IDS[f] for f in p["fac"]],
        "pred_usage_day": [f"{C.DAYS[t.weekday()]} {t:%Y-%m-%d}" for t in use],
        "pred_usage_time": [f"{t:%H:%M}" for t in use],
        "notify_at": nudge.strftime("%Y-%m-%d %H:%M"),
        "confidence_facility": np.round(p["p_fac"], 3),
        "confidence_day": np.round(p["p_day"], 3),
        "confidence_hour": np.round(p["p_hour"], 3),
        "days_since_last_booking": np.round(days_idle, 1),
        "status": np.where(days_idle > 30, "inactive >30d (low reliability)",
                           np.where(roll > 0, "rolled forward", "ok")),
    }).sort_values("notify_at").reset_index(drop=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", type=pd.Timestamp, default=None)
    ap.add_argument("--resident", default=None)
    ap.add_argument("--out", default=str(C.REPORT_DIR / "next_booking_predictions.csv"))
    a = ap.parse_args(argv)
    model = joblib.load(C.MODEL_DIR / "chain_model.joblib")
    df = Fe.load_bookings()
    out = predict_next(df, model, a.as_of)
    if a.resident:  # filter after feature building so community-level features stay correct
        out = out[out.resident_id == a.resident]
        if out.empty:
            raise SystemExit(f"unknown resident {a.resident}")
    C.REPORT_DIR.mkdir(exist_ok=True)
    out.to_csv(a.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.head(20).to_string(index=False))
    print(f"\n{len(out)} residents -> {a.out}")


if __name__ == "__main__":
    main()
