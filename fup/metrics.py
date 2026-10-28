"""Metrics for the four outputs plus overall measures."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, log_loss

from . import config as C
from . import features as Fe


def match_flags(s: Fe.State, p: dict) -> dict:
    m = s.meta
    pred_uday = s.t0_day + p["off"]
    true_uday = m.y_use.to_numpy() // Fe.DAY
    return dict(
        fac=p["fac"] == m.y_fac.to_numpy(),
        day=pred_uday == true_uday,
        hour=p["hour"] == m.y_hour.to_numpy(),
        nudge=np.abs(p["nudge"] - m.y_book.to_numpy()) <= C.NUDGE_TOLERANCE_MIN,
    )


def evaluate(s: Fe.State, p: dict) -> dict:
    m = s.meta
    fl = match_flags(s, p)
    y_f = m.y_fac.to_numpy()
    true_uday = m.y_use.to_numpy() // Fe.DAY
    day_err = (s.t0_day + p["off"]) - true_uday
    hour_err = p["hour"] - m.y_hour.to_numpy()
    nudge_err = p["nudge"] - m.y_book.to_numpy()
    usage_pred = (s.t0_day + p["off"]) * Fe.DAY + p["hour"] * 60
    n_match = sum(fl.values()).astype(int)
    pf = np.clip(p["proba_fac"], 1e-6, 1)
    pf = pf / pf.sum(1, keepdims=True)
    top3 = (np.argsort(-pf, 1)[:, :3] == y_f[:, None]).any(1)
    return {
        "n": len(m),
        # facility
        "facility_acc": fl["fac"].mean(),
        "facility_top3": top3.mean(),
        "facility_macro_f1": f1_score(y_f, p["fac"], average="macro", labels=range(C.N_FAC), zero_division=0),
        "facility_logloss": log_loss(y_f, pf, labels=range(C.N_FAC)),
        # usage day (calendar date of use)
        "day_acc": fl["day"].mean(),
        "day_within1": (np.abs(day_err) <= 1).mean(),
        "day_mae": np.abs(day_err).mean(),
        # usage hour (hour of day)
        "hour_acc": fl["hour"].mean(),
        "hour_within1": (np.abs(hour_err) <= 1).mean(),
        "hour_mae": np.abs(hour_err).mean(),
        # notification time vs actual booking creation time
        "nudge_within60": fl["nudge"].mean(),
        "nudge_within180": (np.abs(nudge_err) <= 180).mean(),
        "nudge_mae_h": np.abs(nudge_err).mean() / 60,
        "nudge_median_ae_h": np.median(np.abs(nudge_err)) / 60,
        "nudge_before_booking": (nudge_err <= 0).mean(),  # a nudge is only useful if it precedes the booking
        # overall
        "mean_outputs_matched": n_match.mean(),
        "mean_output_accuracy": n_match.mean() / 4,
        "all4_match": (n_match == 4).mean(),
        "fac_day_hour_match": (fl["fac"] & fl["day"] & fl["hour"]).mean(),
        "usage_slot_within2h": (fl["fac"] & (np.abs(usage_pred - m.y_use.to_numpy()) <= 120)).mean(),
    }


# single number used for tuning on the validation set
def overall_score(metrics: dict) -> float:
    return metrics["mean_output_accuracy"]
