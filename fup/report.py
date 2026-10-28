"""Build the human-readable prediction review table (rows shared by the spreadsheet and HTML views)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import features as Fe
from .metrics import match_flags


def _fmt_slot(ts: pd.Timestamp) -> str:
    return f"{C.DAYS[ts.weekday()]} {ts:%d-%b} / {ts:%H:%M}"


def past_bookings_text(df: pd.DataFrame, s: Fe.State, k: int = 3) -> list[str]:
    """Last ``k`` bookings visible at each row's anchor time: facility / use day / use time / booked at."""
    by_res = {rid: g for rid, g in df.groupby("resident_id")}
    out = []
    for rid, t0 in zip(s.meta.resident_id, s.t0):
        g = by_res[rid]
        g = g[Fe.to_min(g.booking_ts) <= t0].tail(k)
        out.append("\n".join(
            f"{C.FACILITY_LABEL[f]} / {C.DAYS[u.weekday()]} {u:%d-%b} / {u:%H:%M} / {C.DAYS[b.weekday()]} {b:%H:%M}"
            for f, u, b in zip(g.facility_id, g.usage_ts, g.booking_ts)))
    return out


def review_table(df: pd.DataFrame, s: Fe.State, p: dict) -> pd.DataFrame:
    m = s.meta
    fl = match_flags(s, p)
    pred_use = Fe.from_min((s.t0_day + p["off"]) * Fe.DAY + p["hour"] * 60)
    nudge = Fe.from_min(p["nudge"])
    act_use = Fe.from_min(m.y_use)
    act_book = Fe.from_min(m.y_book)
    anchor = Fe.from_min(s.t0)
    n_match = sum(v.astype(int) for v in fl.values())
    fac_lbl = np.array([C.FACILITY_LABEL[f] for f in C.FACILITY_IDS])
    return pd.DataFrame({
        "record_ref": m.booking_id,
        "resident": m.resident_id,
        "predicted_at": [f"{C.DAYS[t.weekday()]} {t:%d-%b %H:%M}" for t in anchor],
        "history_n": s.base.n_hist.to_numpy(),
        "past_bookings (facility / day / use time / booked at)": past_bookings_text(df, s),
        "pred_facility": fac_lbl[p["fac"]],
        "pred_day": [f"{C.DAYS[t.weekday()]} {t:%d-%b}" for t in pred_use],
        "pred_use_time": [f"{t:%H:%M}" for t in pred_use],
        "pred_nudge": [f"{C.DAYS[t.weekday()]} {t:%d-%b %H:%M}" for t in nudge],
        "act_facility": fac_lbl[m.y_fac],
        "act_day": [f"{C.DAYS[t.weekday()]} {t:%d-%b}" for t in act_use],
        "act_use_time": [f"{t:%H:%M}" for t in act_use],
        "act_booked_at": [f"{C.DAYS[t.weekday()]} {t:%d-%b %H:%M}" for t in act_book],
        "facility_match": fl["fac"],
        "day_match": fl["day"],
        "hour_match": fl["hour"],
        "nudge_match": fl["nudge"],
        "day_error_days": (s.t0_day + p["off"]) - m.y_use.to_numpy() // Fe.DAY,
        "hour_error_h": p["hour"] - m.y_hour.to_numpy(),
        "nudge_error_min": p["nudge"] - m.y_book.to_numpy(),
        "outputs_matched": n_match,
        "match": np.where(n_match == 4, "YES", "NO"),
        "score": [f"{k} of 4" for k in n_match],
        "confidence_facility": np.round(p["p_fac"], 3) if "p_fac" in p else np.nan,
    })
