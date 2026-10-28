"""Leakage-safe feature construction.

Prediction framing
------------------
Every booking ``k`` of a resident (except their first) is a *target*. The
prediction is made at the **anchor time** ``t0`` = the moment the resident's
previous booking ``k-1`` was created ("right after you book, we predict your
next booking and schedule a nudge"). Only bookings with
``booking_ts <= t0`` are visible -- for this resident *and* for the
community-wide popularity features. Bookings created before ``t0`` whose usage
lies in the future are visible (they are already scheduled), which is exactly
what the live system would know.

The four outputs are predicted as a chain of conditional models

    facility  ~ P(f | history)
    usage day ~ P(day offset | history, f)
    hour      ~ P(hour | history, f, day)
    nudge     ~ booking time | history, f, day, hour     (regression)

so the feature builders for the later stages take the conditioning values as
arguments. During training they receive the true values (teacher forcing);
at inference they receive the upstream predictions.

All per-target history summaries are pre-computed once into a ``State`` of
numpy arrays so the conditional features are cheap vectorised look-ups.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

DAY = 1440
HALF_LIFE_DAYS = 30.0
N_OFF = 11            # day-offset classes 0..9 and 10 == "10+"
HOURS = np.arange(5, 23)  # usage start hours that occur in the community (05:00-22:00)
N_H = len(HOURS)
EPOCH = pd.Timestamp("2024-01-01")  # a Monday -> (minutes // DAY) % 7 is Mon=0
TARGET_COLS = ["y_fac", "y_off", "y_off_raw", "y_hour", "y_bkrel", "y_book", "y_use"]


def load_bookings(path=C.BOOKINGS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["booking_ts", "usage_ts"])
    return df.sort_values(["booking_ts", "booking_id"], kind="stable").reset_index(drop=True)


def to_min(ts) -> np.ndarray:
    return ((pd.to_datetime(ts) - EPOCH) // pd.Timedelta(minutes=1)).to_numpy().astype(np.int64)


def from_min(m) -> pd.DatetimeIndex:
    return EPOCH + pd.to_timedelta(np.asarray(m, dtype=np.int64), unit="min")


def _share(counts: np.ndarray) -> np.ndarray:
    tot = counts.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, counts / np.maximum(tot, 1e-9), np.nan)


@dataclass
class State:
    """Per-target snapshot of everything known at anchor time t0 (arrays aligned with ``meta``)."""
    meta: pd.DataFrame
    base: pd.DataFrame          # shared, conditioning-free features
    fac_dow: np.ndarray         # (N, F, 7)   usage-weekday counts per facility
    all_dow_dec: np.ndarray     # (N, 7)      decayed usage-weekday weights
    fac_hour: np.ndarray        # (N, F, 24)  usage-hour counts per facility
    fac_hour_dec: np.ndarray    # (N, F, 24)  decayed
    fac_hour_wkend: np.ndarray  # (N, F, 2, 24) split by weekday/weekend usage
    all_hour: np.ndarray        # (N, 24)
    last_hour_f: np.ndarray     # (N, F) hour of most recent use of facility (nan if never)
    sched: np.ndarray           # (N, N_OFF) usage already scheduled at day offset o (any facility)
    sched_f: np.ndarray         # (N, F, N_OFF)
    off_share: np.ndarray       # (N, N_OFF) historic share of "next-booking day offset"
    off_share_f: np.ndarray     # (N, F, N_OFF)
    bkrel_med_f: np.ndarray     # (N, F) median booking minute relative to usage-day midnight
    lead_med_f: np.ndarray      # (N, F) median lead minutes per facility
    long_off_med: np.ndarray    # (N,) median offset among offsets >= 10 (for the "10+" class)
    t0: np.ndarray              # (N,) anchor minute
    t0_day: np.ndarray          # (N,) anchor day index


def _resident_arrays(g: pd.DataFrame) -> dict:
    b = to_min(g["booking_ts"])
    u = to_min(g["usage_ts"])
    fac = g["facility_id"].map({f: i for i, f in enumerate(C.FACILITY_IDS)}).to_numpy()
    uday = u // DAY
    bday = b // DAY
    off_prev = np.full(len(b), -1)
    off_prev[1:] = uday[1:] - bday[:-1]  # day offset of booking j relative to booking j-1's creation day
    return dict(b=b, u=u, fac=fac, uday=uday, udow=uday % 7, uhour=(u % DAY) // 60,
                lead=u - b, bkrel=b - uday * DAY, btod=b % DAY, ldays=uday - bday,
                off_prev=off_prev, ids=g["booking_id"].to_numpy())


def build_state(df: pd.DataFrame, only_ids: set | None = None, live: bool = False) -> State:
    """Build per-target snapshots. ``df`` may be truncated -- features never look past t0.

    ``live=True`` builds exactly one row per resident anchored at their latest booking (no target
    known yet; ``y_*`` columns are -1). This is the production "predict the next booking" call.
    """
    F = C.N_FAC
    gb = to_min(df["booking_ts"])
    gfac = df["facility_id"].map({f: i for i, f in enumerate(C.FACILITY_IDS)}).to_numpy()
    order = np.argsort(gb, kind="stable")
    gb_sorted = gb[order]
    gcum = np.vstack([np.zeros(F), np.cumsum(np.eye(F)[gfac[order]], axis=0)])  # community counts

    meta_rows, base_rows = [], []
    arrs = {k: [] for k in ["fac_dow", "all_dow_dec", "fac_hour", "fac_hour_dec", "fac_hour_wkend",
                            "all_hour", "last_hour_f", "sched", "sched_f", "off_share", "off_share_f",
                            "bkrel_med_f", "lead_med_f", "long_off_med", "t0", "t0_day"]}

    for rid, g in df.groupby("resident_id", sort=True):
        a = _resident_arrays(g)
        n = len(a["b"])
        ks = [n] if live else range(1, n)
        for k in ks:
            if only_ids is not None and a["ids"][k] not in only_ids:
                continue
            t0 = a["b"][k - 1]
            h = slice(0, k)  # visible history: bookings 0..k-1 (all created <= t0)
            fac, uday, udow, uhour = a["fac"][h], a["uday"][h], a["udow"][h], a["uhour"][h]
            b, u = a["b"][h], a["u"][h]
            t0_day = t0 // DAY
            age_days = (t0 - b) / DAY
            w = 0.5 ** (age_days / HALF_LIFE_DAYS)
            wk = (udow >= 5).astype(int)

            fac_cnt = np.bincount(fac, minlength=F).astype(float)
            fac_dec = np.bincount(fac, weights=w, minlength=F)
            last10 = np.bincount(fac[-10:], minlength=F).astype(float)
            fac_dow = np.zeros((F, 7)); np.add.at(fac_dow, (fac, udow), 1)
            fac_hour = np.zeros((F, 24)); np.add.at(fac_hour, (fac, uhour), 1)
            fac_hour_dec = np.zeros((F, 24)); np.add.at(fac_hour_dec, (fac, uhour), w)
            fac_hour_wk = np.zeros((F, 2, 24)); np.add.at(fac_hour_wk, (fac, wk, uhour), 1)
            all_dow_dec = np.bincount(udow, weights=w, minlength=7)
            last_hour_f = np.full(F, np.nan)
            last_hour_f[fac] = uhour  # later bookings overwrite earlier ones -> most recent

            fut = u > t0
            so = np.clip(uday[fut] - t0_day, 0, N_OFF - 1)
            sched = np.zeros(N_OFF); np.add.at(sched, so, 1)
            sched_f = np.zeros((F, N_OFF)); np.add.at(sched_f, (fac[fut], so), 1)

            offs = a["off_prev"][1:k]
            offs_c = np.clip(offs, 0, N_OFF - 1)
            off_cnt = np.bincount(offs_c, minlength=N_OFF).astype(float)
            off_cnt_f = np.zeros((F, N_OFF)); np.add.at(off_cnt_f, (fac[1:], offs_c), 1)
            long_offs = offs[offs >= N_OFF - 1]

            bkrel, lead = a["bkrel"][h], a["lead"][h]
            bkrel_med_f = np.full(F, np.nan)
            lead_med_f = np.full(F, np.nan)
            for f in np.unique(fac):
                m = fac == f
                bkrel_med_f[f] = np.median(bkrel[m][-15:])
                lead_med_f[f] = np.median(lead[m][-15:])

            # community popularity in the 30 days before t0 (all residents, booking_ts <= t0)
            i1 = np.searchsorted(gb_sorted, t0, side="right")
            i0 = np.searchsorted(gb_sorted, t0 - 30 * DAY, side="right")
            pop30 = gcum[i1] - gcum[i0]
            pop30 = pop30 / max(pop30.sum(), 1)

            gaps = np.diff(b) / DAY
            ld = a["ldays"][h]
            base = {
                "n_hist": k,
                "hist_span_days": (t0 - b[0]) / DAY,
                "recent_28d": int((age_days <= 28).sum()),
                "gap_last": gaps[-1] if k > 1 else np.nan,
                "gap_med5": np.median(gaps[-5:]) if k > 1 else np.nan,
                "t0_dow": t0_day % 7,
                "t0_tod": t0 % DAY,
                "t0_month": (EPOCH + pd.Timedelta(minutes=int(t0))).month,
                "last_fac": fac[-1],
                "prev_fac": fac[-2] if k > 1 else -1,
                "last_fac_streak": int(np.argmax(fac[::-1] != fac[-1])) if (fac != fac[-1]).any() else k,
                "last_uhour": uhour[-1],
                "last_udow": udow[-1],
                "last_off": uday[-1] - t0_day,
                "last_lead_h": lead[-1] / 60,
                "lead_med_h": np.median(lead[-20:]) / 60,
                "lead_iqr_h": np.subtract(*np.percentile(lead[-20:], [75, 25])) / 60,
                "frac_same_day": float((ld[-20:] == 0).mean()),
                "frac_day_before": float((ld[-20:] == 1).mean()),
                "btod_med": np.median(a["btod"][h][-20:]),
                "btod_std": np.std(a["btod"][h][-20:]),
                "bkrel_med": np.median(bkrel[-20:]),
                "bkrel_last": bkrel[-1],
                "n_sched": int(fut.sum()),
                "max_sched_off": (uday[fut].max() - t0_day) if fut.any() else -1,
                "uhour_mean": uhour.mean(),
                "uhour_std": uhour.std(),
                "n_fac_used": int((fac_cnt > 0).sum()),
            }
            sh, dsh, l10 = _share(fac_cnt), _share(fac_dec), _share(last10)
            for f in range(F):
                base[f"share_{f}"] = sh[f]
                base[f"dshare_{f}"] = dsh[f]
                base[f"l10share_{f}"] = l10[f]
                base[f"pop30_{f}"] = pop30[f]
                base[f"sched_f_{f}"] = sched_f[f].sum()
            ds = _share(all_dow_dec)
            for o in range(N_OFF):  # weekday preference aligned to candidate day offsets
                base[f"dow_dshare_off{o}"] = ds[(t0_day + o) % 7]
                base[f"sched_off{o}"] = sched[o]

            base_rows.append(base)
            if live:
                meta_rows.append(dict(booking_id="NEXT", resident_id=rid, anchor_id=a["ids"][k - 1],
                                      **{c: -1 for c in TARGET_COLS}))
            else:
                meta_rows.append(dict(
                    booking_id=a["ids"][k], resident_id=rid, anchor_id=a["ids"][k - 1],
                    y_fac=a["fac"][k], y_off=min(a["uday"][k] - t0_day, N_OFF - 1),
                    y_off_raw=a["uday"][k] - t0_day, y_hour=a["uhour"][k],
                    y_bkrel=a["bkrel"][k], y_book=a["b"][k], y_use=a["u"][k],
                ))
            for key, val in [("fac_dow", fac_dow), ("all_dow_dec", all_dow_dec), ("fac_hour", fac_hour),
                             ("fac_hour_dec", fac_hour_dec), ("fac_hour_wkend", fac_hour_wk),
                             ("all_hour", np.bincount(uhour, minlength=24).astype(float)),
                             ("last_hour_f", last_hour_f), ("sched", sched), ("sched_f", sched_f),
                             ("off_share", _share(off_cnt)), ("off_share_f", _share(off_cnt_f)),
                             ("bkrel_med_f", bkrel_med_f), ("lead_med_f", lead_med_f),
                             ("long_off_med", np.median(long_offs) if len(long_offs) else 14.0),
                             ("t0", t0), ("t0_day", t0_day)]:
                arrs[key].append(val)

    meta = pd.DataFrame(meta_rows)
    base = pd.DataFrame(base_rows)
    return State(meta=meta, base=base, **{k: np.array(v) for k, v in arrs.items()})


def subset(s: State, mask: np.ndarray) -> State:
    idx = np.flatnonzero(mask)
    kw = {k: getattr(s, k)[idx] for k in State.__dataclass_fields__ if k not in ("meta", "base")}
    return State(meta=s.meta.iloc[idx].reset_index(drop=True), base=s.base.iloc[idx].reset_index(drop=True), **kw)


# ---------------------------------------------------------------- stage feature matrices
def X_facility(s: State) -> pd.DataFrame:
    return s.base


def X_day(s: State, f: np.ndarray) -> pd.DataFrame:
    n = len(f)
    r = np.arange(n)
    X = s.base.copy()
    X["c_fac"] = f
    X["c_fac_share"] = s.base.to_numpy()[r, s.base.columns.get_indexer([f"share_{i}" for i in range(C.N_FAC)])[f]]
    fdow = _share(s.fac_dow[r, f])  # (n, 7)
    for o in range(N_OFF):
        dow_o = (s.t0_day + o) % 7
        X[f"c_fdow_off{o}"] = fdow[r, dow_o]
        X[f"c_sched_f_off{o}"] = s.sched_f[r, f, o]
        X[f"off_share{o}"] = s.off_share[:, o]
        X[f"c_off_share_f{o}"] = s.off_share_f[r, f, o]
    return X


def offset_to_days(off_cls: np.ndarray, s: State) -> np.ndarray:
    """Map the day-offset class back to a day count ("10+" -> resident's typical long offset)."""
    return np.where(off_cls >= N_OFF - 1, np.maximum(s.long_off_med, N_OFF - 1), off_cls).astype(int)


def X_hour(s: State, f: np.ndarray, off_days: np.ndarray) -> pd.DataFrame:
    n = len(f)
    r = np.arange(n)
    dow = (s.t0_day + off_days) % 7
    wk = (dow >= 5).astype(int)
    X = s.base.copy()
    X["c_fac"] = f
    X["c_off"] = off_days
    X["c_dow"] = dow
    X["c_last_hour_f"] = s.last_hour_f[r, f]
    fh = _share(s.fac_hour[r, f][:, HOURS])
    fhd = _share(s.fac_hour_dec[r, f][:, HOURS])
    fhw = _share(s.fac_hour_wkend[r, f, wk][:, HOURS])
    ah = _share(s.all_hour[:, HOURS])
    X["c_fac_mode_hour"] = np.where(np.isnan(fh).all(1), np.nan, HOURS[np.nan_to_num(fh, nan=-1).argmax(1)])
    X["c_fac_n"] = s.fac_hour[r, f].sum(1)
    for j, hh in enumerate(HOURS):
        X[f"c_fh{hh}"] = fh[:, j]
        X[f"c_fhd{hh}"] = fhd[:, j]
        X[f"c_fhw{hh}"] = fhw[:, j]
        X[f"ah{hh}"] = ah[:, j]
    return X


def X_nudge(s: State, f: np.ndarray, off_days: np.ndarray, hour: np.ndarray) -> pd.DataFrame:
    r = np.arange(len(f))
    uday = s.t0_day + off_days
    X = s.base.copy()
    X["c_fac"] = f
    X["c_off"] = off_days
    X["c_hour"] = hour
    X["c_dow"] = uday % 7
    X["c_bkrel_med_f"] = s.bkrel_med_f[r, f]
    X["c_lead_med_f"] = s.lead_med_f[r, f]
    X["c_hour_minus_lead"] = hour * 60 - s.base["lead_med_h"].to_numpy() * 60
    X["c_hour_minus_lead_f"] = hour * 60 - s.lead_med_f[r, f]
    X["c_t0_rel"] = s.t0 - uday * DAY  # lower bound: next booking is created after t0
    return X


def nudge_minutes(s: State, off_days: np.ndarray, hour: np.ndarray, bkrel_pred: np.ndarray) -> np.ndarray:
    """Turn the predicted booking minute (relative to usage-day midnight) into an absolute nudge time,
    clamped to (t0, usage - 15 min)."""
    uday = s.t0_day + off_days
    usage = uday * DAY + hour * 60
    nudge = uday * DAY + np.round(bkrel_pred).astype(np.int64)
    return np.clip(nudge, s.t0 + 1, np.maximum(usage - 15, s.t0 + 1))
