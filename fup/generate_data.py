"""Synthetic facility-booking generator for one residential community.

Produces ``data/bookings.csv`` with the four core fields required by the brief
(resident, facility, booking timestamp, usage timestamp) plus a booking id, and
small reference tables. The behaviour it simulates:

* resident preferences   - persona-driven facility mix with individual variation
* facility popularity     - gym >> pool > badminton > ... > clubhouse (imbalance)
* time patterns           - preferred weekdays and per-facility preferred hours
* booking lead times      - planners, same-day bookers, weekly batch bookers, ad-hoc
* sparsity                - heavy-tailed activity rates, casual users, move-ins, breaks
* noise                   - exploratory bookings, hour jitter, booking-minute jitter
* changing behaviour      - per-resident drift events + facility seasonality
* capacity                - slots fill up in booking order; late bookers get shifted

Run:  python -m fup.generate_data
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

N_RESIDENTS = 320
P_EXPLORE = 0.06

FAC_IDX = {f[0]: i for i, f in enumerate(C.FACILITIES)}
OPEN = np.array([f[3] for f in C.FACILITIES])
LAST = np.array([f[4] for f in C.FACILITIES])
CAP = np.array([f[5] for f in C.FACILITIES])
GLOBAL_POP = np.array([f[6] for f in C.FACILITIES])
GLOBAL_POP = GLOBAL_POP / GLOBAL_POP.sum()
CLUBHOUSE = FAC_IDX["F08"]
CLUBHOUSE_STARTS = [11, 16, 19]

# Month (1..12) multipliers per facility -> seasonality / changing behaviour.
SEASON = {
    "F01": [1.35, 1.2, 1.05, 1.0, 0.95, 0.95, 1.0, 1.0, 1.0, 0.95, 0.95, 0.85],  # new-year resolutions
    "F02": [0.45, 0.6, 0.95, 1.4, 1.6, 1.3, 0.9, 0.9, 1.0, 1.0, 0.7, 0.5],        # summer pool
    "F03": [1.0] * 12,
    "F04": [1.1, 1.1, 1.0, 0.9, 0.8, 0.7, 0.45, 0.45, 0.6, 1.0, 1.15, 1.15],      # outdoor, monsoon dip
    "F05": [1.0] * 12,
    "F06": [1.0, 1.0, 1.0, 1.1, 1.2, 1.2, 1.1, 1.1, 1.0, 1.0, 1.0, 1.1],
    "F07": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.15, 1.15, 1.1, 1.0, 1.0, 1.0],        # indoor in monsoon
    "F08": [0.8, 0.8, 0.9, 0.8, 0.8, 0.7, 0.7, 0.8, 1.0, 1.6, 1.7, 1.8],          # festive season
}
SEASON_M = np.array([SEASON[f] for f in C.FACILITY_IDS])  # (n_fac, 12)
MONTH_ACTIVITY = [1.1, 1.05, 1.0, 1.0, 0.95, 0.95, 0.9, 0.95, 1.0, 1.0, 0.95, 0.8]

# Personas: facility weights, candidate preferred hours, day type, mean sessions/week, lead-type mix.
PERSONAS = {
    "early_gym": dict(share=0.17, fac={"F01": 0.85, "F02": 0.08, "F07": 0.07},
                      hours=[5, 6, 6, 7, 7, 8], days="weekday", rate=3.2,
                      lead={"planner": 0.45, "same_day": 0.25, "batch": 0.2, "adhoc": 0.1}),
    "evening_gym": dict(share=0.12, fac={"F01": 0.75, "F03": 0.12, "F06": 0.13},
                        hours=[18, 19, 19, 20, 21], days="weekday", rate=2.6,
                        lead={"planner": 0.3, "same_day": 0.45, "batch": 0.1, "adhoc": 0.15}),
    "swimmer": dict(share=0.10, fac={"F02": 0.78, "F01": 0.22},
                    hours=[6, 7, 17, 18], days="any", rate=2.2,
                    lead={"planner": 0.45, "same_day": 0.3, "batch": 0.05, "adhoc": 0.2}),
    "racket": dict(share=0.12, fac={"F03": 0.5, "F04": 0.2, "F05": 0.2, "F06": 0.1},
                   hours=[19, 20, 20, 21, 7], days="any", rate=2.0,
                   lead={"planner": 0.55, "same_day": 0.2, "batch": 0.05, "adhoc": 0.2}),
    "family_weekend": dict(share=0.12, fac={"F02": 0.35, "F03": 0.2, "F06": 0.2, "F07": 0.1, "F08": 0.15},
                           hours=[9, 10, 11, 16, 17, 18], days="weekend", rate=1.3,
                           lead={"planner": 0.5, "same_day": 0.15, "batch": 0.0, "adhoc": 0.35}),
    "senior_wellness": dict(share=0.08, fac={"F07": 0.72, "F02": 0.14, "F01": 0.14},
                            hours=[6, 7, 7, 8, 9], days="weekday", rate=3.0,
                            lead={"planner": 0.6, "same_day": 0.1, "batch": 0.25, "adhoc": 0.05}),
    "casual": dict(share=0.29, fac={f: p for f, p in zip(C.FACILITY_IDS, GLOBAL_POP)},
                   hours=[7, 9, 11, 17, 18, 19, 20], days="any", rate=0.45,
                   lead={"planner": 0.25, "same_day": 0.25, "batch": 0.0, "adhoc": 0.5}),
}
DRIFT_TYPES = ["time_shift", "facility_switch", "rate_change", "lead_change"]


def _vec(fac_weights: dict) -> np.ndarray:
    v = np.zeros(C.N_FAC)
    for f, w in fac_weights.items():
        v[FAC_IDX[f]] = w
    return v / v.sum()


def _clip_hour(fi: int, h: int) -> int:
    if fi == CLUBHOUSE:
        return min(CLUBHOUSE_STARTS, key=lambda s: abs(s - h))
    return int(np.clip(h, OPEN[fi], LAST[fi]))


def _draw_pref_hours(rng, persona: dict) -> np.ndarray:
    hours = np.zeros(C.N_FAC, dtype=int)
    for fi in range(C.N_FAC):
        h = rng.choice(persona["hours"]) + rng.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
        hours[fi] = _clip_hour(fi, h)
    return hours


def _draw_fac_pref(rng, persona: dict) -> np.ndarray:
    alpha = 25 * _vec(persona["fac"]) + 1.5 * GLOBAL_POP
    return rng.dirichlet(alpha)


def _draw_lead(rng, persona: dict) -> dict:
    kinds, probs = zip(*persona["lead"].items())
    kind = rng.choice(kinds, p=probs)
    return dict(
        kind=kind,
        planner_tod=float(np.clip(rng.normal(19.5 * 60, 100), 7 * 60, 23 * 60)),  # minutes of day
        tod_sd=float(rng.uniform(12, 35)),
        same_day_lead_h=float(rng.uniform(1.0, 4.0)),
        batch_tod=float(np.clip(rng.normal(20 * 60, 60), 9 * 60, 23 * 60)),
    )


def make_residents(rng) -> list[dict]:
    names = list(PERSONAS)
    shares = np.array([PERSONAS[n]["share"] for n in names])
    residents = []
    for i in range(N_RESIDENTS):
        pname = rng.choice(names, p=shares / shares.sum())
        p = PERSONAS[pname]
        base_dow = {"weekday": [1, 1, 1, 1, 1, 0.3, 0.3],
                    "weekend": [0.1, 0.1, 0.1, 0.1, 0.25, 1, 1],
                    "any": [1] * 7}[p["days"]]
        base_dow = np.array(base_dow, float)
        rate = p["rate"] * rng.lognormal(0, 0.45)
        n_pref_days = int(np.clip(round(rate) + 1, 1, 6))
        pref_days = rng.choice(7, size=n_pref_days, replace=False, p=base_dow / base_dow.sum())
        dow_w = 0.12 * base_dow
        dow_w[pref_days] += 1.0

        start = C.DATA_START - pd.Timedelta(days=21)
        if rng.random() < 0.12:  # move-in during the year
            start = C.DATA_START + pd.Timedelta(days=int(rng.integers(30, 290)))
        end = C.DATA_END + pd.Timedelta(days=30)
        if rng.random() < 0.06:  # move-out / stop using facilities
            end = C.DATA_START + pd.Timedelta(days=int(rng.integers(90, 340)))

        breaks = []  # vacations: 0-2 gaps of 1-3 weeks
        for _ in range(rng.integers(0, 3)):
            b0 = C.DATA_START + pd.Timedelta(days=int(rng.integers(0, 350)))
            breaks.append((b0, b0 + pd.Timedelta(days=int(rng.integers(7, 22)))))

        drift = None
        if rng.random() < 0.28:
            drift = dict(date=C.DATA_START + pd.Timedelta(days=int(rng.integers(60, 335))),
                         type=rng.choice(DRIFT_TYPES))

        residents.append(dict(
            resident_id=f"R-{101 + i:03d}", persona=pname, fac_pref=_draw_fac_pref(rng, p),
            pref_hours=_draw_pref_hours(rng, p), weekend_shift=int(rng.choice([0, 0, 1, 2])),
            p_exact_hour=float(rng.beta(8, 2)), rate=rate, dow_w=dow_w / dow_w.sum(),
            lead=_draw_lead(rng, p), start=start, end=end, breaks=breaks, drift=drift,
            pref_days=sorted(int(d) for d in pref_days),
        ))
    return residents


def _apply_drift(rng, r: dict) -> dict:
    """Return the post-drift version of a resident's behaviour."""
    d = r["drift"]
    new = dict(r)
    other = PERSONAS[rng.choice([n for n in PERSONAS if n != r["persona"] and n != "casual"])]
    if d["type"] == "time_shift":  # e.g. new job: morning -> evening sessions
        new["pref_hours"] = _draw_pref_hours(rng, other)
    elif d["type"] == "facility_switch":
        new["fac_pref"] = _draw_fac_pref(rng, other)
    elif d["type"] == "rate_change":
        new["rate"] = r["rate"] * float(rng.choice([0.35, 2.0]))
    elif d["type"] == "lead_change":
        lead = _draw_lead(rng, other)
        while lead["kind"] == r["lead"]["kind"]:
            lead = _draw_lead(rng, other)
        new["lead"] = lead
    return new


# All simulation arithmetic is done in integer minutes since ORIGIN (a Monday) -- fast and exact.
ORIGIN = pd.Timestamp("2024-12-02")  # Monday
DAY = 1440


def _m(ts: pd.Timestamp) -> int:
    return int((ts - ORIGIN) / pd.Timedelta(minutes=1))


def _booking_time(rng, lead: dict, usage: int, fi: int) -> int:
    """Return the booking-creation minute for a usage minute (both ints since ORIGIN)."""
    day = usage - usage % DAY
    uh = (usage % DAY) // 60
    if fi == CLUBHOUSE:  # parties are planned well ahead
        return day - int(rng.integers(3, 21)) * DAY + int(rng.integers(9 * 60, 22 * 60))
    kind = lead["kind"]
    if rng.random() < 0.05:  # noise: occasionally behave ad hoc
        kind = "adhoc"
    if kind == "planner":
        back = 1 if rng.random() < 0.85 else 2
        tod = np.clip(rng.normal(lead["planner_tod"], lead["tod_sd"]), 6 * 60, 23 * 60 + 50)
        return day - back * DAY + int(tod)
    if kind == "same_day":
        if uh < 9:  # nobody books at 4am -> the night before
            tod = np.clip(rng.normal(22 * 60, lead["tod_sd"]), 20 * 60, 23 * 60 + 50)
            return day - DAY + int(tod)
        lead_h = max(0.33, rng.normal(lead["same_day_lead_h"], 0.5))
        return usage - int(lead_h * 60)
    if kind == "batch":  # book the whole week on the preceding Sunday evening
        dow = (day // DAY) % 7  # ORIGIN is Monday -> 0
        sunday = day - (dow + 1) * DAY
        tod = np.clip(rng.normal(lead["batch_tod"], lead["tod_sd"] / 2), 8 * 60, 23 * 60 + 30)
        return sunday + int(tod) + dow * 2
    # adhoc
    back = int(min(rng.geometric(0.45) - 1, 6))
    if back == 0 and uh <= 8:
        back = 1
    tod = rng.uniform(7 * 60, uh * 60 - 20) if back == 0 else rng.uniform(7 * 60, 23 * 60 + 30)
    return day - back * DAY + int(tod)


def simulate(rng, residents: list[dict]) -> pd.DataFrame:
    rows = []
    week = _m(C.DATA_START) - ((_m(C.DATA_START) // DAY) % 7) * DAY - 14 * DAY
    last_week = _m(C.DATA_END) + 21 * DAY
    drifted = {r["resident_id"]: _apply_drift(rng, r) for r in residents if r["drift"]}
    for r in residents:  # pre-convert per-resident dates
        r["_start"], r["_end"] = _m(r["start"]), _m(r["end"])
        r["_breaks"] = [(_m(a), _m(b)) for a, b in r["breaks"]]
        r["_drift"] = _m(r["drift"]["date"]) if r["drift"] else None
    while week < last_week:
        m = (ORIGIN + pd.Timedelta(minutes=week + 3 * DAY)).month - 1
        for r in residents:
            beh = drifted[r["resident_id"]] if r["_drift"] is not None and week >= r["_drift"] else r
            season_fac = beh["fac_pref"] * SEASON_M[:, m]
            rate = beh["rate"] * season_fac.sum() * MONTH_ACTIVITY[m]
            n = min(rng.poisson(rate), 7)
            if n == 0:
                continue
            days = rng.choice(7, size=n, replace=False, p=beh["dow_w"])
            fac_p = season_fac / season_fac.sum()
            for d in sorted(days):
                date = week + int(d) * DAY
                if not (r["_start"] <= date < r["_end"]) or any(a <= date < b for a, b in r["_breaks"]):
                    continue
                fi = int(rng.choice(C.N_FAC, p=GLOBAL_POP if rng.random() < P_EXPLORE else fac_p))
                h = beh["pref_hours"][fi] + (beh["weekend_shift"] if d >= 5 else 0)
                u = rng.random()
                if u > beh["p_exact_hour"]:
                    h = h + rng.choice([-1, 1]) if u < 0.97 else rng.integers(OPEN[fi], LAST[fi] + 1)
                usage = date + _clip_hour(fi, int(h)) * 60
                rows.append((r["resident_id"], fi, _booking_time(rng, beh["lead"], usage, fi), usage))
        week += 7 * DAY
    df = pd.DataFrame(rows, columns=["resident_id", "fi", "booking_m", "usage_m"])
    return df.sort_values(["booking_m", "resident_id"], kind="stable").reset_index(drop=True)


def apply_capacity(df: pd.DataFrame) -> pd.DataFrame:
    """Admit bookings in creation order; full slots push the resident to an adjacent hour or drop it."""
    occupied: dict[tuple, int] = {}
    placed_all = []
    for fi, usage in zip(df["fi"].to_numpy(), df["usage_m"].to_numpy()):
        placed = -1
        for dh in (0, 1, -1):
            u = int(usage) + dh * 60
            if dh and (fi == CLUBHOUSE or not OPEN[fi] <= (u % DAY) // 60 <= LAST[fi]):
                continue
            if occupied.get((fi, u), 0) < CAP[fi]:
                occupied[(fi, u)] = occupied.get((fi, u), 0) + 1
                placed = u
                break
        placed_all.append(placed)
    df = df.assign(usage_m=placed_all)
    print(f"capacity: {(df.usage_m < 0).sum()} booking attempts rejected (slot full)")
    df = df[df.usage_m >= 0]
    return df[df.booking_m < df.usage_m]


def main() -> None:
    rng = np.random.default_rng(C.SEED)
    residents = make_residents(rng)
    df = simulate(rng, residents)
    df = apply_capacity(df)
    df["booking_ts"] = ORIGIN + pd.to_timedelta(df["booking_m"], unit="min")
    df["usage_ts"] = ORIGIN + pd.to_timedelta(df["usage_m"], unit="min")
    df = df[(df["booking_ts"] >= C.DATA_START) & (df["booking_ts"] < C.DATA_END)].copy()
    df = df.sort_values(["booking_ts", "resident_id"], kind="stable").reset_index(drop=True)
    df.insert(0, "booking_id", [f"B{i + 1:06d}" for i in range(len(df))])
    df["facility_id"] = [C.FACILITY_IDS[i] for i in df["fi"]]
    out = df[["booking_id", "resident_id", "facility_id", "booking_ts", "usage_ts"]]

    C.DATA_DIR.mkdir(exist_ok=True)
    (C.DATA_DIR / "generator_truth").mkdir(exist_ok=True)
    out.to_csv(C.BOOKINGS_CSV, index=False, date_format="%Y-%m-%d %H:%M")

    pd.DataFrame(C.FACILITIES, columns=["facility_id", "name", "label", "open_hour", "last_start_hour",
                                        "capacity_per_slot", "base_popularity", "slot_minutes"]
                 ).drop(columns="base_popularity").to_csv(C.FACILITIES_CSV, index=False)

    towers = "ABCDEF"
    pd.DataFrame([dict(resident_id=r["resident_id"],
                       unit=f"{towers[i % 6]}-{(i // 6) % 20 + 1:02d}{i % 4 + 1:02d}")
                  for i, r in enumerate(residents)]).to_csv(C.RESIDENTS_CSV, index=False)

    pd.DataFrame([dict(
        resident_id=r["resident_id"], persona=r["persona"], lead_type=r["lead"]["kind"],
        sessions_per_week=round(r["rate"], 2),
        top_facility=C.FACILITY_IDS[int(np.argmax(r["fac_pref"]))],
        preferred_days="|".join(C.DAYS[d] for d in r["pref_days"]),
        move_in=r["start"].date() if r["start"] > C.DATA_START else "",
        move_out=r["end"].date() if r["end"] < C.DATA_END else "",
        drift_date=r["drift"]["date"].date() if r["drift"] else "",
        drift_type=r["drift"]["type"] if r["drift"] else "",
    ) for r in residents]).to_csv(C.TRUTH_CSV, index=False)

    print(f"wrote {len(out):,} bookings for {out.resident_id.nunique()} residents -> {C.BOOKINGS_CSV}")
    print(out.facility_id.map(C.FACILITY_LABEL).value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
