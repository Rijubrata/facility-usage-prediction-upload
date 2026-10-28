"""Fast checks: data sanity, leakage safety, split hygiene, metric logic. Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from fup import config as C
from fup import features as Fe
from fup import model as M
from fup.metrics import evaluate, match_flags
from fup.split import split_states


@pytest.fixture(scope="module")
def df():
    return Fe.load_bookings()


@pytest.fixture(scope="module")
def small(df):
    # 40 residents keeps the tests quick while exercising every code path
    keep = sorted(df.resident_id.unique())[:40]
    return df[df.resident_id.isin(keep)].reset_index(drop=True)


def test_dataset_sanity(df):
    assert {"booking_id", "resident_id", "facility_id", "booking_ts", "usage_ts"} <= set(df.columns)
    assert df.booking_id.is_unique
    assert (df.booking_ts < df.usage_ts).all(), "a booking must be created before its usage slot"
    assert df.booking_ts.min() >= C.DATA_START and df.booking_ts.max() < C.DATA_END
    assert set(df.facility_id) <= set(C.FACILITY_IDS)
    fac = pd.DataFrame(C.FACILITIES).set_index(0)
    h = df.usage_ts.dt.hour
    assert (h >= df.facility_id.map(fac[3])).all() and (h <= df.facility_id.map(fac[4])).all()
    # capacity respected
    occ = df.groupby(["facility_id", "usage_ts"]).size()
    cap = occ.index.get_level_values(0).map(fac[5])
    assert (occ.to_numpy() <= cap.to_numpy()).all()


def test_features_do_not_peek_into_the_future(small):
    """Features for a target must be identical whether or not later bookings exist in the table."""
    full = Fe.build_state(small)
    cut = pd.Timestamp("2025-07-01")
    trunc = Fe.build_state(small[small.booking_ts <= cut])
    # targets whose anchor (= previous booking) is before the cut AND whose own booking is before the cut
    anchor_ok = Fe.from_min(full.t0) <= cut
    tgt_ok = Fe.from_min(full.meta.y_book.to_numpy()) <= cut
    ids = full.meta.booking_id[anchor_ok & tgt_ok]
    a = full.base[full.meta.booking_id.isin(ids)].reset_index(drop=True)
    b = trunc.base[trunc.meta.booking_id.isin(ids)].reset_index(drop=True)
    assert len(a) == len(b) > 100
    pd.testing.assert_frame_equal(a, b)


def test_features_ignore_target_row(small):
    """Changing the target booking itself must not change its own features."""
    s1 = Fe.build_state(small)
    mod = small.copy()
    i = len(mod) // 2
    target_id = mod.booking_id.iloc[i]
    mod.loc[i, "facility_id"] = "F08" if mod.loc[i, "facility_id"] != "F08" else "F01"
    mod.loc[i, "usage_ts"] = mod.loc[i, "usage_ts"] + pd.Timedelta(hours=1)
    s2 = Fe.build_state(mod)
    r1 = s1.base[s1.meta.booking_id == target_id].reset_index(drop=True)
    r2 = s2.base[s2.meta.booking_id == target_id].reset_index(drop=True)
    # community popularity counts the target only if it was created <= its own anchor -- impossible
    pd.testing.assert_frame_equal(r1, r2)


def test_split_is_chronological_and_disjoint(df):
    sp = split_states(Fe.build_state(df[df.resident_id.isin(sorted(df.resident_id.unique())[:30])]))
    tr, va, te = (Fe.from_min(sp[k].meta.y_book.to_numpy()) for k in ("train", "val", "test"))
    assert tr.max() < C.VAL_START <= va.min() and va.max() < C.TEST_START <= te.min()
    ids = [set(sp[k].meta.booking_id) for k in ("train", "val", "test")]
    assert not (ids[0] & ids[1]) and not (ids[1] & ids[2]) and not (ids[0] & ids[2])


def test_metrics_match_flags():
    s = Fe.build_state(Fe.load_bookings().head(400))
    m = s.meta
    perfect = dict(fac=m.y_fac.to_numpy(), off=(m.y_use.to_numpy() // Fe.DAY) - s.t0_day,
                   hour=m.y_hour.to_numpy(), nudge=m.y_book.to_numpy(),
                   proba_fac=np.eye(C.N_FAC)[m.y_fac.to_numpy()])
    fl = match_flags(s, perfect)
    assert all(v.all() for v in fl.values())
    assert evaluate(s, perfect)["all4_match"] == 1.0


def test_chain_model_end_to_end(small):
    sp = split_states(Fe.build_state(small))
    fast = {st: dict(learning_rate=0.2, max_iter=20, max_leaf_nodes=7, min_samples_leaf=20) for st in M.STAGES}
    mdl = M.ChainModel(params=fast).fit(sp["train"])
    p = mdl.predict(sp["test"])
    assert len(p["fac"]) == len(sp["test"].meta)
    assert (p["nudge"] > sp["test"].t0).all(), "nudge must be after the prediction time"
    usage = (sp["test"].t0_day + p["off"]) * Fe.DAY + p["hour"] * 60
    assert (p["nudge"] <= np.maximum(usage - 15, sp["test"].t0 + 1)).all()
    assert evaluate(sp["test"], p)["mean_output_accuracy"] > 0.2
