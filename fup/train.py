"""Train the chained model.

1. build leakage-safe features for every target booking
2. tune hyper-parameters per stage on the validation window (Sep-Oct 2025) -- test is never touched
3. refit the chosen configuration on train + validation
4. save the model artifact and training metadata to ``models/``

Run:  python -m fup.train [--no-tune]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config as C
from . import features as Fe
from . import model as M
from .split import split_states

GRID = [
    dict(learning_rate=0.1, max_iter=100, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0),
    dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0),
    dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0),
]


def _teacher(s: Fe.State):
    m = s.meta
    off = m.y_off.to_numpy()
    return m.y_fac.to_numpy(), np.where(off >= Fe.N_OFF - 1, m.y_off_raw.to_numpy(), off), m.y_hour.to_numpy()


def _stage_xy(stage: str, s: Fe.State):
    f, od, h = _teacher(s)
    m = s.meta
    if stage == "facility":
        return Fe.X_facility(s), m.y_fac.to_numpy()
    if stage == "day":
        return Fe.X_day(s, f), m.y_off.to_numpy()
    if stage == "hour":
        return Fe.X_hour(s, f, od), m.y_hour.to_numpy()
    return Fe.X_nudge(s, f, od, h), m.y_bkrel.to_numpy()


def tune(tr: Fe.State, va: Fe.State) -> tuple[dict, list]:
    """Per-stage grid search with teacher-forced conditioning; score = accuracy (classifiers) or
    share of booking-time predictions within the nudge tolerance (regressor)."""
    best, log = {}, []
    for stage in M.STAGES:
        Xtr, ytr = _stage_xy(stage, tr)
        Xva, yva = _stage_xy(stage, va)
        Xtr, Xva = M._prep(Xtr), M._prep(Xva)
        scores = []
        for g in GRID:
            if stage == "nudge":
                est = M.nudge_regressor(Xtr, g, C.SEED).fit(Xtr, ytr)
                score = float((np.abs(est.predict(Xva) - yva) <= C.NUDGE_TOLERANCE_MIN).mean())
            else:
                est = HistGradientBoostingClassifier(categorical_features=M._cat_mask(Xtr), random_state=C.SEED,
                                                     early_stopping=False, **g).fit(Xtr, ytr)
                score = float((est.predict(Xva) == yva).mean())
            scores.append(score)
            log.append(dict(stage=stage, **g, val_score=round(score, 4)))
            print(f"  tune {stage:8s} {g} -> {score:.4f}", flush=True)
        best[stage] = GRID[int(np.argmax(scores))]
    return best, log


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tune", action="store_true", help="skip the validation grid search (use defaults)")
    args = ap.parse_args(argv)
    t = time.time()

    df = Fe.load_bookings()
    print(f"loaded {len(df):,} bookings; building leakage-safe features ...", flush=True)
    sp = split_states(Fe.build_state(df))
    for k in ("train", "val", "test"):
        print(f"  {k:5s}: {len(sp[k].meta):,} target bookings")

    if args.no_tune:
        params, tune_log = {st: dict(M.DEFAULT_PARAMS) for st in M.STAGES}, []
    else:
        print("tuning on validation window ...", flush=True)
        params, tune_log = tune(sp["train"], sp["val"])

    # validation score of the full (non-teacher-forced) chain with the chosen params -- for the record
    val_model = M.ChainModel(params=params).fit(sp["train"])
    from .metrics import evaluate
    val_metrics = evaluate(sp["val"], val_model.predict(sp["val"]))
    print(f"validation mean output accuracy (chain, trained on train only): {val_metrics['mean_output_accuracy']:.4f}")

    print("refitting on train + validation ...", flush=True)
    model = M.ChainModel(params=params).fit(sp["train_val"])

    C.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, C.MODEL_DIR / "chain_model.joblib", compress=3)
    meta = dict(
        trained_on=f"target bookings created before {C.TEST_START.date()}",
        n_train_targets=len(sp["train_val"].meta),
        split=dict(train_end=str(C.VAL_START.date()), val_end=str(C.TEST_START.date()), test_end=str(C.DATA_END.date())),
        params=params, decode=model.decode, seed=C.SEED,
        data_sha256=hashlib.sha256(C.BOOKINGS_CSV.read_bytes()).hexdigest(),
        sklearn=sklearn.__version__, val_metrics={k: round(float(v), 4) for k, v in val_metrics.items()},
        tuning_log=tune_log,
    )
    (C.MODEL_DIR / "training_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"saved model -> {C.MODEL_DIR / 'chain_model.joblib'}  ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
