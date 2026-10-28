"""Chained four-output model + heuristic baselines."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from . import config as C
from . import features as Fe

CATEGORICAL = ["last_fac", "prev_fac", "t0_dow", "last_udow", "c_fac", "c_dow"]


def _prep(X):
    X = X.copy()
    if "prev_fac" in X:
        X["prev_fac"] = X["prev_fac"].where(X["prev_fac"] >= 0)  # no previous -> missing
    return X


def _cat_mask(X):
    return [c in CATEGORICAL for c in X.columns]


def _clf(X, params, seed):
    return HistGradientBoostingClassifier(categorical_features=_cat_mask(X), random_state=seed,
                                          early_stopping=False, **params)


DEFAULT_PARAMS = dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=100,
                      l2_regularization=1.0)
STAGES = ("facility", "day", "hour", "nudge")


@dataclass
class ChainModel:
    """facility -> day offset -> hour -> nudge, each stage conditioned on the previous ones."""
    params: dict = field(default_factory=lambda: {st: dict(DEFAULT_PARAMS) for st in STAGES})
    decode: str = "greedy"   # "greedy": argmax per stage | "joint": beam over (facility, day) maximising p(f)p(d|f)p(h|f,d)
    beam_f: int = 3
    beam_d: int = 3
    seed: int = C.SEED

    def fit(self, s: Fe.State) -> "ChainModel":
        m = s.meta
        f, off = m.y_fac.to_numpy(), m.y_off.to_numpy()
        off_days = np.where(off >= Fe.N_OFF - 1, m.y_off_raw.to_numpy(), off)
        hour = m.y_hour.to_numpy()

        P = self.params
        X = _prep(Fe.X_facility(s)); self.fac_ = _clf(X, P["facility"], self.seed).fit(X, f)
        X = _prep(Fe.X_day(s, f)); self.day_ = _clf(X, P["day"], self.seed).fit(X, off)
        X = _prep(Fe.X_hour(s, f, off_days)); self.hour_ = _clf(X, P["hour"], self.seed).fit(X, hour)
        X = _prep(Fe.X_nudge(s, f, off_days, hour))
        self.nudge_ = nudge_regressor(X, P["nudge"], self.seed).fit(X, m.y_bkrel.to_numpy())
        return self

    # ------------------------------------------------------------------ inference
    def _proba(self, model, X, n_classes):
        """Probabilities over the full class range (classes unseen in training get 0)."""
        p = np.zeros((len(X), n_classes))
        p[:, model.classes_] = model.predict_proba(_prep(X))
        return p

    def predict(self, s: Fe.State) -> dict:
        n = len(s.meta)
        pf = self._proba(self.fac_, Fe.X_facility(s), C.N_FAC)
        if self.decode == "greedy":
            cand_f = pf.argmax(1)[:, None]
        else:
            cand_f = np.argsort(-pf, 1)[:, : self.beam_f]

        best = dict(score=np.full(n, -1.0), f=np.zeros(n, int), off=np.zeros(n, int), hour=np.zeros(n, int),
                    p_day=np.zeros(n), p_hour=np.zeros(n))
        for i in range(cand_f.shape[1]):
            f = cand_f[:, i]
            pdays = self._proba(self.day_, Fe.X_day(s, f), Fe.N_OFF)
            cand_d = pdays.argmax(1)[:, None] if self.decode == "greedy" else np.argsort(-pdays, 1)[:, : self.beam_d]
            for j in range(cand_d.shape[1]):
                off_cls = cand_d[:, j]
                off_days = Fe.offset_to_days(off_cls, s)
                ph = self._proba(self.hour_, Fe.X_hour(s, f, off_days), 24)
                hour = ph.argmax(1)
                p_d = pdays[np.arange(n), off_cls]
                p_h = ph[np.arange(n), hour]
                score = pf[np.arange(n), f] * p_d * p_h
                upd = score > best["score"]
                for k, v in [("score", score), ("f", f), ("off", off_days), ("hour", hour), ("p_day", p_d), ("p_hour", p_h)]:
                    best[k] = np.where(upd, v, best[k])

        bkrel = self.nudge_.predict(_prep(Fe.X_nudge(s, best["f"], best["off"], best["hour"])))
        nudge = Fe.nudge_minutes(s, best["off"], best["hour"], bkrel)
        return dict(fac=best["f"], off=best["off"], hour=best["hour"], nudge=nudge,
                    p_fac=pf[np.arange(n), best["f"]], p_day=best["p_day"], p_hour=best["p_hour"],
                    p_joint=best["score"], proba_fac=pf)


def nudge_regressor(X, params, seed):
    # absolute-error loss -> predicts the conditional median booking minute (robust to ad-hoc outliers)
    return HistGradientBoostingRegressor(loss="absolute_error", categorical_features=_cat_mask(X),
                                         random_state=seed, early_stopping=False, **params)


class HeuristicBaseline:
    """Non-ML reference: 'do what you usually do'.

    facility = decayed-most-frequent facility; day = most common gap-to-next-usage-day;
    hour = most common hour at that facility; nudge = median booking minute (relative to the
    usage day) at that facility.
    """

    def fit(self, s):
        return self

    def predict(self, s: Fe.State) -> dict:
        n = len(s.meta)
        r = np.arange(n)
        dsh = s.base[[f"dshare_{i}" for i in range(C.N_FAC)]].to_numpy()
        f = np.nan_to_num(dsh, nan=0).argmax(1)
        offs = np.nan_to_num(s.off_share_f[r, f], nan=0)
        off_cls = np.where(offs.sum(1) > 0, offs.argmax(1), np.nan_to_num(s.off_share, nan=0).argmax(1))
        off_days = Fe.offset_to_days(off_cls, s)
        hour = s.fac_hour[r, f].argmax(1)
        bk = np.where(np.isnan(s.bkrel_med_f[r, f]), s.base["bkrel_med"], s.bkrel_med_f[r, f])
        nudge = Fe.nudge_minutes(s, off_days, hour, bk)
        return dict(fac=f, off=off_days, hour=hour, nudge=nudge, proba_fac=_onehot(f))


class PopularityBaseline:
    """Global prior: community's most popular facility / offset / hour / lead from training data."""

    def fit(self, s):
        m = s.meta
        self.f_ = int(m.y_fac.mode()[0])
        self.off_ = int(m.y_off.mode()[0])
        self.hour_ = int(m.y_hour[m.y_fac == self.f_].mode()[0])
        self.bkrel_ = float(np.median(m.y_bkrel))
        self.pf_ = np.bincount(m.y_fac, minlength=C.N_FAC) / len(m)
        return self

    def predict(self, s):
        n = len(s.meta)
        off_days = Fe.offset_to_days(np.full(n, self.off_), s)
        hour = np.full(n, self.hour_)
        return dict(fac=np.full(n, self.f_), off=off_days, hour=hour,
                    nudge=Fe.nudge_minutes(s, off_days, hour, np.full(n, self.bkrel_)),
                    proba_fac=np.tile(self.pf_, (n, 1)))


def _onehot(f):
    p = np.full((len(f), C.N_FAC), 0.02 / (C.N_FAC - 1))
    p[np.arange(len(f)), f] = 0.98
    return p
