"""Chronological train / validation / test split on the target booking's creation time."""
from __future__ import annotations

from . import config as C
from . import features as Fe


def split_states(s: Fe.State) -> dict[str, Fe.State]:
    bt = Fe.from_min(s.meta.y_book.to_numpy())
    return {
        "train": Fe.subset(s, bt < C.VAL_START),
        "val": Fe.subset(s, (bt >= C.VAL_START) & (bt < C.TEST_START)),
        "test": Fe.subset(s, bt >= C.TEST_START),
        "train_val": Fe.subset(s, bt < C.TEST_START),
    }
