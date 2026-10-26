"""Event-stratified nested label subsets and leave-one-event-out fold construction."""
from __future__ import annotations

import numpy as np
import pandas as pd

FRACTIONS = [0.10, 0.25, 0.50, 1.00]
SEEDS = [0, 1, 2]


def stratified_order(chips: pd.DataFrame, seed: int) -> list[str]:
    """Deterministic order whose prefixes remain approximately event-proportional.

    Chips are shuffled within each event. The k-th chip of an event with n chips
    receives the key (k + u) / n with u ~ U(0, 1). Sorting interleaves events so
    any prefix is an event-stratified random subset and prefixes are nested.
    """
    rng = np.random.default_rng(seed)
    keys = []
    for _ev, grp in chips.groupby("event", sort=True):
        ids = grp.chip_id.to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        u = rng.random(n)
        keys += [((k + u[k]) / n, cid) for k, cid in enumerate(ids)]
    keys.sort()
    return [cid for _, cid in keys]


def fraction_subsets(train_chips: pd.DataFrame, seed: int, fractions=FRACTIONS) -> dict[str, list[str]]:
    order = stratified_order(train_chips, seed)
    n = len(order)
    out = {}
    for f in fractions:
        k = max(1, int(round(f * n)))
        out[f"{f:.2f}"] = sorted(order[:k])
    return out


def protocol_b_fold(chips: pd.DataFrame, held_out_event: str) -> dict:
    """Train/valid = official train/valid of remaining events; test = all chips of the target."""
    rest = chips[chips.event != held_out_event]
    return {
        "protocol": "B",
        "fold": held_out_event,
        "train_pool": sorted(rest[rest.official_split == "train"].chip_id),
        "valid": sorted(rest[rest.official_split == "valid"].chip_id),
        "test": sorted(chips[chips.event == held_out_event].chip_id),
    }


def disjoint(train, valid, test) -> bool:
    tr, va, te = set(train), set(valid), set(test)
    return not (tr & va) and not (tr & te) and not (va & te)
