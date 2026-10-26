"""Event-stratified chip-cluster bootstrap for pooled water IoU."""
from __future__ import annotations

import numpy as np


def iou_from_counts(tp: int, fp: int, fn: int) -> float:
    den = tp + fp + fn
    return float(tp) / float(den) if den > 0 else float("nan")


def cluster_bootstrap_iou(
    chip_counts: list[dict],
    events: list[str],
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Resample chips with replacement within each event, then pool confusion counts.

    chip_counts: list of dicts with keys tp, fp, fn.
    events: event id for each chip, same length as chip_counts.
    Returns (point IoU from all chips, 2.5th percentile, 97.5th percentile).
    """
    if len(chip_counts) != len(events):
        raise ValueError("chip_counts and events must have the same length")
    tp = np.array([c["tp"] for c in chip_counts], dtype=np.int64)
    fp = np.array([c["fp"] for c in chip_counts], dtype=np.int64)
    fn = np.array([c["fn"] for c in chip_counts], dtype=np.int64)
    ev = np.asarray(events)
    point = iou_from_counts(int(tp.sum()), int(fp.sum()), int(fn.sum()))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        tps = fps = fns = 0
        for e in np.unique(ev):
            idx = np.flatnonzero(ev == e)
            draw = rng.choice(idx, size=len(idx), replace=True)
            tps += int(tp[draw].sum())
            fps += int(fp[draw].sum())
            fns += int(fn[draw].sum())
        stats[i] = iou_from_counts(tps, fps, fns)
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return point, float(lo), float(hi)
