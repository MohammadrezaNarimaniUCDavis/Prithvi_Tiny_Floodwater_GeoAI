"""Water-class segmentation and calibration metrics on valid (label != -1) pixels."""
from __future__ import annotations

import numpy as np

N_CAL_BINS = 15


def confusion(pred: np.ndarray, label: np.ndarray) -> dict:
    """pred in {0, 1}; label in {-1, 0, 1}. Counts are over valid pixels only."""
    v = label >= 0
    p = pred[v].astype(bool)
    y = label[v].astype(bool)
    tp = int(np.count_nonzero(p & y))
    fp = int(np.count_nonzero(p & ~y))
    fn = int(np.count_nonzero(~p & y))
    tn = int(np.count_nonzero(~p & ~y))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _div(a, b):
    return float(a) / float(b) if b > 0 else float("nan")


def scores(c: dict) -> dict:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    rec = _div(tp, tp + fn)
    tnr = _div(tn, tn + fp)
    return {
        "iou_water": _div(tp, tp + fp + fn),
        "iou_nonwater": _div(tn, tn + fp + fn),
        "f1_water": _div(2 * tp, 2 * tp + fp + fn),
        "precision_water": _div(tp, tp + fp),
        "recall_water": rec,
        "specificity": tnr,
        "balanced_accuracy": (rec + tnr) / 2 if not (np.isnan(rec) or np.isnan(tnr)) else float("nan"),
        "overall_accuracy": _div(tp + tn, tp + fp + fn + tn),
        "n_valid": tp + fp + fn + tn,
    }


def add(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in ("tp", "fp", "fn", "tn")}


def calibration_accumulator() -> dict:
    return {
        "count": np.zeros(N_CAL_BINS, np.int64),
        "sum_p": np.zeros(N_CAL_BINS),
        "sum_y": np.zeros(N_CAL_BINS),
        "brier_sum": 0.0,
        "nll_sum": 0.0,
        "n": 0,
    }


def update_calibration(acc: dict, p_water: np.ndarray, label: np.ndarray) -> dict:
    v = label >= 0
    p = np.clip(p_water[v].astype(np.float64), 1e-7, 1 - 1e-7)
    y = label[v].astype(np.float64)
    b = np.minimum((p * N_CAL_BINS).astype(int), N_CAL_BINS - 1)
    acc["count"] += np.bincount(b, minlength=N_CAL_BINS)
    acc["sum_p"] += np.bincount(b, weights=p, minlength=N_CAL_BINS)
    acc["sum_y"] += np.bincount(b, weights=y, minlength=N_CAL_BINS)
    acc["brier_sum"] += float(((p - y) ** 2).sum())
    acc["nll_sum"] += float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).sum())
    acc["n"] += int(v.sum())
    return acc


def calibration_summary(acc: dict) -> dict:
    """Expected calibration error (equal-width bins), Brier score, and NLL."""
    n = max(acc["n"], 1)
    cnt = acc["count"]
    with np.errstate(invalid="ignore", divide="ignore"):
        conf = np.where(cnt > 0, acc["sum_p"] / cnt, np.nan)
        freq = np.where(cnt > 0, acc["sum_y"] / cnt, np.nan)
    ece = float(np.nansum(cnt / n * np.abs(conf - freq)))
    return {
        "ece": ece,
        "brier": acc["brier_sum"] / n,
        "nll": acc["nll_sum"] / n,
        "n_pixels": acc["n"],
        "bin_count": cnt.tolist(),
        "bin_mean_prob": np.nan_to_num(conf, nan=-1).tolist(),
        "bin_water_freq": np.nan_to_num(freq, nan=-1).tolist(),
    }


def entropy_binary(p_water: np.ndarray) -> np.ndarray:
    p = np.clip(p_water.astype(np.float64), 1e-7, 1 - 1e-7)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
