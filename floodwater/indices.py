"""NDWI / MNDWI water-index baselines (McFeeters 1996; Xu 2006) and Otsu (1979)."""
from __future__ import annotations

import numpy as np

from .constants import I_GREEN, I_NIR, I_SWIR1, REFLECTANCE_SCALE

GRID = np.round(np.arange(-1.0, 1.0 + 1e-9, 0.005), 3)


def water_index(x_dn: np.ndarray, kind: str) -> np.ndarray:
    """kind is NDWI (green, B8A) or MNDWI (green, B11). x_dn is a 6-band DN stack."""
    x = np.clip(x_dn.astype(np.float32), 0, 10000) * REFLECTANCE_SCALE
    a = x[I_GREEN]
    b = x[I_NIR] if kind.upper() == "NDWI" else x[I_SWIR1]
    with np.errstate(invalid="ignore", divide="ignore"):
        idx = (a - b) / (a + b)
    return np.where(np.isfinite(idx), idx, -1.0)


def otsu_threshold(values: np.ndarray, nbins: int = 256) -> float:
    if values.size < 10:
        return 0.0
    hist, edges = np.histogram(values, bins=nbins, range=(-1, 1))
    centers = (edges[:-1] + edges[1:]) / 2
    w = hist.astype(np.float64)
    w0 = np.cumsum(w)
    w1 = w0[-1] - w0
    m0 = np.cumsum(w * centers)
    mt = m0[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        mu0 = m0 / w0
        mu1 = (mt - m0) / w1
        between = w0 * w1 * (mu0 - mu1) ** 2
    between[~np.isfinite(between)] = -1
    return float(centers[int(np.argmax(between))])


def fit_threshold(idx_list, lab_list) -> float:
    """Single global threshold maximising aggregated water IoU on a 0.005 grid."""
    edges = np.concatenate([GRID - 0.0025, [GRID[-1] + 0.0025]])
    hw = np.zeros(len(GRID))
    hn = np.zeros(len(GRID))
    for idx, lab in zip(idx_list, lab_list):
        v = lab >= 0
        hw += np.histogram(np.clip(idx[v & (lab == 1)], -1, 1), bins=edges)[0]
        hn += np.histogram(np.clip(idx[v & (lab == 0)], -1, 1), bins=edges)[0]
    tp = hw[::-1].cumsum()[::-1]
    fp = hn[::-1].cumsum()[::-1]
    fn = hw.sum() - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    k = int(np.argmax(iou))
    return round(float(GRID[k] - 0.0025), 4)
