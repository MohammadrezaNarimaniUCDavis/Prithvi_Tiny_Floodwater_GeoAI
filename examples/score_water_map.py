"""Score a predicted water mask against a Sen1Floods11-style reference.

Usage:
    python examples/score_water_map.py --pred pred.npy --label label.npy
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from floodwater.metrics import confusion, scores


def main() -> None:
    p = argparse.ArgumentParser(description="Compute water IoU / F1 on valid pixels.")
    p.add_argument("--pred", required=True, help="uint8/int array with 0=land, 1=water")
    p.add_argument("--label", required=True, help="int array with -1=unlabeled, 0=land, 1=water")
    args = p.parse_args()
    pred = np.load(args.pred)
    label = np.load(args.label)
    c = confusion(pred, label)
    print(json.dumps({**c, **scores(c)}, indent=2))


if __name__ == "__main__":
    main()
