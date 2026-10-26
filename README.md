# Prithvi-Tiny Floodwater GeoAI

Public Python helpers for **compact foundation-model floodwater segmentation** on the hand-labeled Sentinel-2 chips of [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) v1.1.

> Farajpoor, P., & Narimani, M. (2026). *Accuracy and efficiency of compact foundation models for surface-water mapping during floods.* University of California, Davis.

**Authors:** Parastoo Farajpoor and Mohammadreza Narimani  
**Affiliation:** Department of Biological and Agricultural Engineering, UC Davis  
**Contact:** [mnarimani@ucdavis.edu](mailto:mnarimani@ucdavis.edu)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23653544.svg)](https://doi.org/10.5281/zenodo.23653544)

| Resource | Link |
| -------- | ---- |
| **Derived products (Zenodo)** | https://doi.org/10.5281/zenodo.23653544 |
| **This code** | https://github.com/MohammadrezaNarimaniUCDavis/Prithvi_Tiny_Floodwater_GeoAI |
| **Sen1Floods11 v1.1** | https://github.com/cloudtostreet/Sen1Floods11 |
| **Prithvi-EO-2.0** | https://huggingface.co/ibm-nasa-geospatial |

This repository is a **code-only public package**. It does **not** include the manuscript, figures, GeoTIFFs, or trained checkpoints. Numerical tables and archived code are on Zenodo. Source imagery and pretrained weights remain with their original providers.

If you use this work, please cite the paper (preferred once posted), plus the Zenodo record and/or this repository.

---

## What this study does

1. Fine-tune **Prithvi-EO-2.0-Tiny** for binary surface-water segmentation on 446 hand-labeled Sentinel-2 chips from 11 flood events.
2. Compare a frozen encoder, a fully fine-tuned encoder, and an identical randomly initialized transformer with a size-matched **U-Net**, a larger Prithvi-100M reference, and **NDWI / MNDWI** thresholds.
3. Measure **label-budget** sensitivity, **event-excluded** fitting, GPU/CPU latency, and probability **calibration / entropy**.
4. Keep the scientific claim specific: pretraining helps the compact transformer relative to random initialization, while U-Net remains the stronger seen-event overlap score and Tiny is faster under the tested implementation.

Claims are framed as **agreement with visible reference water**, not independently verified new inundation and not operational flood-loss reduction.

### Headline numbers (Protocol A, all labels, three seeds)

| Setting | Fine-tuned Tiny | U-Net |
| ------- | --------------- | ----- |
| Seen-event water IoU | **0.833** | **0.856** |
| Bolivia holdout IoU | 0.752 | 0.795 |
| GPU latency (Tesla T4, ms / 512×512 chip) | **16.1** | 30.6 |
| CPU latency (i9-12900H, ms / chip) | **258** | 665 |
| Event-average IoU (Protocol B, 11 events) | 0.768 | 0.777 |

---

## Repository layout

```text
Prithvi_Tiny_Floodwater_GeoAI/
|-- floodwater/
|   |-- constants.py     # Sentinel-2 / Prithvi band maps
|   |-- metrics.py       # IoU, F1, ECE, binary entropy
|   |-- indices.py       # NDWI, MNDWI, Otsu, fitted threshold
|   |-- splits.py        # event-stratified nested subsets, Protocol B folds
|   |-- bootstrap.py     # chip-cluster bootstrap for pooled IoU
|   `-- models.py        # U-Net and TerraTorch Prithvi factory
|-- examples/
|   `-- score_water_map.py
|-- pyproject.toml
|-- requirements.txt
|-- CITATION.cff
`-- LICENSE
```

---

## Install

```bash
pip install -e .
```

PyTorch is optional for metric / index / split helpers. Prithvi models additionally need [TerraTorch](https://github.com/IBM/terratorch) `>=1.2`:

```bash
pip install -e ".[prithvi]"
```

---

## Minimal examples

Water IoU on valid pixels only (`label == -1` is ignored):

```python
import numpy as np
from floodwater.metrics import confusion, scores

pred = np.load("pred.npy")    # 0/1
label = np.load("label.npy")  # -1/0/1
print(scores(confusion(pred, label)))
```

NDWI with a training-fitted threshold:

```python
from floodwater.indices import fit_threshold, water_index

idx = [water_index(chip, "NDWI") for chip in train_chips]
t = fit_threshold(idx, train_labels)
water = water_index(test_chip, "NDWI") > t
```

Event-stratified nested label subsets:

```python
from floodwater.splits import fraction_subsets

subsets = fraction_subsets(train_chip_table, seed=0)
print(len(subsets["0.10"]), len(subsets["1.00"]))
```

---

## Data (not in this repository)

- **Imagery and labels:** Sen1Floods11 v1.1 hand-labeled Sentinel-2 chips (Bonafilia et al., 2020).
- **Pretrained encoder:** NASA–IBM Prithvi-EO-2.0 Tiny / 100M (Szwarcman et al., 2026).
- **Derived tables, split manifests, and a 11-chip sample** (true-colour excerpts, label previews, predicted masks, probability/error GeoTIFFs): Zenodo record linked above. The original Sen1Floods11 GeoTIFF archive is not re-hosted; get it from `gs://sen1floods11`.

---

## Citation

```bibtex
@misc{farajpoor2026floodwater,
  author = {Farajpoor, Parastoo and Narimani, Mohammadreza},
  title  = {Accuracy and efficiency of compact foundation models for surface-water mapping during floods},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/MohammadrezaNarimaniUCDavis/Prithvi_Tiny_Floodwater_GeoAI}
}
```

---

## License

MIT. Source datasets and pretrained weights keep their original licenses.
