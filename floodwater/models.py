"""U-Net (scratch) and Prithvi-EO-2.0 segmentation factory via TerraTorch.

All Prithvi variants share one UPerNet head so that only encoder initialisation
and trainability differ. TerraTorch is required only for the Prithvi models.
"""
from __future__ import annotations

import hashlib

import torch
import torch.nn as nn

from .constants import PRITHVI_BAND_LABELS

MODEL_SPECS = {
    "unet": {
        "family": "cnn",
        "pretrained": False,
        "freeze_encoder": False,
        "backbone": None,
        "label": "U-Net (scratch)",
    },
    "prithvi_tiny_frozen": {
        "family": "prithvi",
        "pretrained": True,
        "freeze_encoder": True,
        "backbone": "prithvi_eo_v2_tiny_tl",
        "label": "Prithvi-Tiny frozen",
    },
    "prithvi_tiny_ft": {
        "family": "prithvi",
        "pretrained": True,
        "freeze_encoder": False,
        "backbone": "prithvi_eo_v2_tiny_tl",
        "label": "Prithvi-Tiny fine-tuned",
    },
    "prithvi_tiny_random": {
        "family": "prithvi",
        "pretrained": False,
        "freeze_encoder": False,
        "backbone": "prithvi_eo_v2_tiny_tl",
        "label": "Prithvi-Tiny random init",
    },
    "prithvi_100m_ft": {
        "family": "prithvi",
        "pretrained": True,
        "freeze_encoder": False,
        "backbone": "prithvi_eo_v2_100_tl",
        "label": "Prithvi-100M fine-tuned",
    },
}


class DoubleConv(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """Ronneberger et al. (2015) U-Net with batch normalisation; base width 32 (~7.8 M parameters)."""

    def __init__(self, in_ch: int = 6, n_classes: int = 2, base: int = 32, depth: int = 4):
        super().__init__()
        ch = [base * 2**i for i in range(depth + 1)]
        self.inc = DoubleConv(in_ch, ch[0])
        self.down = nn.ModuleList(
            [nn.Sequential(nn.MaxPool2d(2), DoubleConv(ch[i], ch[i + 1])) for i in range(depth)]
        )
        self.up = nn.ModuleList(
            [nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))]
        )
        self.dec = nn.ModuleList([DoubleConv(ch[i] * 2, ch[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(ch[0], n_classes, 1)

    def forward(self, x):
        skips = [self.inc(x)]
        for d in self.down:
            skips.append(d(skips[-1]))
        h = skips.pop()
        for up, dec in zip(self.up, self.dec):
            h = up(h)
            h = dec(torch.cat([skips.pop(), h], dim=1))
        return self.head(h)


class PrithviSegmenter(nn.Module):
    def __init__(self, backbone: str, pretrained: bool, freeze_encoder: bool, decoder_channels: int = 128):
        super().__init__()
        from terratorch.models import EncoderDecoderFactory

        self.net = EncoderDecoderFactory().build_model(
            task="segmentation",
            backbone=backbone,
            backbone_pretrained=pretrained,
            backbone_bands=PRITHVI_BAND_LABELS,
            backbone_num_frames=1,
            necks=[
                {"name": "SelectIndices", "indices": [2, 5, 8, 11]},
                {"name": "ReshapeTokensToImage"},
                {"name": "LearnedInterpolateToPyramidal"},
            ],
            decoder="UperNetDecoder",
            decoder_channels=decoder_channels,
            num_classes=2,
            rescale=True,
            head_dropout=0.1,
        )
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.net.encoder.parameters():
                p.requires_grad = False

    @property
    def encoder(self):
        return self.net.encoder

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.net.encoder.eval()
        return self

    def forward(self, x):
        return self.net(x).output


def build_model(name: str) -> nn.Module:
    spec = MODEL_SPECS[name]
    if spec["family"] == "cnn":
        return UNet()
    return PrithviSegmenter(spec["backbone"], spec["pretrained"], spec["freeze_encoder"])


def param_groups(model: nn.Module, name: str, base_lr: float, backbone_lr_mult: float, weight_decay: float):
    spec = MODEL_SPECS[name]
    if spec["family"] == "prithvi" and spec["pretrained"] and not spec["freeze_encoder"]:
        enc = {id(p) for p in model.encoder.parameters()}
        g_enc = [p for p in model.parameters() if id(p) in enc and p.requires_grad]
        g_new = [p for p in model.parameters() if id(p) not in enc and p.requires_grad]
        return [
            {"params": g_enc, "lr": base_lr * backbone_lr_mult, "weight_decay": weight_decay, "name": "encoder"},
            {"params": g_new, "lr": base_lr, "weight_decay": weight_decay, "name": "decoder_head"},
        ]
    return [
        {
            "params": [p for p in model.parameters() if p.requires_grad],
            "lr": base_lr,
            "weight_decay": weight_decay,
            "name": "all_trainable",
        }
    ]


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    enc = sum(p.numel() for p in model.encoder.parameters()) if hasattr(model, "encoder") else 0
    return {
        "params_total": total,
        "params_trainable": trainable,
        "params_encoder": enc,
        "params_decoder_head": total - enc,
    }


def encoder_fingerprint(model: nn.Module) -> str:
    if not hasattr(model, "encoder"):
        return ""
    h = hashlib.sha256()
    for k, v in sorted(model.encoder.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
