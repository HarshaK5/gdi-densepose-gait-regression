"""
Spatiotemporal models for GDI regression from DensePose IUV sequences.

Paper baseline spirit: CNN per frame + LSTM → scalar GDI.
Improvements:
  - Residual CNN encoder
  - Bidirectional LSTM
  - Temporal attention pooling (novelty)
  - Optional multi-task head (GDI + coarse severity bin)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.config import CNN_CHANNELS, DROPOUT, IN_CHANNELS, LSTM_HIDDEN, LSTM_LAYERS


class FrameCNN(nn.Module):
    """Small CNN encoder for one IUV frame (C,H,W) → feature vector."""

    def __init__(self, in_ch: int = IN_CHANNELS, base: int = CNN_CHANNELS, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base, base * 2, 3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base * 2, base * 4, 3, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(base * 4, out_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        h = self.net(x).flatten(1)
        return self.fc(h)


class CNNLSTM(nn.Module):
    """Paper-style: CNN → LSTM → last step → GDI (regression)."""

    def __init__(
        self,
        hidden: int = LSTM_HIDDEN,
        num_layers: int = LSTM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.encoder = FrameCNN(out_dim=hidden)
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        feats = self.encoder(x.reshape(B * T, C, H, W)).reshape(B, T, -1)
        out, _ = self.lstm(feats)
        return self.head(out[:, -1, :]).squeeze(-1)


class CNNLSTMAttn(nn.Module):
    """Improvement: temporal attention over BiLSTM outputs → GDI."""

    def __init__(
        self,
        hidden: int = LSTM_HIDDEN,
        num_layers: int = LSTM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.encoder = FrameCNN(out_dim=hidden)
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attn = nn.Linear(hidden * 2, 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, return_attn: bool = False):
        B, T, C, H, W = x.shape
        feats = self.encoder(x.reshape(B * T, C, H, W)).reshape(B, T, -1)
        out, _ = self.lstm(feats)
        scores = self.attn(out).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        ctx = torch.sum(out * weights.unsqueeze(-1), dim=1)
        pred = self.head(ctx).squeeze(-1)
        if return_attn:
            return pred, weights
        return pred


class CNNLSTMMultiTask(nn.Module):
    """
    Improvement: predict continuous GDI + coarse severity bin (multi-task).
    Severity bins: GDI < 70 severe-ish, 70–90 moderate, >= 90 closer to typical.
    """

    def __init__(
        self,
        hidden: int = LSTM_HIDDEN,
        num_layers: int = LSTM_LAYERS,
        dropout: float = DROPOUT,
        n_bins: int = 3,
    ):
        super().__init__()
        self.encoder = FrameCNN(out_dim=hidden)
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attn = nn.Linear(hidden * 2, 1)
        self.reg_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.cls_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_bins),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats = self.encoder(x.reshape(B * T, C, H, W)).reshape(B, T, -1)
        out, _ = self.lstm(feats)
        weights = torch.softmax(self.attn(out).squeeze(-1), dim=-1)
        ctx = torch.sum(out * weights.unsqueeze(-1), dim=1)
        gdi = self.reg_head(ctx).squeeze(-1)
        logits = self.cls_head(ctx)
        return gdi, logits


def gdi_to_bin(gdi: torch.Tensor) -> torch.Tensor:
    """Map GDI to 0/1/2 bins for multi-task aux loss."""
    bins = torch.zeros_like(gdi, dtype=torch.long)
    bins = torch.where(gdi >= 70, torch.ones_like(bins), bins)
    bins = torch.where(gdi >= 90, torch.full_like(bins, 2), bins)
    return bins


def build_model(name: str) -> nn.Module:
    name = name.lower()
    if name in ("lstm", "cnnlstm", "baseline"):
        return CNNLSTM()
    if name in ("attn", "attention", "cnnlstmattn"):
        return CNNLSTMAttn()
    if name in ("multitask", "mt"):
        return CNNLSTMMultiTask()
    raise ValueError(f"Unknown model: {name}")
