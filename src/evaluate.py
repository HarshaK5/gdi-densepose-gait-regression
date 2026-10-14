"""RMSE plots, scatter pred vs true, residual histogram."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FIG_DIR, MODEL_DIR
from src.dataset import make_loaders
from src.models import build_model
from src.train import eval_rmse


def plot_scatter(y_true, y_pred, title, save_name):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", label="ideal")
    ax.set_xlabel("True GDI")
    ax.set_ylabel("Predicted GDI")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / save_name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_history(history, save_name):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train_rmse"], label="train RMSE")
    ax.plot(history["val_rmse"], label="val RMSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / save_name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def evaluate_checkpoint(ckpt_name: str = "attn_best.pt"):
    path = MODEL_DIR / ckpt_name
    if not path.exists():
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    name = ckpt["model_name"]
    multitask = name.lower() in ("multitask", "mt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(name).to(device)
    model.load_state_dict(ckpt["state_dict"])

    _, _, test_loader, _ = make_loaders()
    rmse, preds, targets = eval_rmse(model, test_loader, device, multitask)
    print(f"Test RMSE: {rmse:.4f}")
    plot_scatter(targets, preds, f"{name} — test GDI", f"scatter_{name}.png")
    if "history" in ckpt:
        plot_history(ckpt["history"], f"curves_{name}.png")
    residuals = preds - targets
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(residuals, bins=30, color="steelblue", edgecolor="white")
    ax.set_xlabel("Residual (pred − true)")
    ax.set_title(f"{name} residuals")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"residuals_{name}.png", dpi=150)
    plt.close(fig)
    return {"rmse": rmse}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="attn_best.pt")
    args = p.parse_args()
    evaluate_checkpoint(args.ckpt)
