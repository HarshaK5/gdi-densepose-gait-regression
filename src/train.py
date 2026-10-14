"""
Train GDI regression models (MSE) with optional multi-task aux classification.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DEVICE, EPOCHS, LR, MODEL_DIR, PATIENCE, RANDOM_SEED, WEIGHT_DECAY
from src.dataset import make_loaders
from src.models import build_model, gdi_to_bin


def set_seed(seed: int = RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_rmse(model, loader, device, multitask: bool = False):
    model.eval()
    se, n = 0.0, 0
    preds, targets = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out[0] if multitask else out
        se += torch.sum((pred - y) ** 2).item()
        n += y.numel()
        preds.append(pred.cpu().numpy())
        targets.append(y.cpu().numpy())
    rmse = float(np.sqrt(se / max(1, n)))
    return rmse, np.concatenate(preds), np.concatenate(targets)


def train_one(model_name: str = "attn", epochs: int = EPOCHS, lr: float = LR):
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, _ = make_loaders()
    model = build_model(model_name).to(device)
    multitask = model_name.lower() in ("multitask", "mt")
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    best_val, best_state, wait = 1e9, None, 0
    history = {"train_rmse": [], "val_rmse": [], "train_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, se, n = 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            if multitask:
                pred, logits = model(x)
                loss = mse(pred, y) + 0.3 * ce(logits, gdi_to_bin(y))
            else:
                pred = model(x)
                loss = mse(pred, y)
            loss.backward()
            optim.step()
            loss_sum += loss.item() * y.size(0)
            se += torch.sum((pred.detach() - y) ** 2).item()
            n += y.numel()

        train_rmse = float(np.sqrt(se / max(1, n)))
        train_loss = loss_sum / max(1, n)
        val_rmse, _, _ = eval_rmse(model, val_loader, device, multitask)
        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_rmse)
        history["train_loss"].append(train_loss)
        print(
            f"Epoch {epoch:03d}/{epochs}  loss={train_loss:.4f}  "
            f"train_RMSE={train_rmse:.3f}  val_RMSE={val_rmse:.3f}"
        )

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_rmse, preds, targets = eval_rmse(model, test_loader, device, multitask)
    print(f"\nTest RMSE = {test_rmse:.4f}  (paper best ~4.4 on their split/data)")

    ckpt = MODEL_DIR / f"{model_name}_best.pt"
    torch.save(
        {
            "model_name": model_name,
            "state_dict": model.state_dict(),
            "history": history,
            "test_rmse": test_rmse,
            "best_val_rmse": best_val,
        },
        ckpt,
    )
    with open(MODEL_DIR / f"{model_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved {ckpt}")
    return {
        "model": model,
        "test_rmse": test_rmse,
        "preds": preds,
        "targets": targets,
        "history": history,
        "device": device,
        "multitask": multitask,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="attn", choices=["lstm", "attn", "multitask"])
    p.add_argument("--epochs", type=int, default=EPOCHS)
    args = p.parse_args()
    train_one(model_name=args.model, epochs=args.epochs)
