"""
Classical baselines on flattened mean IUV features (Linear Regression, Ridge).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import GDISequenceDataset, load_csv, build_sequences, patient_disjoint_split


def seq_to_vec(dataset: GDISequenceDataset) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        # mean over time of spatial mean → (C,) then flatten more: mean over T,H,W per channel
        v = x.mean(dim=(0, 2, 3)).numpy()  # C
        # also spatial std
        s = x.std(dim=(0, 2, 3)).numpy()
        Xs.append(np.concatenate([v, s]))
        ys.append(float(y))
    return np.stack(Xs), np.array(ys, dtype=np.float32)


def run_baselines():
    df = load_csv()
    seqs = build_sequences(df)
    tr, va, te = patient_disjoint_split(seqs)
    X_tr, y_tr = seq_to_vec(GDISequenceDataset(tr))
    X_te, y_te = seq_to_vec(GDISequenceDataset(te))

    results = {}
    for name, model in [
        ("LinearRegression", Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())])),
        ("Ridge", Pipeline([("sc", StandardScaler()), ("rg", Ridge(alpha=1.0))])),
    ]:
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
        results[name] = rmse
        print(f"{name}: test RMSE = {rmse:.4f}")
    return results


if __name__ == "__main__":
    run_baselines()
