"""
Load DensePose IUV PNG sequences + GDI labels from CS229DP CSVs.

CSV columns: image_path, labels (GDI float), label_norm (optional bin)
Paths look like: ./data/densepose-out/07337701-processed/000053_IUV.png

We group by patient folder (e.g. 07337701-processed), build fixed-length
windows, and split by patient so no video leaks into test.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (
    BATCH_SIZE,
    CSV_FULL,
    CSV_SMALL,
    DENSEPOSE_ROOT,
    IMG_SIZE,
    RANDOM_SEED,
    SEQ_LEN,
    TRAIN_FRAC,
    USE_FULL_CSV,
    VAL_FRAC,
)


def _patient_id(path_str: str) -> str:
    """Extract folder name like '07337701-processed' from a path."""
    parts = Path(str(path_str).replace("\\", "/")).parts
    for p in parts:
        if re.search(r"\d{5,}-processed", p):
            return p
    # fallback: parent folder name
    return Path(str(path_str).replace("\\", "/")).parent.name


def _resolve_image(path_str: str) -> Path | None:
    """Map CSV relative path to a local file under DATA_DIR / densepose-out."""
    rel = str(path_str).replace("\\", "/")
    # strip leading ./data/densepose-out/ or data/densepose-out/
    for prefix in (
        "./data/densepose-out/",
        "data/densepose-out/",
        "./data/densepose-out\\",
    ):
        if rel.startswith(prefix.replace("\\", "/")):
            rel = rel[len(prefix.replace("\\", "/")) :]
            break
    candidate = DENSEPOSE_ROOT / rel
    if candidate.exists():
        return candidate
    # try just filename under patient folder
    pid = _patient_id(path_str)
    name = Path(rel).name
    alt = DENSEPOSE_ROOT / pid / name
    if alt.exists():
        return alt
    return None


def load_csv(use_full: bool | None = None) -> pd.DataFrame:
    use_full = USE_FULL_CSV if use_full is None else use_full
    path = CSV_FULL if use_full else CSV_SMALL
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Download from:\n"
            "  https://github.com/agotlin/CS229DP/tree/master/data\n"
            "Place imagepath_gdi.csv and/or imagepath_gdi_10.csv in data/\n"
            "Unzip data.zip so PNG files live under data/densepose-out/"
        )
    df = pd.read_csv(path)
    # normalise column names
    cols = {c.lower().strip(): c for c in df.columns}
    path_col = cols.get("image_path") or cols.get("imagepath")
    label_col = cols.get("labels") or cols.get("label")
    if path_col is None or label_col is None:
        raise KeyError(f"Need image_path and labels columns; got {list(df.columns)}")
    df = df.rename(columns={path_col: "image_path", label_col: "gdi"})
    df["patient"] = df["image_path"].map(_patient_id)
    df["gdi"] = df["gdi"].astype(np.float32)
    return df


def build_sequences(df: pd.DataFrame, seq_len: int = SEQ_LEN):
    """
    Returns list of dicts:
      {patient, paths: [Path,...], gdi: float}
    Only sequences where every frame file exists on disk.
    """
    sequences = []
    missing = 0
    for patient, g in df.groupby("patient", sort=False):
        g = g.sort_values("image_path")
        paths, gdis = [], []
        for _, row in g.iterrows():
            p = _resolve_image(row["image_path"])
            if p is None:
                missing += 1
                continue
            paths.append(p)
            gdis.append(float(row["gdi"]))
        if len(paths) < seq_len:
            continue
        # non-overlapping windows; GDI is constant per patient in this data
        gdi = float(np.median(gdis))
        for i in range(0, len(paths) - seq_len + 1, seq_len):
            sequences.append(
                {
                    "patient": patient,
                    "paths": paths[i : i + seq_len],
                    "gdi": gdi,
                }
            )
    print(
        f"Built {len(sequences)} sequences from {df['patient'].nunique()} patients "
        f"({missing} missing image refs skipped)"
    )
    if len(sequences) == 0:
        raise RuntimeError(
            "No sequences found. Unzip CS229DP data.zip into data/densepose-out/ "
            "so paths like data/densepose-out/<id>-processed/*_IUV.png exist."
        )
    return sequences


def patient_disjoint_split(sequences, seed: int = RANDOM_SEED):
    rng = np.random.default_rng(seed)
    patients = sorted({s["patient"] for s in sequences})
    rng.shuffle(patients)
    n = len(patients)
    n_tr = max(1, int(TRAIN_FRAC * n))
    n_va = max(1, int(VAL_FRAC * n)) if n > 2 else 0
    tr_p = set(patients[:n_tr])
    va_p = set(patients[n_tr : n_tr + n_va])
    te_p = set(patients[n_tr + n_va :])

    def take(ps):
        return [s for s in sequences if s["patient"] in ps]

    return take(tr_p), take(va_p), take(te_p)


def load_iuv_stack(paths: list[Path], size: int = IMG_SIZE) -> np.ndarray:
    """Return (T, C, H, W) float32 in [0, 1]."""
    frames = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        im = im.resize((size, size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0  # H,W,3
        frames.append(arr.transpose(2, 0, 1))  # 3,H,W
    return np.stack(frames, axis=0)


class GDISequenceDataset(Dataset):
    def __init__(self, sequences: list[dict], size: int = IMG_SIZE):
        self.sequences = sequences
        self.size = size

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        s = self.sequences[idx]
        x = load_iuv_stack(s["paths"], self.size)  # T,C,H,W
        y = np.float32(s["gdi"])
        return torch.from_numpy(x), torch.tensor(y)


def make_loaders(batch_size: int = BATCH_SIZE, use_full: bool | None = None):
    df = load_csv(use_full=use_full)
    seqs = build_sequences(df)
    tr, va, te = patient_disjoint_split(seqs)
    print(f"Split sequences — train {len(tr)}, val {len(va)}, test {len(te)}")
    train_loader = DataLoader(
        GDISequenceDataset(tr), batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        GDISequenceDataset(va), batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        GDISequenceDataset(te), batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader, test_loader, (tr, va, te)
