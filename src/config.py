"""
Config for GDI prediction from DensePose IUV sequences
(CS229 Gotlin et al. data layout + improvements).
"""
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
# After you unzip CS229DP data.zip / clone, point this at densepose-out
DENSEPOSE_ROOT = DATA_DIR / "densepose-out"
# CSV from https://github.com/agotlin/CS229DP/tree/master/data
CSV_FULL = DATA_DIR / "imagepath_gdi.csv"
CSV_SMALL = DATA_DIR / "imagepath_gdi_10.csv"  # smaller subset for smoke tests

OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
FIG_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"
for d in (DATA_DIR, MODEL_DIR, FIG_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Sequence / image settings
# ---------------------------------------------------------------------------
SEQ_LEN = 10          # paper used windows of 10 frames
IMG_SIZE = 64         # resize IUV maps (paper used larger; 64 keeps Colab feasible)
IN_CHANNELS = 3       # I, U, V as 3 channels
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use small CSV for quick runs; set USE_FULL_CSV = True for full experiment
USE_FULL_CSV = False

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-4
WEIGHT_DECAY = 1e-5
DROPOUT = 0.3
CNN_CHANNELS = 32
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
PATIENCE = 8

# Split fractions by *patient folder* (video id), not by frame
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
