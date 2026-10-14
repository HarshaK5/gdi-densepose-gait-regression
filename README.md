# GDI Prediction from DensePose Gait Sequences

Regression of the **Gait Deviation Index (GDI)**, a clinical score of how far a person's gait is from typical gait, from sequences of **DensePose IUV** frames. The project re-implements the pipeline of the CS229DP project (Gotlin et al., [agotlin/CS229DP](https://github.com/agotlin/CS229DP)) and extends it with temporal attention, multi-task learning, patient-disjoint splits, and classical baselines.

> **Status:** the full pipeline is implemented and runs end to end up to data loading. The DensePose IUV images are **not publicly available**: the `data.zip` in the CS229DP repository contains only the two CSV files. Because of this, no models have been trained here and **no results are reported**. The results table in the slides is intentionally left blank. If you have the IUV images, the instructions below will run the full experiment.

---

## 1. Problem

- **Input:** a window of `T = 10` consecutive DensePose IUV frames (3 channels: body part index I, and surface coordinates U, V) for one walking video.
- **Output:** a single continuous GDI value (typical gait is around 100, lower means more deviation).
- **Metric:** RMSE (and a scatter plot of predicted vs true GDI).

## 2. What is implemented

### Reproduced from the reference project
- IUV frame loading and fixed-length windows of 10 frames.
- A CNN (per-frame feature extractor) followed by an LSTM over time, with a regression head for GDI.
- MSE training loss, reported as RMSE.

### Added in this repository
| Addition | What it does | File |
|---|---|---|
| **Patient-disjoint splits** | Train/val/test (70/15/15) are split by patient folder, not by frame or window, so frames of one person never appear in both training and test. This avoids data leakage that makes random splits look better than they are. | `src/dataset.py` |
| **Bidirectional LSTM** | BiLSTM (2 layers, 128 hidden units) reads the sequence in both directions. | `src/models.py` |
| **Temporal attention** | Learns a weight per frame and pools the sequence, instead of using only the last hidden state. | `src/models.py` |
| **Multi-task head** | Predicts GDI (regression) and a severity bin (classification) together; the auxiliary task regularizes the shared features. | `src/models.py`, `src/train.py` |
| **BatchNorm, dropout, early stopping** | BatchNorm in the CNN, dropout 0.3, early stopping with patience 8 and weight decay. | `src/models.py`, `src/train.py` |
| **Classical baselines** | Linear regression and Ridge regression on flattened features, as a sanity-check reference for the deep models. | `src/baselines.py` |
| **One-command experiment runner** | Runs baselines and all deep model variants and saves metrics and figures. | `src/run_experiments.py` |
| **Evaluation and plots** | RMSE, predicted-vs-true scatter plots, and training curves. | `src/evaluate.py` |
| **Reproducible config** | Seeds and all hyperparameters in one file; CPU/GPU chosen automatically. | `src/config.py` |

### Model variants compared
1. CNN + BiLSTM
2. CNN + BiLSTM + temporal attention
3. CNN + BiLSTM + attention + multi-task

## 3. Repository structure

```
gait-gdi-cs229/
├── data/
│   ├── imagepath_gdi.csv        # full list of IUV image paths and GDI labels
│   └── imagepath_gdi_10.csv     # smaller subset for quick runs
├── outputs/
│   ├── figures/                 # plots (scatter, curves)
│   ├── logs/                    # training logs
│   └── models/                  # saved weights (not committed)
├── src/
│   ├── config.py                # paths and hyperparameters
│   ├── setup_data.py            # downloads the CSV files
│   ├── dataset.py               # sequence building, patient-disjoint split, loaders
│   ├── models.py                # CNN+BiLSTM, attention, multi-task models
│   ├── baselines.py             # Linear / Ridge baselines
│   ├── train.py                 # training loop with early stopping
│   ├── evaluate.py              # RMSE and plotting
│   └── run_experiments.py       # end-to-end runner
├── Gait_GDI_CS229_Presentation.pptx
├── requirements.txt
└── README.md
```

## 4. Data

Two parts are needed:

1. **CSV files** (included, or re-download with `python -m src.setup_data`): image paths and GDI labels from the CS229DP repository.
2. **IUV images** (not included, not publicly available): the PNGs must be placed at

   ```
   data/densepose-out/<patient-id>-processed/*_IUV.png
   ```

   The CSVs reference these paths. Without the images the code reports `Built 0 sequences ... missing image refs skipped` and stops with a clear error.

## 5. Setup and usage

```bash
pip install -r requirements.txt

# optional: re-download the CSV files
python -m src.setup_data

# quick test (1 epoch, small CSV by default)
python -m src.run_experiments --epochs 1

# full experiment
python -m src.run_experiments --epochs 40
```

Run all commands from the project root. To use the full CSV instead of the small subset, set `USE_FULL_CSV = True` in `src/config.py`.

## 6. Default settings (`src/config.py`)

| Setting | Value |
|---|---|
| Sequence length | 10 frames |
| Image size | 64 x 64 |
| Input channels | 3 (I, U, V) |
| Batch size | 16 |
| Epochs | 40 (max) |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Dropout | 0.3 |
| CNN channels | 32 |
| LSTM | 2 layers, 128 hidden |
| Early stopping patience | 8 |
| Split | 70 / 15 / 15 by patient |
| Seed | 42 |

## 7. Acknowledgements

- CS229DP project data layout and CSVs: [agotlin/CS229DP](https://github.com/agotlin/CS229DP)
- DensePose: Güler et al., *DensePose: Dense Human Pose Estimation in the Wild* (CVPR 2018)
- GDI: Schwartz and Rozumalski, *The Gait Deviation Index* (Gait & Posture, 2008)
