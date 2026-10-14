# Predicting Gait Deviation Index (GDI) from DensePose IUV Sequences

Re-implementation and **improvement** of Gotlin, Pancholi & Agarwal  
*Automated Identification of Gait Abnormalities* (CS229 / Paper 31).

**Data:** [agotlin/CS229DP](https://github.com/agotlin/CS229DP) — DensePose `*_IUV.png` frames + continuous GDI labels.

| Paper 31 | This project |
|----------|--------------|
| CNN + LSTM → GDI (RMSE ~4.4 best) | Same task + **BiLSTM**, **temporal attention**, **multi-task** (GDI + severity bin) |
| Private Gillette videos → DensePose | Uses their **released IUV paths / data.zip** |
| Random-ish video split | **Patient-disjoint** train/val/test |
| Keras-era stack | PyTorch, early stopping, baselines |

---

## Setup

```bash
cd gait-gdi-cs229
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) CSVs
python -m src.setup_data

# 2) Images (required)
# Download data.zip from https://github.com/agotlin/CS229DP
# Unzip so files exist at:
#   data/densepose-out/<patient_id>-processed/*_IUV.png
```

Start with the small CSV (`imagepath_gdi_10.csv`, default `USE_FULL_CSV = False` in `config.py`).  
Set `USE_FULL_CSV = True` only when the full IUV tree is present.

---

## Run

```bash
# Smoke: baselines + all models
python -m src.run_experiments --epochs 20

# Individual
python -m src.baselines
python -m src.train --model lstm --epochs 40
python -m src.train --model attn --epochs 40
python -m src.train --model multitask --epochs 40
python -m src.evaluate --ckpt attn_best.pt
```

Metric: **RMSE** of predicted vs true GDI (lower is better). Paper’s best reported ~**4.4**.

---

## Improvements over the paper

1. **Temporal attention** — pool LSTM outputs by learned frame importance  
2. **Multi-task learning** — regress GDI + classify coarse severity bins (aux loss)  
3. **Patient-disjoint splits** — no frame from the same patient in train and test  
4. **Modern training** — Adam + weight decay, batch norm, early stopping, Ridge/Linear baselines  
5. **Reproducible layout** — single package, `results.json`, scatter + residual plots  

---

## Layout

```
gait-gdi-cs229/
├── data/
│   ├── imagepath_gdi.csv          # from CS229DP
│   ├── imagepath_gdi_10.csv
│   └── densepose-out/             # unzipped IUV PNGs
├── src/
│   ├── config.py
│   ├── setup_data.py
│   ├── dataset.py
│   ├── models.py      # CNNLSTM, CNNLSTMAttn, CNNLSTMMultiTask
│   ├── baselines.py
│   ├── train.py
│   ├── evaluate.py
│   └── run_experiments.py
└── outputs/
```

---

## Citation

```
Gotlin, Pancholi, Agarwal — Automated Identification of Gait Abnormalities (CS229).
Code/data pointers: https://github.com/agotlin/CS229DP
```
