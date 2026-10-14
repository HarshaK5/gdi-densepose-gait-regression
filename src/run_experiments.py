"""Run baselines + all neural models; write results.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines import run_baselines
from src.config import OUTPUT_DIR
from src.evaluate import plot_history, plot_scatter
from src.train import train_one


def main(skip_baselines: bool = False, epochs: int = 40):
    results = {}
    if not skip_baselines:
        print("\n===== BASELINES =====")
        try:
            results["baselines"] = run_baselines()
        except Exception as e:
            print(f"Baselines failed: {e}")

    for name in ("lstm", "attn", "multitask"):
        print(f"\n===== TRAIN {name.upper()} =====")
        out = train_one(model_name=name, epochs=epochs)
        results[name] = {"test_rmse": out["test_rmse"]}
        plot_scatter(
            out["targets"],
            out["preds"],
            f"{name} test GDI",
            f"scatter_{name}.png",
        )
        plot_history(out["history"], f"curves_{name}.png")

    path = OUTPUT_DIR / "results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--epochs", type=int, default=40)
    args = p.parse_args()
    main(skip_baselines=args.skip_baselines, epochs=args.epochs)
