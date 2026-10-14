"""
Download CSVs from agotlin/CS229DP and print instructions for data.zip / PNGs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CSV_FULL, CSV_SMALL, DATA_DIR, DENSEPOSE_ROOT

BASE = "https://raw.githubusercontent.com/agotlin/CS229DP/master/data"
FILES = {
    "imagepath_gdi.csv": CSV_FULL,
    "imagepath_gdi_10.csv": CSV_SMALL,
}


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, dest in FILES.items():
        url = f"{BASE}/{name}"
        print(f"Downloading {url} ...")
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            dest.write_bytes(r.content)
            print(f"  → {dest} ({len(r.content)} bytes)")
        except Exception as e:
            print(f"  FAILED: {e}")

    print(
        f"""
Next steps (required for training):
1. Download data.zip from the repo:
     https://github.com/agotlin/CS229DP/blob/master/data.zip
   Or clone the full repo:
     git clone https://github.com/agotlin/CS229DP.git
2. Unzip so IUV PNGs exist under:
     {DENSEPOSE_ROOT}/<patient>-processed/*_IUV.png
3. Verify:
     python -c "from src.dataset import load_csv, build_sequences; print(len(build_sequences(load_csv())))"
"""
    )


if __name__ == "__main__":
    main()
