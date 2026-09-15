"""
download_datasets.py

Downloads BABE (framing-bias dataset, per project doc section 5) and saves
it locally, then prints its structure so we can confirm the column names
before writing any training code against them.

Only BABE for now - HateXplain/IHC/SBIC/HASOC come later, once Stage 4
(implicit hate) starts. Keeping this to one dataset per step.

    python download_datasets.py
"""

import os
from datasets import load_dataset

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def download_babe():
    print("Downloading BABE (framing-bias, ~4,100 expert-annotated sentences)...")
    ds = load_dataset("mediabiasgroup/BABE")

    save_path = os.path.join(RAW_DIR, "babe")
    ds.save_to_disk(save_path)
    print(f"Saved to {save_path}")

    print("\n--- Dataset structure ---")
    print(ds)

    print("\n--- First 3 examples ---")
    split_name = list(ds.keys())[0]
    for row in ds[split_name].select(range(min(3, len(ds[split_name])))):
        print(row)


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    download_babe()
