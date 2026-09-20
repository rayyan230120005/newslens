"""
inspect_hatexplain.py

Standalone check - loads whatever's already saved at ml/data/raw/hatexplain
and prints its structure, without re-running the download. If this errors
with "not found", the save never actually happened.
"""

import os
from datasets import load_from_disk

path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "hatexplain")
ds = load_from_disk(path)

print("--- Dataset structure ---")
print(ds)

print("\n--- First 3 examples ---")
for row in ds["train"].select(range(3)):
    print(row)