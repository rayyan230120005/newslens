"""
download_datasets.py

Downloads and preprocesses datasets for the bias and hate pipelines.

    python download_datasets.py

Currently runs the IHC download only. BABE and HateXplain were already
fetched earlier and don't need re-running - uncomment their calls below
if you ever need to refresh either.
"""

import os
from collections import Counter
from datasets import load_dataset, Dataset, DatasetDict

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def download_babe():
    print("Downloading BABE (framing-bias, ~4,100 expert-annotated sentences)...")
    ds = load_dataset("mediabiasgroup/BABE")
    save_path = os.path.join(RAW_DIR, "babe")
    ds.save_to_disk(save_path)
    print(f"Saved to {save_path}")


def download_hatexplain():
    """
    Bypasses the `datasets` library's Hub-loading path entirely for this
    one. After three different failures in a row (removed loading scripts,
    a renamed config, a nested-column shape mismatch), pulling the raw JSON
    directly from the dataset's original GitHub repo and preprocessing it
    ourselves is simply more reliable.
    """
    import requests

    base_url = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/"
    print("Fetching HateXplain raw data from GitHub...")
    dataset = requests.get(base_url + "dataset.json").json()
    divisions = requests.get(base_url + "post_id_divisions.json").json()

    def process_split(post_ids):
        rows = []
        for pid in post_ids:
            record = dataset[pid]
            text = " ".join(record["post_tokens"])
            raw_labels = [a["label"] for a in record["annotators"]]
            majority = Counter(raw_labels).most_common(1)[0][0]
            label = 0 if majority == "normal" else 1
            rows.append({"text": text, "label": label})
        return rows

    split_map = {"train": "train", "validation": "val", "test": "test"}
    ds_dict = {}
    for hf_split, source_split in split_map.items():
        rows = process_split(divisions[source_split])
        ds_dict[hf_split] = Dataset.from_list(rows)
        flagged = sum(r["label"] for r in rows)
        print(f"  {hf_split}: {len(rows)} examples ({flagged} flagged, {len(rows) - flagged} normal)")

    ds = DatasetDict(ds_dict)
    save_path = os.path.join(RAW_DIR, "hatexplain")
    ds.save_to_disk(save_path)
    print(f"Saved to {save_path}")


def download_ihc():
    print("Downloading Implicit Hate Corpus (implicit-hate examples only, 6,346 posts)...")
    ds = load_dataset("SALT-NLP/ImplicitHate")

    def to_text_and_label(example):
        # Every row in this mirror is an implicit-hate example - no "not
        # hate" rows exist here, so everything gets label=1.
        return {"text": example["post"], "label": 1}

    ds = ds.map(to_text_and_label)
    keep = {"text", "label"}
    drop = [c for c in ds["train"].column_names if c not in keep]
    ds = ds.remove_columns(drop)

    save_path = os.path.join(RAW_DIR, "ihc")
    ds.save_to_disk(save_path)
    print(f"Saved to {save_path}")
    print(ds)


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    # download_babe()        # already done
    # download_hatexplain()  # already done
    download_ihc()