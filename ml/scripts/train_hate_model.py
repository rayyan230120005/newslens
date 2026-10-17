"""
train_hate_model.py

Fine-tunes bert-base-uncased for binary implicit-hate classification,
combining HateXplain + Implicit Hate Corpus (IHC) for training.

Why combine rather than swap: v1 (HateXplain alone) scored 78.7% test
accuracy / 0.823 F1, but testing on real target-case sentences (implicit
stereotyping without slurs) showed it only caught explicit profanity/slurs
- exactly the gap IHC was built to fill. IHC is added to the TRAINING set
only - validation and test stay pure HateXplain, so accuracy/F1 numbers
stay comparable to v1, and any improvement on implicit-stereotype
sentences reflects real generalization, not test-set contamination.

Same GPU tuning proven in the last run (small batch + gradient
accumulation + fp16 + dynamic padding + fragmentation fix + save only at
the end, given this machine's 4GB VRAM / limited system RAM).
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_from_disk, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = "bert-base-uncased"
HATEXPLAIN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "hatexplain")
IHC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ihc")
# v2, not overwriting v1 - lets us compare, and fall back if this does worse.
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "hate-bert-v2")


def load_and_prepare(tokenizer):
    hatexplain = load_from_disk(HATEXPLAIN_DIR)
    ihc = load_from_disk(IHC_DIR)

    combined_train = concatenate_datasets([hatexplain["train"], ihc["train"]])
    combined_train = combined_train.shuffle(seed=42)

    ds = {
        "train": combined_train,
        "validation": hatexplain["validation"],
        "test": hatexplain["test"],
    }

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    tokenized = {}
    for split_name, split_ds in ds.items():
        t = split_ds.map(tokenize, batched=True)
        t = t.rename_column("label", "labels")
        keep = {"input_ids", "attention_mask", "labels"}
        drop = [c for c in t.column_names if c not in keep]
        t = t.remove_columns(drop)
        t.set_format("torch")
        tokenized[split_name] = t

    return tokenized


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
    }


def main():
    print("CUDA available:", torch.cuda.is_available())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized = load_and_prepare(tokenizer)
    print(f"Combined training examples: {len(tokenized['train'])} (HateXplain {15383} + IHC {6346})")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        max_grad_norm=1.0,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\n--- Final check on held-out test set (never seen during training) ---")
    test_results = trainer.evaluate(tokenized["test"])
    print(test_results)

    trainer.save_model(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()