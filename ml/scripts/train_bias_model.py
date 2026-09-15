"""
train_bias_model.py

Fine-tunes bert-base-uncased for binary framing-bias classification
(biased / not-biased) on BABE, per project doc section 3a.

Tuned for a 4GB-VRAM GPU (fp16 + small batch + gradient accumulation +
dynamic padding). Run check_gpu.py first to confirm CUDA is actually
detected before running this.

    python train_bias_model.py
"""

import os
import torch
from datasets import load_from_disk
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
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "babe")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "bias-bert")


def load_and_prepare(tokenizer):
    ds = load_from_disk(RAW_DIR)

    def tokenize(batch):
        # No padding here - the data collator pads per-batch instead,
        # dynamically, which is what keeps memory use down on a small GPU.
        return tokenizer(batch["text"], truncation=True, max_length=128)

    tokenized = ds.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")

    keep = {"input_ids", "attention_mask", "labels"}
    split_name = list(tokenized.keys())[0]
    drop = [c for c in tokenized[split_name].column_names if c not in keep]
    tokenized = tokenized.remove_columns(drop)
    tokenized.set_format("torch")

    if len(tokenized.keys()) == 1:
        split_name = list(tokenized.keys())[0]
        tokenized = tokenized[split_name].train_test_split(test_size=0.1, seed=42)

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
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,   # effective batch size = 16
        fp16=torch.cuda.is_available(),  # half precision, GPU only
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()