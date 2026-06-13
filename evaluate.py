"""
SentimentNet — evaluate.py
Loads the best checkpoint and produces:
  • Test accuracy, F1, Precision, Recall
  • Confusion matrix (PNG)
  • Training curves (PNG, re-rendered from saved history)
"""

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on all platforms
import matplotlib.pyplot as plt
import seaborn as sns


# ─────────────────────────────────────────────
# Re-use IMDBDataset from train.py
# ─────────────────────────────────────────────
from torch.utils.data import Dataset

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────
def plot_confusion_matrix(cm, output_dir):
    """Saves a styled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title("Confusion Matrix — SentimentNet (DistilBERT)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved → {path}")


def plot_training_curves(history_path, output_dir):
    """Re-renders training curves from the saved JSON history."""
    if not os.path.exists(history_path):
        print(f"[WARN] History file not found at {history_path}. Skipping curves.")
        return

    with open(history_path) as f:
        h = json.load(f)

    epochs = range(1, len(h["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("SentimentNet — Training History", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(epochs, h["train_loss"], "b-o", label="Train Loss",      linewidth=1.8)
    axes[0].plot(epochs, h["val_loss"],   "r-o", label="Validation Loss", linewidth=1.8)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, h["train_acc"], "b-o", label="Train Accuracy",      linewidth=1.8)
    axes[1].plot(epochs, h["val_acc"],   "r-o", label="Validation Accuracy", linewidth=1.8)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Training curves saved → {path}")


# ─────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────
def main(args):
    device = get_device()
    print(f"[INFO] Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────────────
    print(f"[INFO] Loading checkpoint: {args.checkpoint_path}")
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    cfg  = ckpt.get("config", {})

    max_len    = cfg.get("max_len",    args.max_len)
    batch_size = cfg.get("batch_size", args.batch_size)
    model_name = cfg.get("model_name", "distilbert-base-uncased")

    print(f"  Checkpoint epoch   : {ckpt.get('epoch', '?')}")
    print(f"  Checkpoint val_acc : {ckpt.get('val_acc', '?'):.4f}")

    # ── Tokenizer & model ─────────────────────────────────────────────────
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        args.output_dir if os.path.exists(os.path.join(args.output_dir, "tokenizer_config.json"))
        else model_name
    )
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # ── Test data ─────────────────────────────────────────────────────────
    print("[INFO] Loading IMDB test set …")
    raw         = load_dataset("imdb")
    test_texts  = raw["test"]["text"]
    test_labels = raw["test"]["label"]

    test_ds     = IMDBDataset(test_texts, test_labels, tokenizer, max_len)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Inference ─────────────────────────────────────────────────────────
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds   = probs.argmax(axis=-1)

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs[:, 1])   # probability of Positive class

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Metrics ───────────────────────────────────────────────────────────
    acc       = accuracy_score(all_labels, all_preds)
    f1        = f1_score(all_labels, all_preds, average="binary")
    precision = precision_score(all_labels, all_preds, average="binary")
    recall    = recall_score(all_labels, all_preds, average="binary")
    cm        = confusion_matrix(all_labels, all_preds)

    print("\n" + "="*60)
    print("  SENTIMENTNET — TEST SET RESULTS")
    print("="*60)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print("\n  Classification Report:")
    print(classification_report(all_labels, all_preds,
                                 target_names=["Negative", "Positive"]))
    print("="*60 + "\n")

    # Save metrics to JSON
    metrics = {
        "accuracy":  round(acc,       4),
        "f1":        round(f1,        4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
    }
    metrics_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved → {metrics_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_confusion_matrix(cm, args.output_dir)
    plot_training_curves(
        os.path.join(args.output_dir, "training_history.json"),
        args.output_dir,
    )

    print("[INFO] Evaluation complete.")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentimentNet — Evaluation")
    parser.add_argument("--checkpoint_path", type=str, default="outputs/best_model.pt")
    parser.add_argument("--output_dir",      type=str, default="outputs")
    parser.add_argument("--max_len",         type=int, default=256)
    parser.add_argument("--batch_size",      type=int, default=32)
    args = parser.parse_args()
    main(args)
