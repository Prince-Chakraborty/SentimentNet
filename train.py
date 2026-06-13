"""
SentimentNet — train.py
Fine-tunes DistilBERT on the IMDB dataset using a custom PyTorch training loop.
Supports Apple Silicon (mps), CUDA, and CPU.
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DEFAULTS = {
    "model_name": "distilbert-base-uncased",
    "max_len": 256,
    "batch_size": 32,
    "epochs": 5,
    "lr": 2e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "patience": 2,           # early stopping patience (epochs)
    "val_split": 0.1,        # fraction of train set used for validation
    "seed": 42,
    "output_dir": "outputs",
    "checkpoint_path": "outputs/best_model.pt",
}


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class IMDBDataset(Dataset):
    """Tokenises raw IMDB texts on the fly and returns PyTorch tensors."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),       # (max_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def get_device():
    """Pick the best available device: mps > cuda > cpu."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_training_curves(history, output_dir):
    """Saves train/val loss and accuracy curves as a PNG."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curve
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss")
    axes[0].set_title("Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc")
    axes[1].plot(epochs, history["val_acc"],   "r-o", label="Val Acc")
    axes[1].set_title("Accuracy vs Epoch")
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
# Train / Eval helpers
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, epoch):
    """Single training epoch. Returns avg loss and accuracy."""
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits                # (B, 2)

        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if (step + 1) % 50 == 0:
            print(
                f"  Epoch {epoch} | Step {step+1}/{len(loader)} "
                f"| Loss: {loss.item():.4f} "
                f"| LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluation loop. Returns avg loss, accuracy, and F1."""
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits

        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="binary")
    return avg_loss, acc, f1


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(cfg):
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = get_device()
    print(f"[INFO] Device: {device}")

    # ── Load IMDB dataset ──────────────────────────────────────────────────
    print("[INFO] Loading IMDB dataset …")
    raw = load_dataset("imdb")
    train_texts  = raw["train"]["text"]
    train_labels = raw["train"]["label"]
    test_texts   = raw["test"]["text"]
    test_labels  = raw["test"]["label"]

    # Stratified train/val split
    from sklearn.model_selection import train_test_split
    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        train_texts, train_labels,
        test_size=cfg["val_split"],
        random_state=cfg["seed"],
        stratify=train_labels,
    )
    print(
        f"[INFO] Train: {len(tr_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}"
    )

    # ── Tokeniser ─────────────────────────────────────────────────────────
    tokenizer = DistilBertTokenizerFast.from_pretrained(cfg["model_name"])

    # ── Datasets & Loaders ────────────────────────────────────────────────
    train_ds = IMDBDataset(tr_texts,   tr_labels,   tokenizer, cfg["max_len"])
    val_ds   = IMDBDataset(val_texts,  val_labels,  tokenizer, cfg["max_len"])
    test_ds  = IMDBDataset(test_texts, test_labels, tokenizer, cfg["max_len"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"[INFO] Loading {cfg['model_name']} …")
    model = DistilBertForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=2
    )
    model.to(device)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    # Separate weight decay: don't apply to bias / LayerNorm
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(param_groups, lr=cfg["lr"])

    total_steps  = len(train_loader) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    print(
        f"[INFO] Total steps: {total_steps} | Warmup steps: {warmup_steps}"
    )

    # ── Training loop with early stopping ─────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc   = 0.0
    patience_count = 0

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{cfg['epochs']}")
        print(f"{'='*60}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, epoch
        )
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        print(
            f"\n  ► Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}\n"
            f"  ► Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}\n"
            f"  ► Elapsed: {elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save(
                {
                    "epoch":      epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc":    val_acc,
                    "val_f1":     val_f1,
                    "config":     cfg,
                },
                cfg["checkpoint_path"],
            )
            print(f"  ✓ New best val_acc={val_acc:.4f} — checkpoint saved.")
        else:
            patience_count += 1
            print(
                f"  ✗ No improvement. Patience: {patience_count}/{cfg['patience']}"
            )
            if patience_count >= cfg["patience"]:
                print("[INFO] Early stopping triggered.")
                break

    # ── Save training history & curves ────────────────────────────────────
    with open(os.path.join(cfg["output_dir"], "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    save_training_curves(history, cfg["output_dir"])

    # ── Quick test-set evaluation with best checkpoint ────────────────────
    print("\n[INFO] Loading best checkpoint for final test evaluation …")
    ckpt = torch.load(cfg["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
    print(
        f"\n{'='*60}\n"
        f"  TEST RESULTS\n"
        f"  Accuracy : {test_acc:.4f}\n"
        f"  F1 Score : {test_f1:.4f}\n"
        f"{'='*60}\n"
    )

    # Save tokenizer alongside model for easy inference
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"[INFO] Tokenizer saved → {cfg['output_dir']}")
    print("[INFO] Training complete. Run evaluate.py for full metrics.")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentimentNet — DistilBERT fine-tuning")
    parser.add_argument("--batch_size",       type=int,   default=DEFAULTS["batch_size"])
    parser.add_argument("--epochs",           type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--lr",               type=float, default=DEFAULTS["lr"])
    parser.add_argument("--max_len",          type=int,   default=DEFAULTS["max_len"])
    parser.add_argument("--patience",         type=int,   default=DEFAULTS["patience"])
    parser.add_argument("--warmup_ratio",     type=float, default=DEFAULTS["warmup_ratio"])
    parser.add_argument("--weight_decay",     type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--output_dir",       type=str,   default=DEFAULTS["output_dir"])
    parser.add_argument("--checkpoint_path",  type=str,   default=DEFAULTS["checkpoint_path"])
    parser.add_argument("--seed",             type=int,   default=DEFAULTS["seed"])
    args = parser.parse_args()

    cfg = vars(args)
    cfg["model_name"] = DEFAULTS["model_name"]
    cfg["val_split"]  = DEFAULTS["val_split"]

    main(cfg)
