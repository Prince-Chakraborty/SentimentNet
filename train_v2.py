"""
SentimentNet — train_v2.py
Fixed version: label smoothing + frozen lower layers + max_len=256
Targets 92%+ test accuracy on IMDB.
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
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULTS = {
    "model_name": "distilbert-base-uncased",
    "max_len": 256,
    "batch_size": 16,
    "epochs": 4,
    "lr": 2e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "patience": 2,
    "val_split": 0.1,
    "seed": 42,
    "output_dir": "outputs",
    "checkpoint_path": "outputs/best_model.pt",
    "label_smoothing": 0.1,
    "freeze_layers": 3,  # freeze bottom N transformer layers
}


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

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
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def freeze_bottom_layers(model, n_layers):
    """Freeze embeddings and bottom n_layers transformer blocks."""
    # Freeze embeddings
    for param in model.distilbert.embeddings.parameters():
        param.requires_grad = False
    # Freeze bottom n transformer layers
    for i in range(n_layers):
        for param in model.distilbert.transformer.layer[i].parameters():
            param.requires_grad = False
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Frozen: {frozen:,} | Trainable: {trainable:,}")


def save_training_curves(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    axes[0].set_title("Loss vs Epoch"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], "r-o", label="Val Acc")
    axes[1].set_title("Accuracy vs Epoch"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[INFO] Training curves saved → {path}")


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, epoch):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        if (step + 1) % 100 == 0:
            print(f"  Epoch {epoch} | Step {step+1}/{len(loader)} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
    return total_loss / len(loader), accuracy_score(all_labels, all_preds)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="binary")


def main(cfg):
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = get_device()
    print(f"[INFO] Device: {device}")

    print("[INFO] Loading IMDB dataset ...")
    raw = load_dataset("imdb")
    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        raw["train"]["text"], raw["train"]["label"],
        test_size=cfg["val_split"], random_state=cfg["seed"], stratify=raw["train"]["label"]
    )
    test_texts, test_labels = raw["test"]["text"], raw["test"]["label"]
    print(f"[INFO] Train: {len(tr_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(cfg["model_name"])

    train_loader = DataLoader(IMDBDataset(tr_texts, tr_labels, tokenizer, cfg["max_len"]), batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader   = DataLoader(IMDBDataset(val_texts, val_labels, tokenizer, cfg["max_len"]), batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    test_loader  = DataLoader(IMDBDataset(test_texts, test_labels, tokenizer, cfg["max_len"]), batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    print(f"[INFO] Loading {cfg['model_name']} ...")
    model = DistilBertForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=2)

    # Freeze bottom layers to prevent overfitting
    freeze_bottom_layers(model, cfg["freeze_layers"])
    model.to(device)

    # Separate weight decay groups
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], "weight_decay": cfg["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(param_groups, lr=cfg["lr"])

    total_steps = len(train_loader) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Label smoothing to prevent overconfident predictions
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    print(f"[INFO] Total steps: {total_steps} | Warmup: {warmup_steps} | Label smoothing: {cfg['label_smoothing']}")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience_count = 0

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        print(f"\n{'='*60}\nEPOCH {epoch}/{cfg['epochs']}\n{'='*60}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        print(f"\n  ► Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  ► Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        print(f"  ► Elapsed: {time.time()-t0:.1f}s")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_count = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_acc": val_acc, "val_f1": val_f1, "config": cfg}, cfg["checkpoint_path"])
            print(f"  ✓ New best val_acc={val_acc:.4f} — checkpoint saved.")
        else:
            patience_count += 1
            print(f"  ✗ No improvement. Patience: {patience_count}/{cfg['patience']}")
            if patience_count >= cfg["patience"]:
                print("[INFO] Early stopping triggered.")
                break

    with open(os.path.join(cfg["output_dir"], "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    save_training_curves(history, cfg["output_dir"])

    print("\n[INFO] Loading best checkpoint for final test evaluation ...")
    ckpt = torch.load(cfg["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    _, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f"\n{'='*60}\n  TEST RESULTS\n  Accuracy : {test_acc:.4f}\n  F1 Score : {test_f1:.4f}\n{'='*60}\n")

    tokenizer.save_pretrained(cfg["output_dir"])
    print("[INFO] Training complete. Run evaluate.py for full metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len",         type=int,   default=DEFAULTS["max_len"])
    parser.add_argument("--batch_size",      type=int,   default=DEFAULTS["batch_size"])
    parser.add_argument("--epochs",          type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--lr",              type=float, default=DEFAULTS["lr"])
    parser.add_argument("--patience",        type=int,   default=DEFAULTS["patience"])
    parser.add_argument("--warmup_ratio",    type=float, default=DEFAULTS["warmup_ratio"])
    parser.add_argument("--weight_decay",    type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--label_smoothing", type=float, default=DEFAULTS["label_smoothing"])
    parser.add_argument("--freeze_layers",   type=int,   default=DEFAULTS["freeze_layers"])
    parser.add_argument("--output_dir",      type=str,   default=DEFAULTS["output_dir"])
    parser.add_argument("--checkpoint_path", type=str,   default=DEFAULTS["checkpoint_path"])
    parser.add_argument("--seed",            type=int,   default=DEFAULTS["seed"])
    args = parser.parse_args()
    cfg = vars(args)
    cfg["model_name"] = DEFAULTS["model_name"]
    cfg["val_split"]  = DEFAULTS["val_split"]
    main(cfg)
