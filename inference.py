"""
SentimentNet — inference.py
Run sentiment prediction on any movie review text.

Usage:
    # Interactive mode (prompts for text)
    python inference.py

    # Single review via CLI flag
    python inference.py --text "This film was an absolute masterpiece!"

    # From a text file
    python inference.py --file review.txt
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


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
# Predictor class — load once, call many times
# ─────────────────────────────────────────────
class SentimentPredictor:
    """
    Wraps the fine-tuned DistilBERT model for production inference.

    Example:
        predictor = SentimentPredictor("outputs/best_model.pt", "outputs")
        result = predictor.predict("The acting was phenomenal.")
        # {'label': 'POSITIVE', 'confidence': 0.9873, 'scores': {'NEGATIVE': 0.0127, 'POSITIVE': 0.9873}}
    """

    LABEL_MAP = {0: "NEGATIVE", 1: "POSITIVE"}

    def __init__(self, checkpoint_path: str, tokenizer_dir: str = "outputs", max_len: int = 256):
        self.device  = get_device()
        self.max_len = max_len

        print(f"[INFO] Loading model from {checkpoint_path} on {self.device} …")

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg  = ckpt.get("config", {})
        model_name = cfg.get("model_name", "distilbert-base-uncased")
        self.max_len = cfg.get("max_len", max_len)

        # Tokenizer — prefer saved version for reproducibility
        tok_path = tokenizer_dir if os.path.exists(
            os.path.join(tokenizer_dir, "tokenizer_config.json")
        ) else model_name
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tok_path)

        # Model
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(f"[INFO] Model ready  (checkpoint val_acc={ckpt.get('val_acc', '?')})")

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        """
        Parameters
        ----------
        text : str   Raw review text (any length).

        Returns
        -------
        dict with keys:
            label      — 'POSITIVE' or 'NEGATIVE'
            confidence — float [0, 1], probability of the predicted class
            scores     — {'NEGATIVE': float, 'POSITIVE': float}
        """
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs  = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()   # [neg_prob, pos_prob]

        pred_idx   = int(probs[1] >= 0.5)   # threshold at 0.5
        label      = self.LABEL_MAP[pred_idx]
        confidence = probs[pred_idx]

        return {
            "label":      label,
            "confidence": round(confidence, 4),
            "scores": {
                "NEGATIVE": round(probs[0], 4),
                "POSITIVE": round(probs[1], 4),
            },
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Convenience wrapper for a list of texts (sequential, no batching overhead)."""
        return [self.predict(t) for t in texts]


# ─────────────────────────────────────────────
# Pretty print helper
# ─────────────────────────────────────────────
def print_result(text: str, result: dict):
    label      = result["label"]
    confidence = result["confidence"] * 100
    neg_score  = result["scores"]["NEGATIVE"] * 100
    pos_score  = result["scores"]["POSITIVE"] * 100
    emoji      = "😊" if label == "POSITIVE" else "😞"

    bar_len   = 30
    pos_fill  = int(bar_len * result["scores"]["POSITIVE"])
    bar       = "█" * pos_fill + "░" * (bar_len - pos_fill)

    print("\n" + "─" * 60)
    print(f"  Review  : {text[:120]}{'…' if len(text) > 120 else ''}")
    print("─" * 60)
    print(f"  Label      : {emoji}  {label}")
    print(f"  Confidence : {confidence:.2f}%")
    print(f"  NEG ◀ [{bar}] ▶ POS")
    print(f"  Negative: {neg_score:.2f}%   Positive: {pos_score:.2f}%")
    print("─" * 60 + "\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SentimentNet — Inference")
    parser.add_argument("--checkpoint_path", type=str, default="outputs/best_model.pt",
                        help="Path to the saved model checkpoint (.pt file)")
    parser.add_argument("--tokenizer_dir",   type=str, default="outputs",
                        help="Directory containing saved tokenizer files")
    parser.add_argument("--text",            type=str, default=None,
                        help="Review text to classify")
    parser.add_argument("--file",            type=str, default=None,
                        help="Path to a .txt file containing a review")
    args = parser.parse_args()

    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint_path}")
        print("        Run train.py first to generate the checkpoint.")
        sys.exit(1)

    predictor = SentimentPredictor(args.checkpoint_path, args.tokenizer_dir)

    # ── Determine input source ────────────────────────────────────────────
    if args.text:
        # Single text from CLI flag
        result = predictor.predict(args.text)
        print_result(args.text, result)

    elif args.file:
        # Read from file
        with open(args.file, "r") as f:
            text = f.read().strip()
        result = predictor.predict(text)
        print_result(text, result)

    else:
        # Interactive loop
        print("\n  SentimentNet — Interactive Mode")
        print("  Type a movie review and press Enter. Type 'quit' to exit.\n")
        while True:
            try:
                text = input("  Review > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye!")
                break
            if not text:
                continue
            if text.lower() in {"quit", "exit", "q"}:
                print("  Goodbye!")
                break
            result = predictor.predict(text)
            print_result(text, result)


if __name__ == "__main__":
    main()
