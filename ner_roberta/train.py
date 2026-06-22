"""Fine-tune xlm-roberta-base on the windowed MEDDOCAN data.

Loss is computed only on the first subword of each word (continuation subwords
and special/pad positions are masked to -100), matching the postprocessing that
merges subwords back into words.
"""

import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import DATASET_DIR, ID2LABEL
from .model import build_model


class WindowDataset(Dataset):
    def __init__(self, pkl_path: str):
        with open(pkl_path, "rb") as f:
            blob = pickle.load(f)
        self.ids = blob["sample_ids"]
        self.samples = blob["samples"]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i):
        s = self.samples[self.ids[i]]
        # Train only on first-subword positions; ignore the rest.
        train_labels = np.where(s["word_start_mask"] == 1, s["labels"], -100)
        return (
            torch.as_tensor(s["input_ids"], dtype=torch.long),
            torch.as_tensor(s["attention_mask"], dtype=torch.long),
            torch.as_tensor(train_labels, dtype=torch.long),
        )


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    try:
        from seqeval.metrics import classification_report, f1_score
    except Exception:
        classification_report = f1_score = None

    pred_seqs, gold_seqs = [], []
    correct = total = 0
    for input_ids, attn, labels in loader:
        logits = model(input_ids=input_ids.to(device),
                       attention_mask=attn.to(device)).logits
        preds = logits.argmax(-1).cpu().numpy()
        labels = labels.numpy()
        for p_row, l_row in zip(preds, labels):
            mask = l_row != -100
            if not mask.any():
                continue
            p = p_row[mask]
            l = l_row[mask]
            correct += int((p == l).sum())
            total += int(mask.sum())
            pred_seqs.append([ID2LABEL[int(x)] for x in p])
            gold_seqs.append([ID2LABEL[int(x)] for x in l])

    out = {"token_acc": correct / max(total, 1)}
    if f1_score is not None and gold_seqs:
        out["entity_f1"] = float(f1_score(gold_seqs, pred_seqs))
        out["report"] = classification_report(gold_seqs, pred_seqs)
    return out


def train(epochs: int = 3, batch_size: int = 32, lr: float = 5e-5,
          out_dir: str = "checkpoints", limit: int = 0) -> str:
    device = _pick_device()
    print(f"device: {device}")

    train_ds = WindowDataset(os.path.join(DATASET_DIR, "training.pkl"))
    val_ds = WindowDataset(os.path.join(DATASET_DIR, "validation.pkl"))
    if limit:
        train_ds.ids = train_ds.ids[:limit]
        val_ds.ids = val_ds.ids[: max(1, limit // 4)]
    print(f"train windows: {len(train_ds)} | val windows: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for step, (input_ids, attn, labels) in enumerate(train_loader, 1):
            out = model(input_ids=input_ids.to(device),
                        attention_mask=attn.to(device),
                        labels=labels.to(device))
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += out.loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {running / step:.4f}")
        metrics = evaluate(model, val_loader, device)
        print(f"[epoch {epoch}] train_loss={running / len(train_loader):.4f} "
              f"val_token_acc={metrics['token_acc']:.4f} "
              f"val_entity_f1={metrics.get('entity_f1', float('nan')):.4f}")

    if metrics.get("report"):
        print(metrics["report"])

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    print(f"saved model -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--limit", type=int, default=0, help="cap train windows (debug)")
    args = ap.parse_args()
    train(args.epochs, args.batch_size, args.lr, args.out_dir, args.limit)
