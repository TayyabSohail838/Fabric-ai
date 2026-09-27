"""Retrain app.py's BaselineCNN on data/{train,val,test}.

Keeps the epoch with the best validation macro-F1 (accuracy is dominated by
defect_free) and saves its state_dict as fabric_model.pt, the exact format
app.py's load_model() reads.

    python tools/train.py --classes defect_free hole horizontal lines stain verticle broken_end
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# app.py loads weights at import time and exits if they're missing, so the
# architecture and constants are pulled from its source instead of importing it.
_src = (ROOT / "app.py").read_text(encoding="utf-8")
_ns: dict = {}
exec(_src[_src.index("MEAN = "):_src.index("def load_model")],
     {"np": np, "nn": nn, "F": torch.nn.functional}, _ns)
BaselineCNN, MEAN, STD = _ns["BaselineCNN"], _ns["MEAN"].ravel().tolist(), _ns["STD"].ravel().tolist()


class Folder(Dataset):
    def __init__(self, root: Path, classes: list[str], tf):
        self.items = [(p, i) for i, c in enumerate(classes)
                      for p in sorted((root / c).glob("*.jpg"))]
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


def macro_f1(cm: np.ndarray) -> tuple[float, np.ndarray]:
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-9), 0.0)
    return float(f1.mean()), f1


@torch.no_grad()
def evaluate(model, loader, n):
    model.eval()
    cm = np.zeros((n, n), dtype=int)
    for x, y in loader:
        pred = model(x).argmax(1)
        for t, p in zip(y.tolist(), pred.tolist()):
            cm[t, p] += 1
    return cm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    # Written next to the data first; copy over ROOT/fabric_model.pt once it checks out.
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "fabric_model.pt")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(0)
    n = len(args.classes)

    norm = T.Normalize(MEAN, STD)
    train_tf = T.Compose([
        T.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.ColorJitter(0.3, 0.3, 0.2, 0.02),
        T.ToTensor(), norm,
    ])
    eval_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), norm])

    train = Folder(args.data / "train", args.classes, train_tf)
    val = Folder(args.data / "val", args.classes, eval_tf)
    test = Folder(args.data / "test", args.classes, eval_tf)
    counts = np.bincount([y for _, y in train.items], minlength=n)
    print("train counts:", dict(zip(args.classes, counts.tolist())), flush=True)

    # Oversample rare classes to ~balanced batches, and weight the loss by
    # inverse frequency as well (sqrt, so the two don't compound too hard).
    sample_w = (1.0 / counts)[[y for _, y in train.items]]
    sampler = WeightedRandomSampler(torch.tensor(sample_w), num_samples=len(train), replacement=True)
    class_w = torch.tensor(np.sqrt(counts.max() / counts), dtype=torch.float32)
    print("loss weights:", dict(zip(args.classes, [round(v, 2) for v in class_w.tolist()])), flush=True)

    tl = DataLoader(train, args.batch, sampler=sampler, num_workers=args.workers, persistent_workers=args.workers > 0)
    vl = DataLoader(val, 64, num_workers=args.workers)
    model = BaselineCNN(num_classes=n)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)

    best, best_ep = -1.0, -1
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, seen = time.time(), 0.0, 0
        for x, y in tl:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            seen += len(y)
        sched.step()
        f1, per = macro_f1(evaluate(model, vl, n))
        mark = ""
        if f1 > best:
            best, best_ep = f1, ep
            torch.save(model.state_dict(), args.out)
            mark = "  *saved"
        print(f"ep {ep:2d} loss {tot / seen:.3f} val macroF1 {f1:.3f} "
              f"[{' '.join(f'{v:.2f}' for v in per)}] {time.time() - t0:.0f}s{mark}", flush=True)

    model.load_state_dict(torch.load(args.out, map_location="cpu"))
    print(f"\nbest epoch {best_ep}, val macro-F1 {best:.3f}")
    for name, ds in (("val", val), ("test", test)):
        cm = evaluate(model, DataLoader(ds, 64, num_workers=args.workers), n)
        f1, per = macro_f1(cm)
        print(f"\n{name}: macro-F1 {f1:.3f}  accuracy {np.trace(cm) / cm.sum():.3f}")
        print("per-class F1:", {c: round(float(v), 3) for c, v in zip(args.classes, per)})
        print("confusion (rows=true, cols=pred):")
        for c, row in zip(args.classes, cm):
            print(f"  {c:12s} {row.tolist()}")


if __name__ == "__main__":
    main()
