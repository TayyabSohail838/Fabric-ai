"""Split the crops from build_dataset.py into train/val/test by source photo.

Every crop from one photo goes to the same split. Photos are assigned rarest
class first, each to the split that is furthest below its target share for
that class, so even a class with ~20 photos gets some into val and test.

    python tools/split_dataset.py --src D:/fabric_data/_all --out data
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = {"train": 0.80, "val": 0.15, "test": 0.05}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.src / "manifest.csv")))
    group_classes = defaultdict(Counter)
    for r in rows:
        group_classes[r["group"]][r["class"]] += 1
    photos = Counter()  # photos per class
    for cls in group_classes.values():
        photos.update(cls.keys())

    groups = list(group_classes)
    random.Random(args.seed).shuffle(groups)
    rarest = {g: min(group_classes[g], key=lambda c: photos[c]) for g in groups}
    groups.sort(key=lambda g: photos[rarest[g]])

    assigned = defaultdict(Counter)  # class -> split -> photos
    split_of = {}
    for g in groups:
        c = rarest[g]
        s = max(SPLITS, key=lambda s: SPLITS[s] * photos[c] - assigned[c][s])
        split_of[g] = s
        for cls in group_classes[g]:
            assigned[cls][s] += 1

    crops = defaultdict(Counter)
    for r in rows:
        s = split_of[r["group"]]
        r["split"] = s
        dst = args.out / s / r["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(args.src / r["file"]), dst)
        crops[r["class"]][s] += 1

    with open(args.out / "manifest.csv", "w", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=["split", "file", "class", "group", "source"])
        cw.writeheader()
        cw.writerows(rows)

    print(f"{'class':12s}" + "".join(f"{s:>16s}" for s in SPLITS) + "   (crops / photos)")
    for c in sorted(crops):
        print(f"{c:12s}" + "".join(f"{crops[c][s]:>9d} / {assigned[c][s]:<4d}" for s in SPLITS))
    tot = Counter()
    for c in crops:
        tot.update(crops[c])
    print(f"{'total':12s}" + "".join(f"{tot[s]:>9d}       " for s in SPLITS))


if __name__ == "__main__":
    main()
