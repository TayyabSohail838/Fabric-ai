"""Build the classification dataset from the downloaded detection/segmentation sets.

Reads the archives in place (nothing is extracted), crops every mapped box or
polygon with padding, and writes one image per crop plus a manifest that records
which source photo each crop came from. split_dataset.py uses that manifest to
split by photo, so crops of one photo never land in two splits.

    python tools/build_dataset.py --out D:/fabric_data/_all
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

FDD_ZIP = ROOT / "Fabric Defect Detection.v1i.coco.zip"
AITEX_ZIP = ROOT / "DataSet AITEX.v1i.coco.zip"
SEG_ZIP = ROOT / "Fabric-Defects.v1i.coco-segmentation.zip"
FOLDER_ZIP = ROOT / "Fabric Defects Dataset.zip"

PAD = 0.12       # margin added on each side, as a fraction of box size
MIN_SIDE = 64    # tiny boxes are grown to at least this, around their centre
MAX_SIDE = 448   # crops are stored no larger than this (the model sees 224)

# Label mapping agreed in step 1. Anything not listed is dropped.
FDD_MAP = {"hole": "hole", "horizontal": "horizontal", "line": "lines", "vertical": "verticle"}
AITEX_MAP = {"002-Broken-End": "broken_end"}
FOLDER_MAP = {"defect free": "defect_free", "stain": "stain", "hole": "hole",
              "lines": "lines", "horizontal": "horizontal", "Vertical": "verticle"}
# Segmentation set: only the textile families. MVTec, the generic "defect"
# set and the AITEX copies (stretched to 1024x1024; the native-resolution
# AITEX set above is used instead) are skipped.
SEG_MAP = {
    "tilda": {"hole": "hole", "Oil stain": "stain", "contamination": "stain"},
    "named": {"broken_end": "broken_end", "hole": "hole"},
}

AITEX_CODE = re.compile(r"^(\d{4}_\d{3}_\d{2})")


def seg_family(name: str) -> str | None:
    if re.match(r"^c\d+r\d+e\d+n\d+", name):
        return "tilda"
    if re.match(r"^(BrokenEnd|Hole)_", name):
        return "named"
    return None


def source_key(prefix: str, name: str) -> str:
    """Stable id for the source photo, shared by augmented/renamed copies."""
    name = re.sub(r"\.rf\.[0-9a-f]+\.jpg$", "", name)
    name = re.sub(r"_(png|jpg|jpeg|bmp)$", "", name)
    m = AITEX_CODE.match(name)
    if m:
        return "aitex:" + m.group(1)
    stem = re.sub(r"\.(png|jpe?g|bmp)$", "", name, flags=re.I)
    stem = re.sub(r"[_ -]?processed.*$", "", stem)
    stem = re.sub(r"[^0-9a-z]+", "", stem.lower())
    return f"{prefix}:{stem}"


def dhash(img: Image.Image) -> np.ndarray:
    """256-bit difference hash. The 64-bit version collides on plain woven
    textures, which chained unrelated photos into one giant group."""
    g = np.asarray(img.convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).ravel()


def shrink(img: Image.Image) -> Image.Image:
    if max(img.size) > MAX_SIDE:
        img = img.copy()
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.BILINEAR)
    return img


def padded_crop(img: Image.Image, box) -> Image.Image:
    x, y, w, h = box
    W, H = img.size
    cx, cy = x + w / 2, y + h / 2
    w, h = max(w * (1 + 2 * PAD), MIN_SIDE), max(h * (1 + 2 * PAD), MIN_SIDE)
    x0, y0 = max(0, round(cx - w / 2)), max(0, round(cy - h / 2))
    x1, y1 = min(W, round(cx + w / 2)), min(H, round(cy + h / 2))
    return img.crop((x0, y0, x1, y1))


class Writer:
    def __init__(self, out: Path):
        self.out = out
        self.rows: list[dict] = []
        self.hashes: dict[str, int] = {}
        self.n = Counter()

    def save(self, img: Image.Image, cls: str, group: str, source: str, whole: bool):
        self.n[cls] += 1
        d = self.out / cls
        d.mkdir(parents=True, exist_ok=True)
        fname = f"{source}_{self.n[cls]:06d}.jpg"
        shrink(img.convert("RGB")).save(d / fname, quality=92)
        self.rows.append({"file": f"{cls}/{fname}", "class": cls, "group": group, "source": source})


def coco_crops(w: Writer, zpath: Path, label_map: dict, source: str, family=None):
    z = zipfile.ZipFile(zpath)
    for ann_path in sorted(n for n in z.namelist() if n.endswith("_annotations.coco.json")):
        folder = ann_path.rsplit("/", 1)[0]
        d = json.loads(z.read(ann_path))
        cats = {c["id"]: c["name"] for c in d["categories"]}
        by_img = defaultdict(list)
        for a in d["annotations"]:
            by_img[a["image_id"]].append(a)
        for im in d["images"]:
            fam_map = label_map
            if family is not None:
                fam = family(im["file_name"])
                if fam is None:
                    continue
                fam_map = label_map[fam]
            keep = [(a, fam_map[cats[a["category_id"]]]) for a in by_img[im["id"]]
                    if cats[a["category_id"]] in fam_map]
            if not keep:
                continue
            img = Image.open(io.BytesIO(z.read(f"{folder}/{im['file_name']}")))
            img.load()
            group = source_key(source, im.get("extra", {}).get("name", im["file_name"]))
            if re.fullmatch(r"fdd:\d+", group):
                # Bare numbers ("9_processed-2-") restart in every class folder
                # of the original dataset, so qualify them by the photo's label.
                group += "-" + Counter(c for _, c in keep).most_common(1)[0][0]
            if source == "fdd":
                w.hashes[group] = dhash(img)
            for a, cls in keep:
                w.save(padded_crop(img, a["bbox"]), cls, group, source, whole=False)


def folder_images(w: Writer):
    z = zipfile.ZipFile(FOLDER_ZIP)
    for n in z.namelist():
        parts = n.split("/")
        if n.endswith("/") or len(parts) < 4 or parts[2] not in FOLDER_MAP:
            continue
        cls = FOLDER_MAP[parts[2]]
        img = Image.open(io.BytesIO(z.read(n)))
        img.load()
        # Plain numeric names ("1.jpg") repeat across class folders, so they're
        # only the same photo within one folder.
        group = source_key("folder-" + cls, parts[-1])
        if cls != "defect_free":  # the Roboflow export has no clean photos to match
            w.hashes[group] = dhash(img)
        w.save(img, cls, group, "folder", whole=True)


def merge_duplicates(w: Writer, max_bits: int = 12) -> int:
    """Group near-identical whole photos (e.g. the same hole photo in both the
    Roboflow export and the class-folder zip) so the split keeps them together.
    Only Roboflow-vs-folder pairs are compared; within one source the file
    names already identify the photo."""
    fdd = [k for k in w.hashes if k.startswith("fdd:")]
    folder = [k for k in w.hashes if not k.startswith("fdd:")]
    if not fdd or not folder:
        return 0
    F = np.stack([w.hashes[k] for k in folder])
    rename = {}
    for k in fdd:
        dist = (F != w.hashes[k]).sum(axis=1)
        j = int(dist.argmin())
        if dist[j] <= max_bits:
            rename[folder[j]] = k
    for r in w.rows:
        r["group"] = rename.get(r["group"], r["group"])
    return len(rename)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    w = Writer(args.out)

    print("Fabric Defect Detection ...", flush=True)
    coco_crops(w, FDD_ZIP, FDD_MAP, "fdd")
    print("DataSet AITEX ...", flush=True)
    coco_crops(w, AITEX_ZIP, AITEX_MAP, "aitex")
    print("Fabric-Defects segmentation (textile subsets) ...", flush=True)
    coco_crops(w, SEG_ZIP, SEG_MAP, "seg", family=seg_family)
    print("Fabric Defects Dataset folders ...", flush=True)
    folder_images(w)
    print(f"merged {merge_duplicates(w)} near-duplicate photo pairs across sources", flush=True)

    with open(args.out / "manifest.csv", "w", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=["file", "class", "group", "source"])
        cw.writeheader()
        cw.writerows(w.rows)

    groups = defaultdict(set)
    by_src = defaultdict(Counter)
    for r in w.rows:
        groups[r["class"]].add(r["group"])
        by_src[r["class"]][r["source"]] += 1
    print(f"\n{'class':12s} {'crops':>6s} {'photos':>7s}  by source")
    for cls in sorted(by_src):
        print(f"{cls:12s} {sum(by_src[cls].values()):6d} {len(groups[cls]):7d}  {dict(by_src[cls])}")


if __name__ == "__main__":
    main()
