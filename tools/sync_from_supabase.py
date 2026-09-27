"""Sync fabric captures and images from Supabase for dataset retraining.

Downloads newly logged inspection images and their labels from Supabase into
data/captures/ (or a directory of your choice).

Usage:
    python tools/sync_from_supabase.py
    python tools/sync_from_supabase.py --dest data/train
    python tools/sync_from_supabase.py --limit 500
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "static" / "supabase-config.js"
DEFAULT_DEST = ROOT / "data" / "captures"


def load_config() -> tuple[str, str, str, str]:
    if not CONFIG_FILE.exists():
        sys.exit(f"Missing config: {CONFIG_FILE}. Run setup or specify --url and --key.")

    content = CONFIG_FILE.read_text(encoding="utf-8")
    url_match = re.search(r'url:\s*["\']([^"\']+)["\']', content)
    key_match = re.search(r'anonKey:\s*["\']([^"\']+)["\']', content)
    table_match = re.search(r'table:\s*["\']([^"\']+)["\']', content)
    bucket_match = re.search(r'bucket:\s*["\']([^"\']+)["\']', content)

    if not url_match or not key_match:
        sys.exit(f"Could not parse url or anonKey in {CONFIG_FILE}")

    url = url_match.group(1).rstrip("/")
    key = key_match.group(1)
    table = table_match.group(1) if table_match else "captures"
    bucket = bucket_match.group(1) if bucket_match else "fabric-captures"
    return url, key, table, bucket


def fetch_captures(url: str, key: str, table: str, limit: int = 1000) -> list[dict]:
    query = (
        f"{url}/rest/v1/{table}?"
        "select=id,captured_at,class_name,confidence,verdict,image_path&"
        "image_path=not.is.null&"
        "order=captured_at.desc&"
        f"limit={limit}"
    )
    req = urllib.request.Request(
        query,
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8")
        if "permission denied" in body or "42501" in body:
            sys.exit(
                "\n[!] Permission Denied on table 'public.captures'.\n"
                "To allow dataset synchronization, run this in your Supabase SQL Editor:\n\n"
                "    grant select on public.captures to anon;\n"
                "    create policy \"anon can select captures\" on public.captures for select to anon using (true);\n"
            )
        sys.exit(f"Failed to query Supabase ({err.code}): {body}")


def download_image(img_url: str, dest_path: Path) -> bool:
    if dest_path.exists():
        return False  # Already downloaded
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(img_url, dest_path)
        return True
    except Exception as err:
        print(f"Warning: Failed to download {img_url}: {err}")
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help="Folder to save downloaded captures into (default: data/captures)")
    parser.add_argument("--limit", type=int, default=1000,
                        help="Maximum records to fetch (default: 1000)")
    parser.add_argument("--merge-into-train", action="store_true",
                        help="Also copy downloaded images directly into data/train/<class_name>/")
    args = parser.parse_args()

    url, key, table, bucket = load_config()
    print(f"Fetching captures from Supabase ({url})...")
    records = fetch_captures(url, key, table, args.limit)
    print(f"Found {len(records)} records with stored images.")

    if not records:
        print("No image records found yet. Capture fabrics via the app to populate data.")
        return

    dest_dir = args.dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dest_dir / "manifest.csv"

    downloaded = 0
    skipped = 0

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "class_name", "verdict", "confidence", "captured_at", "local_file"])

        for r in records:
            img_path = r.get("image_path")
            if not img_path:
                continue

            cls = r.get("class_name", "unknown")
            file_name = Path(img_path).name
            target_file = dest_dir / cls / file_name

            public_url = f"{url}/storage/v1/object/public/{bucket}/{img_path}"
            if download_image(public_url, target_file):
                downloaded += 1
            else:
                skipped += 1

            writer.writerow([
                r.get("id"),
                cls,
                r.get("verdict"),
                r.get("confidence"),
                r.get("captured_at"),
                str(target_file.relative_to(ROOT)),
            ])

            if args.merge_into_train:
                train_target = ROOT / "data" / "train" / cls / file_name
                train_target.parent.mkdir(parents=True, exist_ok=True)
                if not train_target.exists() and target_file.exists():
                    train_target.write_bytes(target_file.read_bytes())

    print(f"Sync complete: {downloaded} new images downloaded, {skipped} already present.")
    print(f"Dataset manifest written to: {manifest_path}")
    if args.merge_into_train:
        print("Images merged into data/train/ for immediate retraining.")


if __name__ == "__main__":
    main()
