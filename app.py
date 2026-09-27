"""Fabric defect inspection - web app backend.

Serves the BaselineCNN fabric classifier over HTTP so a browser can stream
camera frames to it. Run:  python app.py
"""

from __future__ import annotations

import argparse
import io
import ipaddress
import json
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
WEIGHTS = ROOT / "fabric_model.pt"

# Raw class order is fixed by training - do not reorder.
# (name, display label, what the operator is looking at)
CLASSES = [
    ("defect_free", "Clean", "No defect in the aperture."),
    ("hole", "Hole", "Broken ends or picks - a gap in the weave."),
    ("horizontal", "Weft bar", "A band running across the weft."),
    ("lines", "Streaks", "Irregular streaking over the surface."),
    ("stain", "Stain", "Discolouration or soil on the face."),
    ("verticle", "Warp line", "A line running along the warp."),
]
CLASS_NAMES = [c[0] for c in CLASSES]
CLEAN_INDEX = CLASS_NAMES.index("defect_free")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def build_model(num_classes: int) -> nn.Module:
    """The architecture fabric_model.pt holds: MobileNetV3-small with a
    num_classes head, fine-tuned from ImageNet weights by
    `tools/train.py --arch mobilenet_v3_small`. BaselineCNN above is kept for
    `--arch baseline` runs, but its checkpoints won't load here."""
    from torchvision.models import mobilenet_v3_small

    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def load_model() -> nn.Module:
    if not WEIGHTS.exists():
        sys.exit(f"Weights not found: {WEIGHTS}\nPut fabric_model.pt next to app.py.")
    model = build_model(len(CLASSES))
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()
    return model


MODEL = load_model()
torch.set_grad_enabled(False)
INFER_LOCK = threading.Lock()  # torch module is shared; serialise access


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL image -> normalised CHW float tensor at 224x224."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (224, 224):
        img = img.resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


def tile_images(img: Image.Image, n: int = 3) -> list[Image.Image]:
    """Split an image into an n x n grid, row-major."""
    w, h = img.size
    out = []
    for r in range(n):
        for c in range(n):
            box = (
                round(c * w / n),
                round(r * h / n),
                round((c + 1) * w / n),
                round((r + 1) * h / n),
            )
            out.append(img.crop(box))
    return out


def run_batch(tensors: list[torch.Tensor], tta: bool) -> np.ndarray:
    """Return (N, num_classes) probabilities, optionally flip-averaged."""
    batch = torch.stack(tensors)
    views = [batch]
    if tta:
        views.append(torch.flip(batch, dims=[3]))  # mirror across the warp
        views.append(torch.flip(batch, dims=[2]))  # mirror across the weft
    with INFER_LOCK:
        probs = None
        for view in views:
            p = torch.softmax(MODEL(view), dim=1)
            probs = p if probs is None else probs + p
    return (probs / len(views)).numpy()


def summarise(probs: np.ndarray) -> dict:
    idx = int(probs.argmax())
    name, label, _ = CLASSES[idx]
    return {
        "index": idx,
        "name": name,
        "label": label,
        "prob": float(probs[idx]),
        "defect": float(1.0 - probs[CLEAN_INDEX]),
        "probs": [float(p) for p in probs],
    }


def probe_model() -> dict:
    """Check that the checkpoint actually responds to its input.

    A net saved before training returns softmax(fc2.bias) for every image, so
    the app would show confident-looking labels that mean nothing. Push a few
    very different patterns through and measure how far the logits travel.
    """
    plates = [Image.new("RGB", (224, 224), c) for c in ((0, 0, 0), (255, 255, 255), (128, 128, 128))]
    vert = np.zeros((224, 224, 3), np.uint8)
    vert[:, ::16] = 255
    horiz = np.zeros((224, 224, 3), np.uint8)
    horiz[::16, :] = 255
    plates += [Image.fromarray(vert), Image.fromarray(horiz)]

    with INFER_LOCK:
        logits = MODEL(torch.stack([to_tensor(p) for p in plates])).numpy()
    spread = float(logits.std(axis=0).mean())
    return {
        "spread": round(spread, 4),
        "trained": spread > 0.15,
        "note": (
            "Weights match an untrained initialisation - every frame returns the same "
            "prediction regardless of what the camera sees. Retrain the model and replace "
            "fabric_model.pt to get real results."
        ),
    }


HEALTH = probe_model()
if not HEALTH["trained"]:
    print(
        "\n  WARNING: fabric_model.pt looks untrained.\n"
        f"  Logit spread across black / white / grey / striped test plates is only {HEALTH['spread']}\n"
        "  (a trained net moves by whole units). Predictions will be meaningless until\n"
        "  the model is retrained. The app will flag this in the interface."
    )


app = FastAPI(title="Fabric inspection")


@app.get("/api/meta")
def meta():
    return {
        "classes": [
            {"name": n, "label": lab, "hint": hint, "clean": n == "defect_free"}
            for n, lab, hint in CLASSES
        ],
        "input": 224,
        "threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "health": HEALTH,
    }


@app.post("/api/predict")
async def predict(request: Request, tta: bool = False, grid: bool = False):
    """Classify one square frame. Body is raw JPEG/PNG bytes.

    tta   - average over horizontal and vertical flips (slower, steadier)
    grid  - also score a 3x3 tiling so the defect can be located
    """
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty request body.")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Body is not a decodable image.")

    started = time.perf_counter()
    # The verdict gets the expensive treatment; the tiles only need to say where,
    # so they skip flip-averaging - 12 forward passes instead of 30.
    frame = run_batch([to_tensor(img)], tta)[0]
    result = {"frame": summarise(frame)}
    if grid:
        tiles = run_batch([to_tensor(t) for t in tile_images(img, 3)], False)
        result["tiles"] = [{"row": i // 3, "col": i % 3, **summarise(tiles[i])} for i in range(9)]
    result["ms"] = round((time.perf_counter() - started) * 1000, 1)
    return JSONResponse(result)


# Mounted last so /api/* wins. Serving static/ at the root means the page uses
# the same relative paths here as it does on a static host, where static/ IS the root.
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")


# --------------------------------------------------------------------------
# Serving
# --------------------------------------------------------------------------

def lan_ip() -> str:
    """Best guess at this machine's address on the local network."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def self_signed_cert(host: str) -> tuple[str, str]:
    """Create (or reuse) a self-signed cert so phones on the LAN can use the
    camera - browsers only expose getUserMedia over HTTPS or on localhost."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    from datetime import datetime, timedelta, timezone

    certs = ROOT / ".certs"
    certs.mkdir(exist_ok=True)
    crt, key = certs / "dev.crt", certs / "dev.key"
    if crt.exists() and key.exists():
        return str(crt), str(key)

    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "fabric-inspection")])
    alts = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    for candidate in {host, lan_ip()}:
        try:
            alts.append(x509.IPAddress(ipaddress.ip_address(candidate)))
        except ValueError:
            pass
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(priv.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(alts), critical=False)
        .sign(priv, hashes.SHA256())
    )
    crt.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key.write_bytes(
        priv.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    return str(crt), str(key)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fabric defect inspection web app")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 to expose on the LAN)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--https", action="store_true", help="serve over HTTPS with a self-signed cert")
    ap.add_argument("--no-open", action="store_true", help="don't open a browser")
    args = ap.parse_args()

    import uvicorn

    ssl = {}
    scheme = "http"
    if args.https:
        crt, key = self_signed_cert(args.host)
        ssl = {"ssl_certfile": crt, "ssl_keyfile": key}
        scheme = "https"

    shown = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    url = f"{scheme}://{shown}:{args.port}/"
    print(f"\n  Fabric inspection  ->  {url}")
    if args.host == "0.0.0.0":
        print(f"  On this network    ->  {scheme}://{lan_ip()}:{args.port}/")
        if not args.https:
            print("  Note: other devices need --https; browsers block the camera on plain HTTP.")
    print()

    if not args.no_open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", **ssl)


if __name__ == "__main__":
    main()
