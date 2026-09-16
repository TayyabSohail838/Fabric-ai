"""Export fabric_model.pt to ONNX so the browser can run it directly.

The web app loads static/fabric_model.onnx through onnxruntime-web, which means
no Python at runtime - the site deploys as static files and inference happens on
the device. Re-run this after retraining:

    python export_onnx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from app import CLASS_NAMES, BaselineCNN, MEAN, STD

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "fabric_model.pt"
OUT = ROOT / "static" / "fabric_model.onnx"


def main() -> None:
    if not WEIGHTS.exists():
        sys.exit(f"Weights not found: {WEIGHTS}")

    model = BaselineCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()

    # Raw logits, not probabilities: the browser applies softmax itself, and the
    # untrained-model check measures logit spread exactly as app.py does.
    sample = torch.randn(1, 3, 224, 224)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (sample,),
        str(OUT),
        input_names=["input"],
        output_names=["logits"],
        # Tiles are scored as one batch, so batch has to stay dynamic.
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    # The exported graph has to agree with PyTorch, or the app lies quietly.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    probe = torch.randn(5, 3, 224, 224)
    with torch.no_grad():
        expected = model(probe).numpy()
    got = sess.run(["logits"], {"input": probe.numpy()})[0]

    drift = float(np.abs(expected - got).max())
    size_mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT.relative_to(ROOT)}  ({size_mb:.1f} MB)")
    print(f"batch shapes: torch {tuple(expected.shape)} == onnx {tuple(got.shape)}")
    print(f"max logit drift vs pytorch: {drift:.3e}")
    if drift > 1e-5:
        sys.exit("ONNX output does not match PyTorch - not safe to ship.")
    print("normalisation for the browser:")
    print(f"  mean {MEAN.ravel().tolist()}")
    print(f"  std  {STD.ravel().tolist()}")
    print("OK")


if __name__ == "__main__":
    main()
