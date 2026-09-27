# Fabric Inspection

A browser-based fabric defect detector. Point a camera at cloth, get a classification per
frame — live or one capture at a time — with a session log you can export.

**The model runs on the device, in the browser.** Camera frames never leave the machine and
there is no inference server, so the whole thing deploys as static files.

Replaces the OpenCV desktop window in `webcam_demo.py`, which is kept as-is for reference.

## Run it locally

```bash
pip install -r requirements.txt
python app.py
```

Opens `http://localhost:8000/`. `app.py` serves `static/` at the root, which is the same
layout a static host sees, so local and deployed behave identically.

Useful flags:

| Flag | What it does |
| --- | --- |
| `--port 8000` | Port to serve on. |
| `--host 0.0.0.0` | Serve to other devices on the network. |
| `--https` | Self-signed TLS, so phones and tablets can use their camera. |
| `--no-open` | Don't launch a browser. |

Browsers only hand over a camera on `localhost` or over HTTPS. To inspect with a phone:

```bash
python app.py --host 0.0.0.0 --https
```

then open the printed `https://<your-ip>:8000/` address and accept the certificate warning.

## Deploy

The site is static — `static/` is the whole deployment. On Vercel it is already configured
by `vercel.json` (`outputDirectory: static`, no build step); push and it deploys.

**Do not let Vercel build `app.py`.** Vercel auto-detects FastAPI and will try to bundle it
into a serverless function, which pulls in PyTorch and its CUDA wheels — about 5.3 GB against
a 250 MB function limit, so the build fails. `.vercelignore` keeps the Python files out of
the upload to stop that detection. The same static output works on Netlify, Cloudflare Pages
or GitHub Pages.

If you ever do want the Python API deployed, it needs a container host (Render, Railway,
Fly.io, Hugging Face Spaces) — not a serverless function platform.

## Using it

- **Scan live** — runs continuously, smoothing results over time so the verdict doesn't
  flicker. Press `L`.
- **Capture** — one frame, always flip-averaged, added to the log. Press `Space`.
- **Locate 3×3** — also scores nine tiles of the aperture so you can see *where* in the
  window the defect reads. Press `G`.
- **Flip average** — averages the prediction over horizontal and vertical mirrors. Slower,
  steadier, and appropriate for woven fabric where a defect looks the same mirrored.
- **Full frame** — sends the whole visible frame instead of the centre aperture.
- **Call a defect above** — the confidence needed before the app commits to a verdict.
  Below it, the app says "Hold steady" rather than guessing.

The aperture reticle is not decoration: it is exactly the crop fed to the model. What falls
outside it is not looked at.

Measured on a laptop CPU through WebAssembly, single-threaded:

| Mode | Latency | Rate |
| --- | --- | --- |
| Plain | 21 ms | capped at 30/s (camera rate) |
| Flip average | 38 ms | ~18/s |
| Flip average + 3×3 locating | 127 ms | ~7.5/s |

The masthead shows `CPU` or `GPU` depending on whether WebGPU was available.

Captures export to CSV with the full probability vector per row, for auditing.

## The model

`fabric_model.pt` is a MobileNetV3-small, fine-tuned from ImageNet weights on six classes,
`['defect_free', 'hole', 'horizontal', 'lines', 'stain', 'verticle']` (order is fixed).
On held-out source photos it scores val macro-F1 0.76 (accuracy 0.84) and test macro-F1 0.76
(accuracy 0.91). `verticle` is the weakest class, and `horizontal` and `lines` are next;
more photos of line-type defects from the real camera would help most.

`app.py` and the browser both run an untrained-weights check at startup (logit spread across
flat and striped test plates must exceed 0.15) and show a banner if it fails.

### Retraining

The dataset archives go in the repo root; the generated crops go in `data/` (gitignored).

1. `python tools/build_dataset.py --out data/_all` crops every mapped box or polygon.
2. `python tools/split_dataset.py --src data/_all --out data` splits 80/15/5 by source photo.
3. `python tools/train.py --arch mobilenet_v3_small --classes defect_free hole horizontal lines stain verticle --epochs 15 --lr 5e-4`
   keeps the epoch with the best val macro-F1 in `data/fabric_model.pt`. Copy it over
   `fabric_model.pt`.
4. `python export_onnx.py` regenerates `static/fabric_model.onnx` and refuses to write a
   graph that drifts from PyTorch by more than 1e-5.

## Optional: the Python API

The browser no longer needs it, but `app.py` still exposes the model over HTTP for scripting
and batch work:

`POST /api/predict?tta=0|1&grid=0|1` — body is raw JPEG or PNG bytes:

```json
{
  "frame": { "index": 3, "name": "lines", "label": "Streaks",
             "prob": 0.188, "defect": 0.842, "probs": [...] },
  "tiles": [ { "row": 0, "col": 0, "...": "same shape as frame" } ],
  "ms": 21.2
}
```

`GET /api/meta` — class list, input size, and the startup health check.

## Layout

```
static/index.html        page
static/styles.css        design tokens and layout
static/app.js            camera, inference, readout, capture log
static/fabric_model.onnx the deployed model (25.8 MB)
app.py                   local dev server + optional REST API
export_onnx.py           fabric_model.pt -> static/fabric_model.onnx, with a parity check
vercel.json              static deploy config
webcam_demo.py           the original OpenCV script, untouched
fabric_model.pt          PyTorch weights, gitignored (currently untrained - see above)
```
