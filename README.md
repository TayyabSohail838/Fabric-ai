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

## Heads-up: the shipped model is untrained

`fabric_model.pt` contains a network that was saved **before it was trained**:

- Every weight tensor matches PyTorch's default `kaiming_uniform_` initialisation bounds
  exactly, and the `fc1` weight histogram is perfectly flat across that range. Training
  deforms that distribution; nothing here is deformed.
- Feeding it black, white, grey, vertical stripes and horizontal stripes moves the logits by
  a standard deviation of ~0.015. A trained network moves by whole units.
- The output is therefore just `softmax(fc2.bias)` — a fixed ~19% on `lines` for every
  image, no matter what the camera sees.

Both `app.py` and the browser run that same check at startup — and independently measure the
same 0.0151 — so the interface shows a banner instead of presenting meaningless numbers as
real. `webcam_demo.py` has the same problem silently.

Everything else — camera, preprocessing, batching, tiling — is correct and will produce real
results the moment trained weights are dropped in. To fix it:

1. Train `BaselineCNN` on your six-class dataset and overwrite `fabric_model.pt` with
   `torch.save(model.state_dict(), "fabric_model.pt")`. Keep the class order:
   `['defect_free', 'hole', 'horizontal', 'lines', 'stain', 'verticle']`
2. Run `python export_onnx.py` to regenerate `static/fabric_model.onnx`. It verifies the
   exported graph matches PyTorch to 1e-7 and refuses to write a model that doesn't.
3. Commit and push. The banner disappears on its own once the logit spread clears 0.15.

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
