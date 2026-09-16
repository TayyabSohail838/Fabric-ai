# Fabric Inspection

A browser-based version of the fabric defect detector. Point a camera at cloth, get a
classification per frame — live or one capture at a time — with a session log you can export.

Replaces the OpenCV desktop window in `webcam_demo.py`, which is kept as-is for reference.

## Run it

```bash
pip install -r requirements.txt
python app.py
```

It opens `http://localhost:8000/`. Nothing else is needed — the page is served by the same
process that holds the model.

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

## Using it

- **Scan live** — runs continuously, smoothing results over time so the verdict doesn't
  flicker. Press `L`. Roughly 8–10 frames/second on CPU; 5–7/second with flip-averaging and
  3×3 locating both on.
- **Capture** — one high-quality frame, always flip-averaged, added to the log. Press `Space`.
- **Locate 3×3** — also scores nine tiles of the aperture so you can see *where* in the
  window the defect reads. Press `G`.
- **Flip average** — averages the prediction over horizontal and vertical mirrors. Slower,
  steadier, and appropriate for woven fabric where a defect looks the same mirrored.
- **Full frame** — sends the whole visible frame instead of the centre aperture.
- **Call a defect above** — the confidence needed before the app commits to a verdict.
  Below it, the app says "Hold steady" rather than guessing.

The aperture reticle is not decoration: it is exactly the crop sent to the model. What falls
outside it is not looked at.

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

`app.py` checks this at startup and the interface shows a banner, so the app never presents
meaningless numbers as if they were real. `webcam_demo.py` has the same problem silently.

Everything else — camera, preprocessing, batching, tiling, serving — is correct and will
produce real results the moment trained weights are dropped in. To fix it, train
`BaselineCNN` on your six-class dataset and overwrite `fabric_model.pt` with
`torch.save(model.state_dict(), "fabric_model.pt")`. Keep the class order:

```
['defect_free', 'hole', 'horizontal', 'lines', 'stain', 'verticle']
```

## API

`POST /api/predict?tta=0|1&grid=0|1` — body is raw JPEG or PNG bytes, response is JSON:

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
app.py              model, preprocessing, HTTP API, server
static/index.html   page
static/styles.css   design tokens and layout
static/app.js       camera, live loop, readout, capture log
webcam_demo.py      the original OpenCV script, untouched
fabric_model.pt     weights (currently untrained - see above)
```
