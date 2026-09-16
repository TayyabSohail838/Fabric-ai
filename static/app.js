/* Fabric inspection - camera capture, live scanning, readout.
   The model runs on this device via onnxruntime-web. No frames leave the browser. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const feed = $("feed");
  const viewport = $("viewport");
  const aperture = $("aperture");
  const tileGrid = $("tile-grid");
  const note = $("viewport-note");
  const freeze = $("freeze");

  const btnLive = $("btn-live");
  const btnLiveText = $("btn-live-text");
  const btnCapture = $("btn-capture");
  const btnExport = $("btn-export");
  const btnClear = $("btn-clear");
  const optGrid = $("opt-grid");
  const optTta = $("opt-tta");
  const optFull = $("opt-full");
  const optThreshold = $("opt-threshold");
  const thresholdOut = $("threshold-out");
  const cameraSelect = $("camera-select");

  const verdict = $("verdict");
  const vEyebrow = $("verdict-eyebrow");
  const vLabel = $("verdict-label");
  const vHint = $("verdict-hint");
  const vFill = $("verdict-fill");
  const vConf = $("verdict-conf");
  const vRaw = $("verdict-raw");
  const barsEl = $("bars");
  const logStrip = $("log-strip");
  const logEmpty = $("log-empty");

  // Raw class order is fixed by training - must match app.py.
  const CLASSES = [
    { name: "defect_free", label: "Clean", hint: "No defect in the aperture.", clean: true },
    { name: "hole", label: "Hole", hint: "Broken ends or picks - a gap in the weave." },
    { name: "horizontal", label: "Weft bar", hint: "A band running across the weft." },
    { name: "lines", label: "Streaks", hint: "Irregular streaking over the surface." },
    { name: "stain", label: "Stain", hint: "Discolouration or soil on the face." },
    { name: "verticle", label: "Warp line", hint: "A line running along the warp." },
  ];
  const CLEAN = 0;
  const SIDE = 224;
  const PIXELS = SIDE * SIDE;
  const MEAN = [0.485, 0.456, 0.406];
  const STD = [0.229, 0.224, 0.225];

  const APERTURE_FRACTION = 0.68; // of the shorter side of the viewport
  const SMOOTHING = 0.4;          // EMA weight on the newest live frame
  const MIN_GAP_MS = 33;          // no point scanning faster than the camera delivers

  let session = null;
  let stream = null;
  let currentDeviceId = null;
  let rebuilding = false;
  let live = false;
  let busy = false;
  let smoothed = null;
  let timer = null;
  let roundTrips = [];
  const shots = [];

  // One scratch canvas per job so the tile loop never fights the thumbnail.
  const work = document.createElement("canvas");
  work.width = work.height = SIDE;
  const workCtx = work.getContext("2d", { willReadFrequently: true });
  const thumbCanvas = document.createElement("canvas");
  thumbCanvas.width = thumbCanvas.height = 256;
  const thumbCtx = thumbCanvas.getContext("2d");

  /* ── Model ──────────────────────────────────────────────────── */

  async function loadModel() {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    ort.env.wasm.simd = true;
    // Threads need cross-origin isolation; without those headers this must stay 1.
    ort.env.wasm.numThreads = self.crossOriginIsolated
      ? Math.min(4, navigator.hardwareConcurrency || 1)
      : 1;

    const open = (ep) =>
      ort.InferenceSession.create("fabric_model.onnx", {
        executionProviders: [ep],
        graphOptimizationLevel: "all",
      });

    if ("gpu" in navigator) {
      try {
        return { session: await open("webgpu"), backend: "GPU" };
      } catch {
        /* fall through to wasm */
      }
    }
    const threads = ort.env.wasm.numThreads;
    return { session: await open("wasm"), backend: threads > 1 ? `CPU ×${threads}` : "CPU" };
  }

  // Whatever is on the work canvas -> normalised CHW floats, written at `offset`.
  function readInto(buffer, offset) {
    const { data } = workCtx.getImageData(0, 0, SIDE, SIDE);
    const r = offset, g = offset + PIXELS, b = offset + 2 * PIXELS;
    for (let i = 0; i < PIXELS; i++) {
      const p = i * 4;
      buffer[r + i] = (data[p] / 255 - MEAN[0]) / STD[0];
      buffer[g + i] = (data[p + 1] / 255 - MEAN[1]) / STD[1];
      buffer[b + i] = (data[p + 2] / 255 - MEAN[2]) / STD[2];
    }
  }

  // Mirror one image in place. Flipping the tensor is exact and cheaper than
  // re-drawing the frame through a transformed canvas.
  function mirror(src, offset, axis) {
    const out = new Float32Array(3 * PIXELS);
    for (let c = 0; c < 3; c++) {
      const base = offset + c * PIXELS;
      const dst = c * PIXELS;
      for (let y = 0; y < SIDE; y++) {
        const sy = axis === "v" ? SIDE - 1 - y : y;
        for (let x = 0; x < SIDE; x++) {
          const sx = axis === "h" ? SIDE - 1 - x : x;
          out[dst + y * SIDE + x] = src[base + sy * SIDE + sx];
        }
      }
    }
    return out;
  }

  function softmax(logits, offset) {
    let max = -Infinity;
    for (let i = 0; i < CLASSES.length; i++) max = Math.max(max, logits[offset + i]);
    let sum = 0;
    const out = new Array(CLASSES.length);
    for (let i = 0; i < CLASSES.length; i++) {
      out[i] = Math.exp(logits[offset + i] - max);
      sum += out[i];
    }
    return out.map((p) => p / sum);
  }

  async function runBatch(buffer, count) {
    const tensor = new ort.Tensor("float32", buffer, [count, 3, SIDE, SIDE]);
    const out = await session.run({ input: tensor });
    return out.logits.data;
  }

  /* ── Class bars ─────────────────────────────────────────────── */

  function buildBars() {
    barsEl.innerHTML = "";
    CLASSES.forEach((c, i) => {
      const li = document.createElement("li");
      li.className = "bar";
      li.dataset.clean = c.clean ? "1" : "0";
      li.innerHTML =
        `<span class="bar__name"></span><span class="bar__val">0.0%</span>` +
        `<span class="bar__track"><span class="bar__fill"></span></span>`;
      li.querySelector(".bar__name").textContent = c.label;
      li.title = `${c.hint}  (${c.name})`;
      li.id = "bar-" + i;
      barsEl.appendChild(li);
    });
  }

  function paintBars(probs, leadIndex) {
    probs.forEach((p, i) => {
      const li = $("bar-" + i);
      if (!li) return;
      li.dataset.lead = i === leadIndex ? "1" : "0";
      li.querySelector(".bar__val").textContent = (p * 100).toFixed(1) + "%";
      li.querySelector(".bar__fill").style.width = (p * 100).toFixed(1) + "%";
    });
  }

  /* ── Verdict ────────────────────────────────────────────────── */

  const threshold = () => Number(optThreshold.value) / 100;

  function worstDefect(probs) {
    let worst = -1;
    probs.forEach((p, i) => {
      if (i !== CLEAN && (worst < 0 || p > probs[worst])) worst = i;
    });
    return worst;
  }

  function render(probs) {
    const defectProb = 1 - probs[CLEAN];
    const t = threshold();
    const worst = worstDefect(probs);

    let state, eyebrow, label, hint, meter, conf, raw;
    if (defectProb >= t) {
      state = "defect";
      eyebrow = "Defect";
      label = CLASSES[worst].label;
      hint = CLASSES[worst].hint;
      meter = conf = defectProb;
      raw = CLASSES[worst].name;
    } else if (probs[CLEAN] >= t) {
      state = "clean";
      eyebrow = "Pass";
      label = "Clean";
      hint = "No defect in the aperture. Move the cloth to inspect the next section.";
      meter = conf = probs[CLEAN];
      raw = CLASSES[CLEAN].name;
    } else {
      state = "uncertain";
      eyebrow = "Not sure";
      label = "Hold steady";
      hint = `Closest call is ${CLASSES[worst].label} at ${(defectProb * 100).toFixed(0)}%. ` +
             `Fill the aperture with cloth and hold still, or lower the threshold.`;
      meter = conf = defectProb;
      raw = "below threshold";
    }

    verdict.dataset.state = state;
    aperture.dataset.state = state === "defect" ? "defect" : live ? "live" : state;
    vEyebrow.textContent = eyebrow;
    vLabel.textContent = label;
    vHint.textContent = hint;
    vFill.style.width = (meter * 100).toFixed(1) + "%";
    vConf.textContent = (conf * 100).toFixed(1) + "%";
    vRaw.textContent = raw;

    let lead = 0;
    probs.forEach((p, i) => { if (p > probs[lead]) lead = i; });
    paintBars(probs, lead);
  }

  function paintTiles(tiles) {
    if (!tiles) return;
    const t = threshold();
    // Tint every tile that reads defective, but name only the worst one -
    // nine identical labels say less than one.
    let worstTile = -1;
    tiles.forEach((tile, i) => {
      if (tile.defect >= t && (worstTile < 0 || tile.defect > tiles[worstTile].defect)) worstTile = i;
    });
    tiles.forEach((tile, i) => {
      const cell = tileGrid.children[i];
      if (!cell) return;
      const hit = tile.defect >= t;
      const weight = Math.max(0, Math.min(1, (tile.defect - 0.25) / 0.75));
      cell.style.backgroundColor = hit
        ? `rgba(168,51,31,${(0.18 + weight * 0.34).toFixed(2)})`
        : "transparent";
      cell.dataset.hit = hit ? "1" : "0";
      cell.firstChild.textContent = i === worstTile ? CLASSES[tile.top].label : "";
    });
  }

  function buildTiles() {
    tileGrid.innerHTML = "";
    for (let i = 0; i < 9; i++) {
      const cell = document.createElement("div");
      cell.className = "tile";
      cell.appendChild(document.createElement("span"));
      tileGrid.appendChild(cell);
    }
  }

  /* ── Aperture geometry ──────────────────────────────────────── */

  // Where the reticle sits on screen, and which source pixels that is.
  function geometry() {
    const W = viewport.clientWidth;
    const H = viewport.clientHeight;
    const vw = feed.videoWidth;
    const vh = feed.videoHeight;

    let box;
    if (optFull.checked) {
      box = { left: 0, top: 0, w: W, h: H };
    } else {
      const side = Math.round(Math.min(W, H) * APERTURE_FRACTION);
      box = { left: Math.round((W - side) / 2), top: Math.round((H - side) / 2), w: side, h: side };
    }
    if (!vw || !vh) return { box, src: null };

    // object-fit: cover - the video is scaled up until it covers the box.
    const scale = Math.max(W / vw, H / vh);
    const ox = (W - vw * scale) / 2;
    const oy = (H - vh * scale) / 2;
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    const sx = clamp((box.left - ox) / scale, 0, vw);
    const sy = clamp((box.top - oy) / scale, 0, vh);
    const sw = clamp(box.w / scale, 1, vw - sx);
    const sh = clamp(box.h / scale, 1, vh - sy);
    return { box, src: { sx, sy, sw, sh } };
  }

  function placeAperture() {
    const { box } = geometry();
    aperture.style.left = box.left + "px";
    aperture.style.top = box.top + "px";
    aperture.style.width = box.w + "px";
    aperture.style.height = box.h + "px";
    aperture.style.setProperty("--side", box.h + "px");
  }

  /* ── Inference on the current frame ─────────────────────────── */

  async function scan({ tta, grid }) {
    const { src } = geometry();
    if (!src || !session) return null;

    const started = performance.now();
    const views = 1 + (tta ? 2 : 0) + (grid ? 9 : 0);
    const buffer = new Float32Array(views * 3 * PIXELS);

    workCtx.drawImage(feed, src.sx, src.sy, src.sw, src.sh, 0, 0, SIDE, SIDE);
    readInto(buffer, 0);

    let cursor = 3 * PIXELS;
    if (tta) {
      buffer.set(mirror(buffer, 0, "h"), cursor); cursor += 3 * PIXELS;
      buffer.set(mirror(buffer, 0, "v"), cursor); cursor += 3 * PIXELS;
    }
    if (grid) {
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
          workCtx.drawImage(
            feed,
            src.sx + (c * src.sw) / 3, src.sy + (r * src.sh) / 3, src.sw / 3, src.sh / 3,
            0, 0, SIDE, SIDE
          );
          readInto(buffer, cursor);
          cursor += 3 * PIXELS;
        }
      }
    }

    const logits = await runBatch(buffer, views);

    // The verdict averages its mirrors; tiles only need to say where.
    let probs = softmax(logits, 0);
    if (tta) {
      const a = softmax(logits, CLASSES.length);
      const b = softmax(logits, 2 * CLASSES.length);
      probs = probs.map((p, i) => (p + a[i] + b[i]) / 3);
    }

    let tiles = null;
    if (grid) {
      const base = (1 + (tta ? 2 : 0)) * CLASSES.length;
      tiles = [];
      for (let i = 0; i < 9; i++) {
        const p = softmax(logits, base + i * CLASSES.length);
        tiles.push({ probs: p, defect: 1 - p[CLEAN], top: worstDefect(p) });
      }
    }
    return { probs, tiles, ms: performance.now() - started };
  }

  function trackRate(ms) {
    roundTrips.push(ms);
    if (roundTrips.length > 12) roundTrips.shift();
    const avg = roundTrips.reduce((a, b) => a + b, 0) / roundTrips.length;
    $("tally-fps").textContent = live ? (1000 / avg).toFixed(1) + " /s" : "—";
  }

  /* ── Live scanning ──────────────────────────────────────────── */

  async function tick() {
    if (!live || busy) return;
    busy = true;
    const started = performance.now();
    try {
      const out = await scan({ tta: optTta.checked, grid: optGrid.checked });
      if (!out) return schedule(200);
      smoothed = smoothed
        ? smoothed.map((p, i) => p * (1 - SMOOTHING) + out.probs[i] * SMOOTHING)
        : out.probs.slice();
      render(smoothed);
      paintTiles(out.tiles);
      $("spec-latency").textContent = out.ms.toFixed(0) + " ms";
      trackRate(performance.now() - started);
    } catch (err) {
      stopLive();
      showNote("Inference failed: " + err.message, true);
    } finally {
      busy = false;
      schedule(MIN_GAP_MS);
    }
  }

  function schedule(delay) {
    clearTimeout(timer);
    if (live) timer = setTimeout(tick, delay);
  }

  function startLive() {
    if (!stream || !session) return;
    live = true;
    smoothed = null;
    roundTrips = [];
    btnLive.setAttribute("aria-pressed", "true");
    btnLiveText.textContent = "Hold";
    aperture.dataset.state = "live";
    freeze.hidden = true;
    schedule(0);
  }

  function stopLive() {
    live = false;
    clearTimeout(timer);
    btnLive.setAttribute("aria-pressed", "false");
    btnLiveText.textContent = "Scan live";
    if (aperture.dataset.state === "live") aperture.dataset.state = "idle";
    $("tally-fps").textContent = "—";
  }

  /* ── Capture ────────────────────────────────────────────────── */

  async function capture() {
    if (!stream || !session) return;
    const { src } = geometry();
    if (!src) return;

    thumbCtx.drawImage(feed, src.sx, src.sy, src.sw, src.sh, 0, 0, 256, 256);
    const thumb = thumbCanvas.toDataURL("image/jpeg", 0.72);
    freeze.src = thumb;
    freeze.hidden = false;
    setTimeout(() => { if (live) freeze.hidden = true; }, 260);

    btnCapture.disabled = true;
    try {
      // Captures always flip-average, whatever the live setting says.
      const out = await scan({ tta: true, grid: optGrid.checked });
      if (out) {
        smoothed = out.probs.slice();
        render(out.probs);
        paintTiles(out.tiles);
        $("spec-latency").textContent = out.ms.toFixed(0) + " ms";
        addShot(thumb, out.probs);
      }
    } catch (err) {
      showNote("Capture failed: " + err.message, true);
    } finally {
      btnCapture.disabled = false;
      if (!live) setTimeout(() => { freeze.hidden = true; }, 600);
    }
  }

  function addShot(thumb, probs) {
    const defectProb = 1 - probs[CLEAN];
    const isDefect = defectProb >= threshold();
    const shown = isDefect ? CLASSES[worstDefect(probs)] : CLASSES[CLEAN];
    const conf = isDefect ? defectProb : probs[CLEAN];
    const at = new Date();

    shots.unshift({ at: at.toISOString(), label: shown.label, name: shown.name, conf, probs, defect: isDefect });

    const li = document.createElement("li");
    li.className = "shot";
    li.dataset.defect = isDefect ? "1" : "0";
    li.innerHTML =
      `<img alt="Capture at ${at.toLocaleTimeString()}">` +
      `<div class="shot__meta"><div class="shot__class"></div><span class="shot__sub"></span></div>`;
    li.querySelector("img").src = thumb;
    li.querySelector(".shot__class").textContent = shown.label;
    li.querySelector(".shot__sub").textContent =
      `${(conf * 100).toFixed(0)}% · ${at.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`;
    li.title = CLASSES.map((c, i) => `${c.label}: ${(probs[i] * 100).toFixed(1)}%`).join("\n");

    logEmpty.hidden = true;
    logStrip.insertBefore(li, logStrip.firstChild);
    btnExport.disabled = false;
    btnClear.disabled = false;
    updateTally();
  }

  function updateTally() {
    const total = shots.length;
    const defects = shots.filter((s) => s.defect).length;
    $("tally-total").textContent = String(total);
    $("tally-defects").textContent = String(defects);
    $("tally-rate").textContent = total ? ((defects / total) * 100).toFixed(0) + "%" : "—";
  }

  function exportCsv() {
    const head = ["captured_at", "verdict", "class", "confidence"].concat(CLASSES.map((c) => "p_" + c.name));
    const rows = shots.slice().reverse().map((s) =>
      [s.at, s.defect ? "defect" : "pass", s.name, s.conf.toFixed(4)]
        .concat(s.probs.map((p) => p.toFixed(4)))
        .join(",")
    );
    const blob = new Blob([[head.join(","), ...rows].join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `fabric-inspection-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "")}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function clearLog() {
    shots.length = 0;
    logStrip.querySelectorAll(".shot").forEach((n) => n.remove());
    logEmpty.hidden = false;
    btnExport.disabled = true;
    btnClear.disabled = true;
    updateTally();
  }

  /* ── Camera ─────────────────────────────────────────────────── */

  function showNote(text, isError) {
    note.textContent = text;
    note.hidden = false;
    note.classList.toggle("viewport__note--error", !!isError);
  }
  function hideNote() { note.hidden = true; }

  async function listCameras() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter((d) => d.kind === "videoinput");
    // Replacing the options changes the select's value, which some browsers
    // report as a change event. Ignore our own rebuild or we restart the
    // camera - and kill the live loop - every time the list refreshes.
    rebuilding = true;
    cameraSelect.innerHTML = "";
    cams.forEach((cam, i) => {
      const opt = document.createElement("option");
      opt.value = cam.deviceId;
      opt.textContent = cam.label || `Camera ${i + 1}`;
      if (cam.deviceId === currentDeviceId) opt.selected = true;
      cameraSelect.appendChild(opt);
    });
    cameraSelect.disabled = cams.length < 2;
    rebuilding = false;
  }

  async function startCamera(deviceId) {
    stopLive();
    if (stream) stream.getTracks().forEach((t) => t.stop());
    const video = deviceId
      ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } };
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
    } catch (err) {
      stream = null;
      const msg =
        err.name === "NotAllowedError" ? "Camera blocked. Allow camera access for this site, then reload."
        : err.name === "NotFoundError" ? "No camera found. Connect one and reload."
        : err.name === "NotReadableError" ? "Camera is busy. Close Teams, Zoom or any other tab using it, then reload."
        : "Could not start the camera: " + err.message;
      showNote(msg, true);
      btnLive.disabled = true;
      btnCapture.disabled = true;
      return;
    }
    feed.srcObject = stream;
    await feed.play().catch(() => {});
    btnLive.disabled = false;
    btnCapture.disabled = false;

    const track = stream.getVideoTracks()[0];
    const s = track.getSettings ? track.getSettings() : {};
    currentDeviceId = s.deviceId || deviceId || null;
    const size = s.width && s.height ? `${s.width}×${s.height}. ` : "";
    showNote(`Point the aperture at the cloth. ${size}Space captures, L scans live.`);
    setTimeout(hideNote, 4200);
    placeAperture();
    await listCameras();
  }

  /* ── Wiring ─────────────────────────────────────────────────── */

  btnLive.addEventListener("click", () => (live ? stopLive() : startLive()));
  btnCapture.addEventListener("click", capture);
  btnExport.addEventListener("click", exportCsv);
  btnClear.addEventListener("click", clearLog);
  cameraSelect.addEventListener("change", () => {
    if (rebuilding || cameraSelect.value === currentDeviceId) return;
    startCamera(cameraSelect.value);
  });

  optGrid.addEventListener("change", () => {
    tileGrid.hidden = !optGrid.checked;
    if (!optGrid.checked) {
      [...tileGrid.children].forEach((c) => {
        c.style.backgroundColor = "transparent";
        c.dataset.hit = "0";
      });
    }
  });
  optFull.addEventListener("change", () => { placeAperture(); smoothed = null; });
  optThreshold.addEventListener("input", () => {
    thresholdOut.textContent = optThreshold.value + "%";
    if (smoothed) render(smoothed);
  });

  window.addEventListener("resize", placeAperture);
  feed.addEventListener("loadedmetadata", placeAperture);

  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input,select,textarea")) return;
    if (e.code === "Space") { e.preventDefault(); capture(); }
    else if (e.key === "l" || e.key === "L") { live ? stopLive() : startLive(); }
    else if (e.key === "g" || e.key === "G") { optGrid.checked = !optGrid.checked; optGrid.dispatchEvent(new Event("change")); }
  });

  document.addEventListener("visibilitychange", () => {
    if (document.hidden && live) stopLive();
  });

  /* ── Is this model actually trained? ────────────────────────── */

  // A net saved before training returns the same logits for every image, so the
  // app would show confident labels that mean nothing. Push very different
  // plates through and measure how far the logits travel. Mirrors app.py.
  async function probeModel() {
    const plates = [[0, 0, 0], [255, 255, 255], [128, 128, 128]];
    const buffer = new Float32Array(5 * 3 * PIXELS);
    plates.forEach((rgb, i) => {
      workCtx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
      workCtx.fillRect(0, 0, SIDE, SIDE);
      readInto(buffer, i * 3 * PIXELS);
    });
    [["v", 3], ["h", 4]].forEach(([axis, slot]) => {
      workCtx.fillStyle = "#000";
      workCtx.fillRect(0, 0, SIDE, SIDE);
      workCtx.fillStyle = "#fff";
      for (let k = 0; k < SIDE; k += 16) {
        if (axis === "v") workCtx.fillRect(k, 0, 1, SIDE);
        else workCtx.fillRect(0, k, SIDE, 1);
      }
      readInto(buffer, slot * 3 * PIXELS);
    });

    const logits = await runBatch(buffer, 5);
    const n = CLASSES.length;
    let spread = 0;
    for (let c = 0; c < n; c++) {
      let mean = 0;
      for (let i = 0; i < 5; i++) mean += logits[i * n + c];
      mean /= 5;
      let variance = 0;
      for (let i = 0; i < 5; i++) variance += (logits[i * n + c] - mean) ** 2;
      spread += Math.sqrt(variance / 5);
    }
    spread /= n;
    return { spread, trained: spread > 0.15 };
  }

  /* ── Boot ───────────────────────────────────────────────────── */

  (async function boot() {
    buildTiles();
    buildBars();
    paintBars(CLASSES.map(() => 0), -1);
    thresholdOut.textContent = optThreshold.value + "%";

    showNote("Loading the model… (26 MB, cached after the first visit)");
    try {
      const loaded = await loadModel();
      session = loaded.session;
      $("spec-backend").textContent = loaded.backend;
    } catch (err) {
      showNote("Could not load fabric_model.onnx: " + err.message, true);
      return;
    }

    const health = await probeModel();
    if (!health.trained) {
      $("alert-body").textContent =
        "Weights match an untrained initialisation - every frame returns the same prediction " +
        "regardless of what the camera sees. Retrain the model and replace fabric_model.pt to " +
        `get real results. Logit spread across black, white, grey and striped test plates is ` +
        `${health.spread.toFixed(4)} — a trained network moves by whole units. Everything ` +
        "below still works; the numbers just do not mean anything yet.";
      $("alert").hidden = false;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showNote("This browser will not share a camera over an insecure connection. Open the app on localhost or over HTTPS.", true);
      return;
    }
    await startCamera(null);
  })();
})();
