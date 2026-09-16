/* Fabric inspection - camera capture, live scanning, readout. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const feed = $("feed");
  const viewport = $("viewport");
  const aperture = $("aperture");
  const tileGrid = $("tile-grid");
  const note = $("viewport-note");
  const freeze = $("freeze");
  const grab = $("grab");
  const ctx = grab.getContext("2d", { willReadFrequently: false });

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

  const APERTURE_FRACTION = 0.68; // of the shorter side of the viewport
  const SMOOTHING = 0.4;          // EMA weight on the newest live frame
  const MIN_GAP_MS = 90;          // never hammer the backend faster than this

  let classes = [];
  let cleanIndex = 0;
  let stream = null;
  let currentDeviceId = null;
  let rebuilding = false;
  let live = false;
  let inFlight = false;
  let smoothed = null;
  let timer = null;
  let roundTrips = [];
  const shots = [];

  /* ── Class bars ─────────────────────────────────────────────── */

  function buildBars() {
    barsEl.innerHTML = "";
    classes.forEach((c, i) => {
      const li = document.createElement("li");
      li.className = "bar";
      li.dataset.clean = c.clean ? "1" : "0";
      li.innerHTML =
        `<span class="bar__name"></span><span class="bar__val">0.0%</span>` +
        `<span class="bar__track"><span class="bar__fill"></span></span>`;
      li.querySelector(".bar__name").textContent = c.label;
      li.title = c.hint + "  (" + c.name + ")";
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

  function threshold() {
    return Number(optThreshold.value) / 100;
  }

  function render(probs) {
    const defectProb = 1 - probs[cleanIndex];
    const t = threshold();

    // Strongest defect class, regardless of whether clean leads.
    let worst = -1;
    probs.forEach((p, i) => {
      if (i !== cleanIndex && (worst < 0 || p > probs[worst])) worst = i;
    });

    let state, eyebrow, label, hint, meter, conf, raw;
    if (defectProb >= t) {
      state = "defect";
      eyebrow = "Defect";
      label = classes[worst].label;
      hint = classes[worst].hint;
      meter = defectProb;
      conf = defectProb;
      raw = classes[worst].name;
    } else if (probs[cleanIndex] >= t) {
      state = "clean";
      eyebrow = "Pass";
      label = "Clean";
      hint = "No defect in the aperture. Move the cloth to inspect the next section.";
      meter = probs[cleanIndex];
      conf = probs[cleanIndex];
      raw = classes[cleanIndex].name;
    } else {
      state = "uncertain";
      eyebrow = "Not sure";
      label = "Hold steady";
      hint = `Closest call is ${classes[worst].label} at ${(defectProb * 100).toFixed(0)}%. ` +
             `Fill the aperture with cloth and hold still, or lower the threshold.`;
      meter = defectProb;
      conf = defectProb;
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
      cell.firstChild.textContent = i === worstTile ? tile.label : "";
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

  /* ── Frame grab and inference ───────────────────────────────── */

  function grabFrame(size) {
    const { src } = geometry();
    if (!src) return null;
    grab.width = size;
    grab.height = size;
    ctx.drawImage(feed, src.sx, src.sy, src.sw, src.sh, 0, 0, size, size);
    return grab;
  }

  function toBlob(canvas, quality) {
    return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
  }

  async function infer(blob, { tta, grid }) {
    const res = await fetch(`/api/predict?tta=${tta ? 1 : 0}&grid=${grid ? 1 : 0}`, {
      method: "POST",
      headers: { "Content-Type": "image/jpeg" },
      body: blob,
    });
    if (!res.ok) throw new Error(`Backend returned ${res.status}`);
    return res.json();
  }

  function trackRate(ms) {
    roundTrips.push(ms);
    if (roundTrips.length > 12) roundTrips.shift();
    const avg = roundTrips.reduce((a, b) => a + b, 0) / roundTrips.length;
    $("tally-fps").textContent = live ? (1000 / avg).toFixed(1) + " /s" : "—";
  }

  /* ── Live scanning ──────────────────────────────────────────── */

  async function tick() {
    if (!live || inFlight) return;
    const grid = optGrid.checked;
    const canvas = grabFrame(grid ? 672 : 320);
    if (!canvas) return schedule(200);

    inFlight = true;
    const started = performance.now();
    try {
      const blob = await toBlob(canvas, 0.8);
      const out = await infer(blob, { tta: optTta.checked, grid });
      smoothed = smoothed
        ? smoothed.map((p, i) => p * (1 - SMOOTHING) + out.frame.probs[i] * SMOOTHING)
        : out.frame.probs.slice();
      render(smoothed);
      paintTiles(out.tiles);
      $("spec-latency").textContent = out.ms.toFixed(0) + " ms";
      trackRate(performance.now() - started);
      hideNote();
    } catch (err) {
      stopLive();
      showNote("Lost the backend: " + err.message + ". Is app.py still running?", true);
    } finally {
      inFlight = false;
      schedule(MIN_GAP_MS);
    }
  }

  function schedule(delay) {
    clearTimeout(timer);
    if (live) timer = setTimeout(tick, delay);
  }

  function startLive() {
    if (!stream) return;
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
    if (!stream) return;
    const grid = optGrid.checked;
    const canvas = grabFrame(grid ? 672 : 448);
    if (!canvas) return;

    const thumb = canvas.toDataURL("image/jpeg", 0.72);
    freeze.src = thumb;
    freeze.hidden = false;
    setTimeout(() => { if (live) freeze.hidden = true; }, 260);

    btnCapture.disabled = true;
    try {
      const blob = await toBlob(canvas, 0.92);
      const out = await infer(blob, { tta: true, grid }); // captures always flip-average
      smoothed = out.frame.probs.slice();
      render(out.frame.probs);
      paintTiles(out.tiles);
      $("spec-latency").textContent = out.ms.toFixed(0) + " ms";
      addShot(thumb, out.frame);
      hideNote();
    } catch (err) {
      showNote("Capture failed: " + err.message, true);
    } finally {
      btnCapture.disabled = false;
      if (!live) setTimeout(() => { freeze.hidden = true; }, 600);
    }
  }

  function addShot(thumb, frame) {
    const defectProb = 1 - frame.probs[cleanIndex];
    const isDefect = defectProb >= threshold();
    let worst = -1;
    frame.probs.forEach((p, i) => {
      if (i !== cleanIndex && (worst < 0 || p > frame.probs[worst])) worst = i;
    });
    const shown = isDefect ? classes[worst] : classes[cleanIndex];
    const conf = isDefect ? defectProb : frame.probs[cleanIndex];
    const at = new Date();

    shots.unshift({
      at: at.toISOString(),
      label: shown.label,
      name: shown.name,
      conf,
      probs: frame.probs,
      defect: isDefect,
    });

    const li = document.createElement("li");
    li.className = "shot";
    li.dataset.defect = isDefect ? "1" : "0";
    li.innerHTML =
      `<img alt="Capture at ${at.toLocaleTimeString()}">` +
      `<div class="shot__meta"><div class="shot__class"></div>` +
      `<span class="shot__sub"></span></div>`;
    li.querySelector("img").src = thumb;
    li.querySelector(".shot__class").textContent = shown.label;
    li.querySelector(".shot__sub").textContent =
      `${(conf * 100).toFixed(0)}% · ${at.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`;
    li.title = classes
      .map((c, i) => `${c.label}: ${(frame.probs[i] * 100).toFixed(1)}%`)
      .join("\n");

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
    const head = ["captured_at", "verdict", "class", "confidence"]
      .concat(classes.map((c) => "p_" + c.name));
    const rows = shots
      .slice()
      .reverse()
      .map((s) =>
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
  function hideNote() {
    note.hidden = true;
  }

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
        err.name === "NotAllowedError"
          ? "Camera blocked. Allow camera access for this site, then reload."
          : err.name === "NotFoundError"
          ? "No camera found. Connect one and reload."
          : err.name === "NotReadableError"
          ? "Camera is busy. Close Teams, Zoom or any other tab using it, then reload."
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
  optFull.addEventListener("change", () => {
    placeAperture();
    smoothed = null;
  });
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

  /* ── Boot ───────────────────────────────────────────────────── */

  (async function boot() {
    buildTiles();
    thresholdOut.textContent = optThreshold.value + "%";
    try {
      const meta = await fetch("/api/meta").then((r) => r.json());
      classes = meta.classes;
      cleanIndex = classes.findIndex((c) => c.clean);
      buildBars();
      paintBars(classes.map(() => 0), -1);
      if (meta.health && !meta.health.trained) {
        $("alert-body").textContent =
          `${meta.health.note} Logit spread across black, white, grey and striped test ` +
          `plates is ${meta.health.spread} — a trained network moves by whole units. ` +
          `Everything below still works; the numbers just do not mean anything yet.`;
        $("alert").hidden = false;
      }
    } catch {
      showNote("Backend not responding. Start it with: python app.py", true);
      return;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showNote("This browser will not share a camera over an insecure connection. Open the app on localhost, or serve it with --https.", true);
      return;
    }
    await startCamera(null);
  })();
})();
