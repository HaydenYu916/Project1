// Govee MPPI dashboard frontend
const $ = (id) => document.getElementById(id);

// ---------------- Collapsible panels ----------------
(() => {
  const KEY = "govee_panel_collapsed";
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (_) {}
  document.querySelectorAll(".panel").forEach((panel, i) => {
    const h = panel.querySelector("h2");
    if (!h) return;
    const id = h.textContent.trim() || `panel-${i}`;
    if (saved[id]) panel.classList.add("collapsed");
    h.addEventListener("click", () => {
      panel.classList.toggle("collapsed");
      saved[id] = panel.classList.contains("collapsed");
      try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (_) {}
    });
  });
})();

let pendingTimer = null;
const DEBOUNCE_MS = 250;

function fmt(x, n = 2) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return Number(x).toFixed(n);
}

function fmtTime(iso) {
  if (!iso) return "—";
  return iso.replace("T", " ");
}

// Slider race guard: when user touches a slider we mark a "do not snap back"
// window. Polled /api/state arriving inside this window will not overwrite the
// slider position (otherwise the user's drag visibly bounces back to the old
// server value during the 250 ms debounce, then jumps again when the dispatch
// resolves).
let _userInteractingUntil = 0;
const USER_INTERACT_FREEZE_MS = 1500;
function _markUserInteracting() { _userInteractingUntil = Date.now() + USER_INTERACT_FREEZE_MS; }

function applyState(s) {
  if (!s) return;
  const userActive = Date.now() < _userInteractingUntil;
  if (!userActive) {
    $("r-slider").value = s.pwm_r;
    $("b-slider").value = s.pwm_b;
    $("r-val").textContent = s.pwm_r;
    $("b-val").textContent = s.pwm_b;
  }
  // labels/preview always reflect server truth even while user interacting
  // — but only via the dispatch echo, not via stale poll.
  $("m-pred2").textContent = fmt(s.predicted_ppfd, 2);
  $("m-meas").textContent = fmt(s.last_measured_ppfd, 2);
  $("m-measat").textContent = fmtTime(s.last_measured_at);
  $("m-disp").textContent = fmtTime(s.last_dispatch_at);
  $("m-lat").textContent = s.last_dispatch_latency_ms === null ? "—" : Math.round(s.last_dispatch_latency_ms);
  $("m-vcap").textContent = fmt(s.vcap_v, 3);

  // color preview from server PWM (skipped during user interaction so the
  // live drag preview from scheduleDispatch isn't briefly overwritten).
  if (!userActive) {
    if (s.mode === "full_white") {
      $("color-preview").style.background = "rgb(255, 255, 255)";
    } else {
      const r255 = Math.round((s.pwm_r / 100) * 255);
      const b255 = Math.round((s.pwm_b / 100) * 255);
      $("color-preview").style.background = `rgb(${r255}, 0, ${b255})`;
    }
  }
}

async function fetchState() {
  try {
    const r = await fetch("/api/state");
    if (!r.ok) return;
    const s = await r.json();
    applyState(s);
  } catch (e) {
    console.warn("state fetch failed:", e);
  }
}

async function dispatchPwm(pwm_r, pwm_b) {
  try {
    const r = await fetch("/api/led", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pwm_r, pwm_b }),
    });
    if (!r.ok) {
      const txt = await r.text();
      showWarn(`LED dispatch failed: ${txt}`);
      return;
    }
    const s = await r.json();
    if (s.warn) {
      showWarn(s.warn);
    } else {
      hideWarn();
    }
    applyState(s);
  } catch (e) {
    showWarn(`network error: ${e.message}`);
  }
}

function scheduleDispatch() {
  const r = parseInt($("r-slider").value, 10);
  const b = parseInt($("b-slider").value, 10);
  $("r-val").textContent = r;
  $("b-val").textContent = b;
  // immediate preview
  const r255 = Math.round((r / 100) * 255);
  const b255 = Math.round((b / 100) * 255);
  $("color-preview").style.background = `rgb(${r255}, 0, ${b255})`;
  if (pendingTimer) clearTimeout(pendingTimer);
  pendingTimer = setTimeout(() => dispatchPwm(r, b), DEBOUNCE_MS);
}

["r-slider", "b-slider"].forEach((id) => {
  const el = $(id);
  el.addEventListener("input", () => { _markUserInteracting(); scheduleDispatch(); });
  el.addEventListener("change", _markUserInteracting);
  el.addEventListener("pointerdown", _markUserInteracting);
});

// Click the color preview circle → fire full-white max brightness ("charging").
(function attachFullWhite() {
  const c = $("color-preview");
  if (!c) return;
  c.style.cursor = "pointer";
  c.title = "Click → full white at max brightness (charging mode)";
  c.addEventListener("click", async () => {
    _markUserInteracting();
    c.style.opacity = "0.5";
    // Collapse Measure PPFD + Riotee Sensor panels (charging mode focuses
    // attention on the LED control row only).
    document.querySelectorAll(".panel").forEach((panel) => {
      const h = panel.querySelector("h2");
      if (!h) return;
      const title = h.textContent.trim();
      if (title === "Measure PPFD" || title === "Riotee Sensor") {
        panel.classList.add("collapsed");
        // Persist the collapsed state so a subsequent reload keeps the
        // user-visible compact view (uses the same key as the h2 click
        // handler at the top of this file).
        try {
          const KEY = "govee_panel_collapsed";
          const saved = JSON.parse(localStorage.getItem(KEY) || "{}");
          saved[title] = true;
          localStorage.setItem(KEY, JSON.stringify(saved));
        } catch (_) {}
      }
    });
    try {
      const r = await fetch("/api/led/full_white", { method: "POST" });
      if (!r.ok) {
        showWarn(`full white failed: ${await r.text()}`);
        return;
      }
      const s = await r.json();
      if (s.warn) showWarn(s.warn); else hideWarn();
      // Reflect target state in UI (R=B=100 sliders, white preview).
      $("r-slider").value = 100;
      $("b-slider").value = 100;
      $("r-val").textContent = 100;
      $("b-val").textContent = 100;
      c.style.background = "rgb(255, 255, 255)";
      applyState({ ...s, pwm_r: 100, pwm_b: 100 });
      // Re-paint pure white on top (applyState would compute it from
      // R/B as rgb(255,0,255) — magenta — which is wrong for white mode).
      c.style.background = "rgb(255, 255, 255)";
    } catch (e) {
      showWarn(`network error: ${e.message}`);
    } finally {
      c.style.opacity = "1";
    }
  });
})();

$("measure-btn").addEventListener("click", async () => {
  const btn = $("measure-btn");
  const stat = $("measure-status");
  const n = Math.max(1, Math.min(5, parseInt($("trigger-count").value, 10) || 1));
  btn.disabled = true;
  stat.innerHTML = `<span class="spinner"></span> measuring (${n}×)…`;
  try {
    const r = await fetch("/api/measure", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trigger_count: n }),
    });
    if (!r.ok) {
      const txt = await r.text();
      stat.textContent = `error: ${txt}`;
      return;
    }
    const data = await r.json();
    applyState(data.state);
    if (data.delta_pct !== null && data.delta_pct !== undefined) {
      $("m-delta").textContent = (data.delta_abs >= 0 ? "+" : "") + fmt(data.delta_abs, 2);
      $("m-delta-pct").textContent = `(${data.delta_pct >= 0 ? "+" : ""}${fmt(data.delta_pct, 1)}%)`;
      $("delta-card").className = "metric " + (Math.abs(data.delta_pct) < 10 ? "good" : Math.abs(data.delta_pct) < 25 ? "warn" : "bad");
    } else {
      $("m-delta").textContent = "—";
      $("m-delta-pct").textContent = "";
    }
    stat.textContent = `done in ${n}× trigger(s)`;
  } catch (e) {
    stat.textContent = `error: ${e.message}`;
  } finally {
    btn.disabled = false;
  }
});

let _warnHideTimer = null;
function showWarn(msg, autoHideMs = 5000) {
  const el = $("warn");
  // Build with a close button instead of plain textContent
  el.innerHTML = "";
  const span = document.createElement("span");
  span.textContent = "⚠️ " + msg;
  span.style.flex = "1";
  const close = document.createElement("button");
  close.textContent = "✕";
  close.title = "dismiss";
  close.style.cssText =
    "background:transparent;color:#fed7aa;border:0;font-size:16px;" +
    "cursor:pointer;padding:0 4px;margin-left:8px;line-height:1;font-weight:700;";
  close.addEventListener("click", hideWarn);
  el.style.display = "flex";
  el.style.alignItems = "center";
  el.appendChild(span);
  el.appendChild(close);
  if (_warnHideTimer) clearTimeout(_warnHideTimer);
  if (autoHideMs > 0) _warnHideTimer = setTimeout(hideWarn, autoHideMs);
}
function hideWarn() {
  $("warn").style.display = "none";
  if (_warnHideTimer) { clearTimeout(_warnHideTimer); _warnHideTimer = null; }
}

// initial + polling. Stop polling while the tab is hidden (browsers throttle
// setInterval anyway, but this keeps things explicit) and re-fire on focus so
// users see fresh data immediately when they come back.
let _pollTimers = [];
function _startPolling() {
  _stopPolling();
  fetchState();
  _pollTimers.push(setInterval(fetchState, 3000));
}
function _stopPolling() {
  _pollTimers.forEach(clearInterval);
  _pollTimers = [];
}
_startPolling();
document.addEventListener("visibilitychange", () => {
  if (document.hidden) { _stopPolling(); }
  else { _startPolling(); refreshSensorLatest && refreshSensorLatest(); }
});

// ---------------- MPPI panel ----------------

const MAX_LOG_ROWS = 30;
const HISTORY_KEY = "govee_mppi_history";
const LOG_KEY     = "govee_mppi_log";
const mppiHistory = []; // {step, target, pred, meas, power}
const mppiLog     = []; // raw step events, newest first

function persist() {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(mppiHistory));
    localStorage.setItem(LOG_KEY, JSON.stringify(mppiLog.slice(0, MAX_LOG_ROWS)));
  } catch (_) {}
}

function restoreFromStorage() {
  try {
    const h = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    if (Array.isArray(h)) { mppiHistory.length = 0; h.forEach(x => mppiHistory.push(x)); }
    const log = JSON.parse(localStorage.getItem(LOG_KEY) || "[]");
    if (Array.isArray(log)) {
      mppiLog.length = 0; log.forEach(x => mppiLog.push(x));
      const tb = $("mppi-log-body");
      if (tb) { tb.innerHTML = ""; mppiLog.forEach(ev => appendLogRow(ev, /*persistAfter=*/false)); }
    }
  } catch (_) {}
}

function clearMppiData() {
  mppiHistory.length = 0;
  mppiLog.length = 0;
  const tb = $("mppi-log-body");
  if (tb) tb.innerHTML = "";
  try {
    localStorage.removeItem(HISTORY_KEY);
    localStorage.removeItem(LOG_KEY);
  } catch (_) {}
  drawChart();
}

function setMppiRunning(running) {
  $("mppi-start").disabled = !!running;
  $("mppi-stop").disabled = !running;
  ["mppi-dt","mppi-horizon","mppi-samples","mppi-rb","mppi-steps","mppi-gain","mppi-measure"]
    .forEach(id => { const el = $(id); if (el) el.disabled = !!running; });
  // gate manual sliders too
  $("r-slider").disabled = !!running;
  $("b-slider").disabled = !!running;
}

function applyMppiState(s) {
  if (!s) return;
  $("mp-step").textContent = s.step ?? "—";
  $("mp-u").textContent    = fmt(s.last_u, 2);
  $("mp-pred").textContent = fmt(s.last_pred, 2);
  $("mp-meas").textContent = fmt(s.last_measured, 2);
  $("mp-cost").textContent = fmt(s.last_min_cost, 2);
  $("mp-pow").textContent  = fmt(s.last_power_w, 2);
  if (s.last_err_pct === null || s.last_err_pct === undefined) {
    $("mp-err").textContent = "—";
    $("mp-err-card").className = "metric";
  } else {
    const sign = s.last_err_pct >= 0 ? "+" : "";
    $("mp-err").textContent = sign + fmt(s.last_err_pct, 1) + "%";
    const a = Math.abs(s.last_err_pct);
    $("mp-err-card").className = "metric " + (a < 10 ? "good" : a < 25 ? "warn" : "bad");
  }
  setMppiRunning(s.running);
  $("mppi-status").textContent = s.running
    ? `running since ${fmtTime(s.started_at)}`
    : (s.stopped_at ? `stopped at ${fmtTime(s.stopped_at)}` : "idle");
  if (s.last_error) $("mppi-status").textContent += ` — ${s.last_error}`;
}

function appendLogRow(ev, persistAfter = true) {
  const tb = $("mppi-log-body");
  const tr = document.createElement("tr");
  const errCell = ev.err_pct === null || ev.err_pct === undefined
    ? "—"
    : ((ev.err_pct >= 0 ? "+" : "") + fmt(ev.err_pct, 1) + "%");
  const errColor = ev.err_pct === null || ev.err_pct === undefined ? "var(--muted)"
    : Math.abs(ev.err_pct) < 10 ? "var(--good)"
    : Math.abs(ev.err_pct) < 25 ? "var(--warn)" : "var(--bad)";
  tr.innerHTML = `
    <td style="padding:4px 8px;">${fmtTime(ev.t_iso)}</td>
    <td style="padding:4px 8px;text-align:right;">${ev.step}</td>
    <td style="padding:4px 8px;text-align:right;">${fmt(ev.mppi_u,2)}</td>
    <td style="padding:4px 8px;text-align:right;color:var(--red-led);">${fmt(ev.pwm_r,1)}</td>
    <td style="padding:4px 8px;text-align:right;color:var(--blue-led);">${fmt(ev.pwm_b,1)}</td>
    <td style="padding:4px 8px;text-align:right;">${fmt(ev.pred_ppfd,2)}</td>
    <td style="padding:4px 8px;text-align:right;">${fmt(ev.measured_ppfd,2)}</td>
    <td style="padding:4px 8px;text-align:right;color:${errColor};">${errCell}</td>`;
  tb.insertBefore(tr, tb.firstChild);
  while (tb.children.length > MAX_LOG_ROWS) tb.removeChild(tb.lastChild);
  if (persistAfter) {
    mppiLog.unshift(ev);
    while (mppiLog.length > MAX_LOG_ROWS) mppiLog.pop();
    persist();
  }
}

const SERIES = [
  { key: "target", label: "target", color: "#fbbf24", dashed: true,  axis: "L", unit: "µmol/m²/s" },
  { key: "pred",   label: "pred",   color: "#38bdf8", dashed: false, axis: "L", unit: "µmol/m²/s" },
  { key: "meas",   label: "meas",   color: "#4ade80", dashed: false, axis: "L", unit: "µmol/m²/s" },
  { key: "power",  label: "power",  color: "#f97316", dashed: false, axis: "R", unit: "W" },
];
const seriesEnabled = { target: true, pred: true, meas: true, power: true };
let _legendHitboxes = []; // {x,y,w,h,key}

function pushHistory(ev) {
  mppiHistory.push({
    step:   ev.step,
    target: ev.target_ppfd,
    pred:   ev.pred_ppfd,
    meas:   ev.measured_ppfd,
    power:  ev.power_w,
  });
  while (mppiHistory.length > 80) mppiHistory.shift();
  drawChart();
  persist();
}

function drawChart() {
  const canvas = $("mppi-chart");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  _legendHitboxes = [];
  if (mppiHistory.length === 0) {
    ctx.fillStyle = "#475569"; ctx.font = "13px sans-serif";
    ctx.fillText("waiting for MPPI events…", 12, H/2);
    drawLegend(ctx, W);
    return;
  }
  // separate scales for left (PPFD) and right (W) axes
  const collect = axis => {
    const vs = [];
    SERIES.filter(s => s.axis === axis && seriesEnabled[s.key]).forEach(s => {
      mppiHistory.forEach(h => { if (h[s.key] != null && !Number.isNaN(h[s.key])) vs.push(h[s.key]); });
    });
    return vs;
  };
  const span = vs => {
    if (vs.length === 0) return [0, 1];
    let lo = Math.min(...vs), hi = Math.max(...vs);
    if (hi - lo < 1) hi = lo + 1;
    const pad = (hi - lo) * 0.1; return [lo - pad, hi + pad];
  };
  const [Lmin, Lmax] = span(collect("L"));
  const [Rmin, Rmax] = span(collect("R"));
  const padL = 36, padR = 40, padT = 22, padB = 20;
  const innerW = W - padL - padR, innerH = H - padT - padB;
  const n = mppiHistory.length;
  const xstep = innerW / Math.max(1, n - 1);
  const yL = v => padT + innerH - (v - Lmin)/(Lmax - Lmin) * innerH;
  const yR = v => padT + innerH - (v - Rmin)/(Rmax - Rmin) * innerH;

  // grid + dual-axis labels
  ctx.strokeStyle = "#1e293b"; ctx.lineWidth = 1; ctx.font = "10px sans-serif";
  for (let i = 0; i <= 4; i++) {
    const y = padT + i * innerH / 4;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    const lv = Lmax - i * (Lmax - Lmin) / 4;
    const rv = Rmax - i * (Rmax - Rmin) / 4;
    ctx.fillStyle = "#475569"; ctx.fillText(lv.toFixed(1), 2, y + 3);
    ctx.fillStyle = "#f97316aa"; ctx.fillText(rv.toFixed(1), W - padR + 4, y + 3);
  }

  SERIES.forEach(s => {
    if (!seriesEnabled[s.key]) return;
    const yfor = s.axis === "L" ? yL : yR;
    ctx.strokeStyle = s.color; ctx.lineWidth = 2;
    ctx.setLineDash(s.dashed ? [4, 4] : []);
    ctx.beginPath();
    let started = false;
    mppiHistory.forEach((h, i) => {
      const v = h[s.key];
      if (v == null || Number.isNaN(v)) return;
      const x = padL + i * xstep, y = yfor(v);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    });
    ctx.stroke(); ctx.setLineDash([]);
  });

  drawLegend(ctx, W);
}

function drawLegend(ctx, W) {
  // Legend intentionally hidden per UX request; series remain toggleable
  // programmatically via `seriesEnabled` if needed.
  _legendHitboxes = [];
}

(function attachLegendClicks() {
  const canvas = $("mppi-chart");
  if (!canvas) return;
  canvas.style.cursor = "pointer";
  canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width  / rect.width;
    const sy = canvas.height / rect.height;
    const cx = (e.clientX - rect.left) * sx;
    const cy = (e.clientY - rect.top)  * sy;
    for (const hb of _legendHitboxes) {
      if (cx >= hb.x && cx <= hb.x + hb.w && cy >= hb.y && cy <= hb.y + hb.h) {
        seriesEnabled[hb.key] = !seriesEnabled[hb.key];
        drawChart();
        return;
      }
    }
  });
})();

function readMppiForm() {
  return {
    dt:           parseFloat($("mppi-dt").value),
    horizon:      parseInt($("mppi-horizon").value, 10),
    num_samples:  parseInt($("mppi-samples").value, 10),
    rb:           parseFloat($("mppi-rb").value),
    steps:        parseInt($("mppi-steps").value, 10) || 0,
    demo_gain:    parseFloat($("mppi-gain").value),
    measure:      $("mppi-measure").value === "1",
  };
}

$("mppi-start").addEventListener("click", async () => {
  $("mppi-status").textContent = "starting…";
  try {
    const r = await fetch("/api/mppi/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readMppiForm()),
    });
    if (!r.ok) {
      const txt = await r.text();
      $("mppi-status").textContent = `start failed: ${txt}`;
      return;
    }
    setMppiRunning(true);
  } catch (e) {
    $("mppi-status").textContent = `error: ${e.message}`;
  }
});

$("mppi-stop").addEventListener("click", async () => {
  $("mppi-status").textContent = "stopping…";
  try {
    await fetch("/api/mppi/stop", { method: "POST" });
  } catch (e) {
    $("mppi-status").textContent = `error: ${e.message}`;
  }
});

$("mppi-clear").addEventListener("click", () => {
  if (mppiHistory.length === 0 && mppiLog.length === 0) return;
  if (!confirm("Clear MPPI chart and log history?")) return;
  clearMppiData();
});

async function fetchMppiSnapshot() {
  try {
    const r = await fetch("/api/mppi");
    if (!r.ok) return;
    applyMppiState(await r.json());
  } catch (_) {}
}

let ws = null;
function connectWs() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws/mppi`);
  ws.onmessage = (m) => {
    let ev;
    try { ev = JSON.parse(m.data); } catch (_) { return; }
    if (ev.type === "snapshot") {
      applyMppiState(ev.state);
    } else if (ev.type === "started") {
      setMppiRunning(true);
      $("mppi-status").textContent = `running since ${fmtTime(ev.started_at)}`;
      // History is preserved across runs; user can wipe via the Clear button.
    } else if (ev.type === "step") {
      appendLogRow(ev);
      pushHistory(ev);
      applyMppiState({
        running: true,
        step: ev.step,
        last_u: ev.mppi_u,
        last_pred: ev.pred_ppfd,
        last_measured: ev.measured_ppfd,
        last_min_cost: ev.min_cost,
        last_power_w: ev.power_w,
        last_err_pct: ev.err_pct,
        started_at: $("mppi-status").textContent.replace(/^running since /, ""),
      });
    } else if (ev.type === "stopped") {
      setMppiRunning(false);
      $("mppi-status").textContent = `stopped at ${fmtTime(ev.stopped_at)} (step ${ev.step})`;
    } else if (ev.type === "error") {
      $("mppi-status").textContent = `error: ${ev.msg}`;
    }
  };
  ws.onclose = () => { setTimeout(connectWs, 2000); };
  ws.onerror = () => { try { ws.close(); } catch (_) {} };
}
// MPPI panel hidden? Skip WS + snapshot polling so it can't disable the
// manual sliders via setMppiRunning(true) and can't churn empty redraws.
const _mppiHidden = !!document.querySelector('.panel[data-disabled-mppi="1"]');
if (!_mppiHidden) {
  restoreFromStorage();
  connectWs();
  fetchMppiSnapshot();
  drawChart();
}

// ---------------- GL-Gym Closed Loop ----------------

const glHistory = []; // {tick, target, pred, meas, u}
const GL_MAX_POINTS = 240;

function glReadForm() {
  return {
    tick_sec:       parseFloat($("gl-tick").value),
    max_ticks:      parseInt($("gl-max").value, 10),
    max_ppfd:       parseFloat($("gl-maxppfd").value),
    red_frac:       parseFloat($("gl-rfrac").value),
    start_day:      parseInt($("gl-startday").value, 10),
    start_hour:     parseFloat($("gl-starthour").value),
    steps_per_tick: parseInt($("gl-stepspertick").value, 10),
    sim_only:       $("gl-simonly").value === "1",
    measure:        $("gl-measure").value === "1",
  };
}

function glSetRunning(running) {
  $("gl-start").disabled = !!running;
  $("gl-stop").disabled  = !running;
  ["gl-tick","gl-max","gl-maxppfd","gl-rfrac","gl-startday",
   "gl-starthour","gl-stepspertick","gl-simonly","gl-measure"]
    .forEach(id => { const el = $(id); if (el) el.disabled = !!running; });
  // Disable manual sliders + measure button while GLGym holds the loop.
  $("r-slider").disabled = !!running;
  $("b-slider").disabled = !!running;
  $("measure-btn").disabled = !!running;
}

function glApply(s) {
  if (!s) return;
  $("gl-tickval").textContent = s.tick ?? "—";
  $("gl-simtime").textContent = (s.sim_day != null && s.sim_hour != null)
    ? `d${Math.floor(s.sim_day)} ${fmt(s.sim_hour,1)}h` : "—";
  $("gl-ulamp").textContent  = fmt(s.u_lamp, 2);
  $("gl-target").textContent = fmt(s.target_ppfd, 1);
  $("gl-pwm").textContent    = (s.pwm_r != null && s.pwm_b != null)
    ? `${s.pwm_r}/${s.pwm_b}` : "—";
  $("gl-pred").textContent   = fmt(s.predicted_ppfd, 1);
  $("gl-meas").textContent   = fmt(s.measured_ppfd, 1);
  glSetRunning(s.running);
  $("gl-status").textContent = s.running
    ? `running since ${fmtTime(s.started_at)} (tick ${s.tick||0})`
    : (s.stopped_at ? `stopped at ${fmtTime(s.stopped_at)}` : "idle");
  if (s.last_error) $("gl-status").textContent += ` — ${s.last_error}`;
}

function glPushHistory(ev) {
  glHistory.push({
    tick: ev.tick,
    target: ev.target_ppfd,
    pred: ev.predicted_ppfd,
    meas: ev.measured_ppfd,
    u: ev.u_lamp,
  });
  while (glHistory.length > GL_MAX_POINTS) glHistory.shift();
  glDrawChart();
}

const GL_SERIES = [
  { key: "target", color: "#fbbf24", dashed: true,  label: "target" },
  { key: "pred",   color: "#38bdf8", dashed: false, label: "pred" },
  { key: "meas",   color: "#4ade80", dashed: false, label: "meas" },
];

function glDrawChart() {
  const c = $("gl-chart"); if (!c) return;
  const ctx = c.getContext("2d");
  const W = c.width, H = c.height;
  ctx.clearRect(0,0,W,H);
  if (glHistory.length === 0) {
    ctx.fillStyle = "#475569"; ctx.font = "13px sans-serif";
    ctx.fillText("waiting for GL-Gym ticks…", 12, H/2);
    return;
  }
  let lo = Infinity, hi = -Infinity;
  GL_SERIES.forEach(s => {
    glHistory.forEach(h => {
      const v = h[s.key];
      if (v != null && !Number.isNaN(v)) { lo = Math.min(lo,v); hi = Math.max(hi,v); }
    });
  });
  if (lo === Infinity) { lo = 0; hi = 1; }
  if (hi - lo < 1) hi = lo + 1;
  const pad = (hi-lo)*0.1; lo -= pad; hi += pad;
  const padL = 36, padR = 12, padT = 22, padB = 22;
  const innerW = W - padL - padR, innerH = H - padT - padB;
  const n = glHistory.length;
  const xstep = innerW / Math.max(1, n - 1);
  const y = v => padT + innerH - (v-lo)/(hi-lo) * innerH;

  ctx.strokeStyle = "#1e293b"; ctx.lineWidth = 1; ctx.font = "10px sans-serif";
  for (let i=0;i<=4;i++){
    const yy = padT + i*innerH/4;
    ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(W-padR, yy); ctx.stroke();
    const v = hi - i*(hi-lo)/4;
    ctx.fillStyle = "#475569"; ctx.fillText(v.toFixed(1), 2, yy+3);
  }
  GL_SERIES.forEach(s => {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2;
    ctx.setLineDash(s.dashed ? [4,4] : []);
    ctx.beginPath();
    let started = false;
    glHistory.forEach((h,i) => {
      const v = h[s.key];
      if (v == null || Number.isNaN(v)) return;
      const xx = padL + i*xstep, yy = y(v);
      if (!started) { ctx.moveTo(xx,yy); started = true; } else ctx.lineTo(xx,yy);
    });
    ctx.stroke(); ctx.setLineDash([]);
  });
  // legend
  ctx.font = "11px sans-serif";
  let lx = padL + 6;
  GL_SERIES.forEach(s => {
    ctx.fillStyle = s.color; ctx.fillRect(lx, 4, 12, 4);
    ctx.fillStyle = "#94a3b8"; ctx.fillText(s.label, lx + 16, 12);
    lx += 60;
  });
}

$("gl-start").addEventListener("click", async () => {
  $("gl-status").textContent = "starting…";
  try {
    const r = await fetch("/api/glgym/start", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(glReadForm()),
    });
    if (!r.ok) {
      $("gl-status").textContent = `start failed: ${await r.text()}`;
      return;
    }
    glSetRunning(true);
  } catch (e) { $("gl-status").textContent = `error: ${e.message}`; }
});

$("gl-stop").addEventListener("click", async () => {
  $("gl-status").textContent = "stopping…";
  try { await fetch("/api/glgym/stop", { method: "POST" }); }
  catch (e) { $("gl-status").textContent = `error: ${e.message}`; }
});

$("gl-clear").addEventListener("click", () => {
  glHistory.length = 0; glDrawChart();
});

async function glFetchSnapshot() {
  try {
    const r = await fetch("/api/glgym");
    if (r.ok) glApply(await r.json());
  } catch (_) {}
}

// Hook GL-Gym events into the existing /ws/mppi WebSocket (same channel).
// Reuse the connectWs() above by patching its onmessage path: both MPPI and
// GLGym events share `_broadcast` server-side, so the same WS gets both.
(function attachGLGymWs() {
  // Open our own WS (cheap, lets us run alongside the MPPI WS or replace it).
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  let ws;
  function open() {
    ws = new WebSocket(`${proto}//${location.host}/ws/mppi`);
    ws.onmessage = (m) => {
      let ev; try { ev = JSON.parse(m.data); } catch (_) { return; }
      if (ev.type === "glgym_started") {
        glSetRunning(true);
        $("gl-status").textContent = `running since ${fmtTime(ev.started_at)}`;
      } else if (ev.type === "glgym_tick") {
        glPushHistory(ev);
        glApply({
          running: true,
          tick: ev.tick, sim_hour: ev.sim_hour, sim_day: ev.sim_day,
          u_lamp: ev.u_lamp, target_ppfd: ev.target_ppfd,
          pwm_r: ev.pwm_r, pwm_b: ev.pwm_b,
          predicted_ppfd: ev.predicted_ppfd, measured_ppfd: ev.measured_ppfd,
          started_at: $("gl-status").textContent.replace(/^running since /,"").split(" (")[0],
        });
      } else if (ev.type === "glgym_stopped") {
        glSetRunning(false);
        $("gl-status").textContent = `stopped at ${fmtTime(ev.stopped_at)} (tick ${ev.tick})`;
      } else if (ev.type === "glgym_error") {
        $("gl-status").textContent = `error: ${ev.msg}`;
      }
    };
    ws.onclose = () => setTimeout(open, 2000);
    ws.onerror = () => { try { ws.close(); } catch (_) {} };
  }
  open();
})();

glFetchSnapshot();
glDrawChart();

// ---------------- GL-Gym Comparison ----------------

const CMP_COLORS = {
  rule_based: "#38bdf8",   // sky blue
  glgym_rb:   "#a855f7",   // purple
  lightfarm:  "#f97316",   // orange
};
const CMP_LABEL = {
  rule_based: "rule_based (8–22 ON)",
  glgym_rb:   "glgym_rb (smart RB)",
  lightfarm:  "LightFarm (MPPI)",
};
const cmpHistory = []; // { tick, sim_hour, sim_day, ctls: {name: {pn, power_w}} }
const CMP_MAX_POINTS = 600;

function cmpReadForm() {
  // Hardcoded sim setup: indoor 22°C lab, lamp-only, lamp scale 8 (lets
  // rule_based overheat so MPPI's energy savings + Pn protection show clearly).
  // Pn is evaluated by the user's chamber HistGBR model (Tool/Model/EnvtoPN),
  // not GreenLight's internal Farquhar — see server._compare_run.
  return {
    days:           parseFloat($("cmp-days").value),
    tick_sec:       parseFloat($("cmp-tick").value),
    steps_per_tick: parseInt($("cmp-steps").value, 10),
    start_day:          200,
    lamp_par_scale:     3.0,
    indoor:             true,
    no_climate_control: true,
    pn_source:          "chamber",   // "chamber" | "greenlight"
    lamp_ppfd_max:      300.0,        // u_lamp=1 → 300 µmol/m²/s — keep controllers
                                      // in chamber Pn model's LINEAR segment so
                                      // u_lamp differences translate to visible Pn diff
                                      // (Pn at 200 ≈ 10, at 500 ≈ 19 — saturating)
    chamber_rb:         0.83,         // R fraction for the chamber Pn
    chamber_lai_scale:  3.0,          // leaf Pn × LAI ≈ canopy Pn (greenhouse tomato 2–4)
    controllers:        ["rule_based", "lightfarm"],
  };
}

function cmpSetRunning(r) {
  $("cmp-start").disabled = !!r;
  $("cmp-stop").disabled  = !r;
  ["cmp-days","cmp-tick","cmp-steps"].forEach(id => {
    const el = $(id); if (el) el.disabled = !!r;
  });
}

function cmpDrawChart(canvasId, key, ylabel) {
  const c = $(canvasId); if (!c) return;
  const ctx = c.getContext("2d");
  const W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  if (cmpHistory.length === 0) {
    ctx.fillStyle = "#475569"; ctx.font = "13px sans-serif";
    ctx.fillText("waiting for compare ticks…", 12, H/2);
    return;
  }
  // determine y range
  let lo = Infinity, hi = -Infinity;
  cmpHistory.forEach(h => {
    Object.values(h.ctls).forEach(v => {
      const x = v[key];
      if (x != null && !Number.isNaN(x)) { lo = Math.min(lo, x); hi = Math.max(hi, x); }
    });
  });
  if (lo === Infinity) { lo = 0; hi = 1; }
  if (hi - lo < 0.001) hi = lo + 0.001;
  const pad = (hi-lo)*0.1; lo -= pad; hi += pad;
  const padL = 42, padR = 12, padT = 22, padB = 24;
  const innerW = W - padL - padR, innerH = H - padT - padB;
  const n = cmpHistory.length;
  const xstep = innerW / Math.max(1, n - 1);
  const y = v => padT + innerH - (v-lo)/(hi-lo) * innerH;

  // grid
  ctx.strokeStyle = "#1e293b"; ctx.lineWidth = 1; ctx.font = "10px sans-serif";
  for (let i=0;i<=4;i++){
    const yy = padT + i*innerH/4;
    ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(W-padR, yy); ctx.stroke();
    const v = hi - i*(hi-lo)/4;
    ctx.fillStyle = "#475569"; ctx.fillText(v.toFixed(3), 2, yy+3);
  }
  ctx.fillStyle = "#94a3b8"; ctx.fillText(ylabel, padL, padT - 8);

  // staircase per controller (each tick is a piecewise-constant hold).
  // Chart x positions samples at i*xstep; we extend each level horizontally
  // until the next sample, then jump vertically — so control-style ZOH lines.
  const seenCtls = new Set();
  cmpHistory.forEach(h => Object.keys(h.ctls).forEach(c => seenCtls.add(c)));
  Array.from(seenCtls).forEach(name => {
    ctx.strokeStyle = CMP_COLORS[name] || "#888";
    ctx.lineWidth = 2; ctx.beginPath();
    let prevY = null;
    cmpHistory.forEach((h, i) => {
      const v = h.ctls[name] && h.ctls[name][key];
      if (v == null || Number.isNaN(v)) return;
      const xx = padL + i * xstep, yy = y(v);
      if (prevY === null) {
        ctx.moveTo(xx, yy);
      } else {
        ctx.lineTo(xx, prevY);   // hold at previous level until next sample
        ctx.lineTo(xx, yy);      // then step vertically
      }
      prevY = yy;
    });
    ctx.stroke();
  });

  // legend
  let lx = padL + 6;
  ctx.font = "11px sans-serif";
  Array.from(seenCtls).forEach(name => {
    ctx.fillStyle = CMP_COLORS[name] || "#888"; ctx.fillRect(lx, 4, 12, 4);
    ctx.fillStyle = "#94a3b8"; ctx.fillText(CMP_LABEL[name] || name, lx + 16, 12);
    lx += 90;
  });
}

function cmpRedraw() {
  cmpDrawChart("cmp-pn-chart", "pn",      "mg CO₂·m⁻²·s⁻¹");
  cmpDrawChart("cmp-pw-chart", "power_w", "W·m⁻²");
  cmpDrawChart("cmp-t-chart",  "t_air",   "°C");
}

function cmpRenderSummary(summary) {
  const card = $("cmp-delta-cards");
  if (!summary || Object.keys(summary).length === 0) {
    $("cmp-summary").textContent = "";
    if (card) card.innerHTML = "";
    return;
  }
  // Cumulative summary table
  const lines = ["<b>Cumulative summary:</b>"];
  for (const [name, s] of Object.entries(summary)) {
    const pn = s.cum_pn_mgCO2_m2 / 1000.0;
    const e  = s.cum_energy_kwh_m2;
    lines.push(
      `<span style="color:${CMP_COLORS[name]||'#888'};">●</span> ` +
      `<b>${CMP_LABEL[name]||name}</b>: ` +
      `∫Pn = ${pn.toFixed(2)} g CO₂/m² · ` +
      `energy = ${e.toFixed(3)} kWh/m²`
    );
  }
  $("cmp-summary").innerHTML = lines.join("<br>");

  // % delta cards (LightFarm vs rule_based) — energy and Pn
  if (!card) return;
  const rb = summary["rule_based"], lf = summary["lightfarm"];
  if (!rb || !lf) { card.innerHTML = ""; return; }
  const e_save_pct = (rb.cum_energy_kwh_m2 - lf.cum_energy_kwh_m2)
                  / Math.max(rb.cum_energy_kwh_m2, 1e-9) * 100;
  const pn_chg_pct = (lf.cum_pn_mgCO2_m2 - rb.cum_pn_mgCO2_m2)
                  / Math.max(rb.cum_pn_mgCO2_m2, 1e-9) * 100;
  const klass = (v, good) => {
    const pos = good ? v > 0 : v < 0;
    return pos ? "good" : (Math.abs(v) < 1 ? "" : "bad");
  };
  card.innerHTML =
    `<div class="metric ${klass(e_save_pct, true)}">` +
    `  <span class="label">⚡ energy saved (LightFarm vs rule_based)</span>` +
    `  <span class="value">${e_save_pct >= 0 ? "−" : "+"}${Math.abs(e_save_pct).toFixed(1)}%</span>` +
    `</div>` +
    `<div class="metric ${klass(pn_chg_pct, true)}">` +
    `  <span class="label">🌱 ∫Pn change (LightFarm vs rule_based)</span>` +
    `  <span class="value">${pn_chg_pct >= 0 ? "+" : ""}${pn_chg_pct.toFixed(1)}%</span>` +
    `</div>`;
}

$("cmp-start").addEventListener("click", async () => {
  $("cmp-status").textContent = "starting…";
  cmpHistory.length = 0; cmpRedraw(); $("cmp-summary").textContent = "";
  try {
    const r = await fetch("/api/glgym/compare/start", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(cmpReadForm()),
    });
    if (!r.ok) {
      $("cmp-status").textContent = `start failed: ${await r.text()}`;
      return;
    }
    cmpSetRunning(true);
  } catch (e) { $("cmp-status").textContent = `error: ${e.message}`; }
});
$("cmp-stop").addEventListener("click", async () => {
  $("cmp-status").textContent = "stopping…";
  try { await fetch("/api/glgym/compare/stop", { method: "POST" }); }
  catch (e) { $("cmp-status").textContent = `error: ${e.message}`; }
});
$("cmp-clear").addEventListener("click", () => {
  cmpHistory.length = 0; cmpRedraw(); $("cmp-summary").textContent = "";
});

// Hook compare events into the same WS used elsewhere.
(function attachCmpWs() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  let ws;
  function open() {
    ws = new WebSocket(`${proto}//${location.host}/ws/mppi`);
    ws.onmessage = (m) => {
      let ev; try { ev = JSON.parse(m.data); } catch (_) { return; }
      if (ev.type === "glgym_compare_started") {
        cmpSetRunning(true);
        $("cmp-status").textContent = `running since ${fmtTime(ev.started_at)}`;
      } else if (ev.type === "glgym_compare_tick") {
        cmpHistory.push({
          tick: ev.tick, sim_hour: ev.sim_hour, sim_day: ev.sim_day,
          ctls: ev.controllers,
        });
        while (cmpHistory.length > CMP_MAX_POINTS) cmpHistory.shift();
        cmpRedraw();
        $("cmp-status").textContent =
          `tick ${ev.tick} | sim d${Math.floor(ev.sim_day)} ${ev.sim_hour.toFixed(1)}h`;
      } else if (ev.type === "glgym_compare_stopped") {
        cmpSetRunning(false);
        $("cmp-status").textContent = `stopped at ${fmtTime(ev.stopped_at)} (tick ${ev.tick})`;
        cmpRenderSummary(ev.summary);
      } else if (ev.type === "glgym_compare_error") {
        $("cmp-status").textContent = `error: ${ev.msg}`;
      }
    };
    ws.onclose = () => setTimeout(open, 2000);
    ws.onerror = () => { try { ws.close(); } catch (_) {} };
  }
  open();
})();
cmpRedraw();

// ---------------- Sensor panel ----------------

const SP_CHANNELS = ["sp_415","sp_445","sp_480","sp_515","sp_555","sp_590","sp_630","sp_680"];
const SP_COLORS   = {  // approximate visible color per peak wavelength
  sp_415: "#8b5cf6", sp_445: "#3b82f6", sp_480: "#06b6d4", sp_515: "#10b981",
  sp_555: "#84cc16", sp_590: "#facc15", sp_630: "#f97316", sp_680: "#ef4444",
};
const tempHistory = []; // {t, temp, hum}
let lastSpectrum = null;

function applySensorLatest(row) {
  if (!row) return;
  if ($("s-temp")) $("s-temp").textContent  = fmt(row.temperature, 2);
  if ($("s-hum"))  $("s-hum").textContent   = fmt(row.humidity, 1);
  if ($("s-co2"))  $("s-co2").textContent   = (row.co2_ppm == null || row.co2_ppm < 0) ? "—" : fmt(row.co2_ppm, 0);
  if ($("s-vcap")) $("s-vcap").textContent  = fmt(row.vcap_raw, 3);
  if ($("s-ts"))   $("s-ts").textContent    = row.timestamp || "—";
  if ($("sensor-sp-chart")) {
    lastSpectrum = SP_CHANNELS.map(k => row[k]);
    drawSpectrum();
  }
}

function applySensorHistory(rows) {
  tempHistory.length = 0;
  rows.forEach(r => {
    if (r.temperature == null) return;
    tempHistory.push({ t: r.timestamp, temp: r.temperature, hum: r.humidity });
  });
  drawTempChart();
}

function drawTempChart() {
  const c = $("sensor-temp-chart");
  const ctx = c.getContext("2d");
  const W = c.width, H = c.height;
  ctx.clearRect(0,0,W,H);
  if (tempHistory.length < 2) {
    ctx.fillStyle = "#475569"; ctx.font = "13px sans-serif";
    ctx.fillText("waiting for sensor history…", 12, H/2);
    return;
  }
  const temps = tempHistory.map(h => h.temp).filter(v => v != null);
  const hums  = tempHistory.map(h => h.hum ).filter(v => v != null);
  let tmin = Math.min(...temps), tmax = Math.max(...temps);
  if (tmax - tmin < 0.5) { tmax = tmin + 0.5; }
  const tpad = (tmax - tmin) * 0.15; tmin -= tpad; tmax += tpad;
  let hmin = hums.length ? Math.min(...hums) : 0;
  let hmax = hums.length ? Math.max(...hums) : 100;
  if (hmax - hmin < 1) hmax = hmin + 1;
  const hpad = (hmax - hmin) * 0.15; hmin -= hpad; hmax += hpad;
  const n = tempHistory.length;
  const xstep = (W - 60) / Math.max(1, n - 1);
  const yT = v => H - 20 - (v - tmin)/(tmax - tmin) * (H - 40);
  const yH = v => H - 20 - (v - hmin)/(hmax - hmin) * (H - 40);

  // grid + left axis (temp)
  ctx.strokeStyle = "#1e293b"; ctx.lineWidth = 1;
  ctx.font = "10px sans-serif";
  for (let i = 0; i <= 4; i++) {
    const y = 20 + i * (H - 40)/4;
    ctx.beginPath(); ctx.moveTo(40, y); ctx.lineTo(W - 20, y); ctx.stroke();
    const tv = tmax - i * (tmax - tmin)/4;
    ctx.fillStyle = "#f87171"; ctx.fillText(tv.toFixed(1), 4, y+3);
    const hv = hmax - i * (hmax - hmin)/4;
    ctx.fillStyle = "#38bdf8"; ctx.fillText(hv.toFixed(0), W - 18, y+3);
  }
  // temp line
  ctx.strokeStyle = "#f87171"; ctx.lineWidth = 2; ctx.beginPath();
  let st = false;
  tempHistory.forEach((h, i) => {
    if (h.temp == null) return;
    const x = 40 + i * xstep, y = yT(h.temp);
    if (!st) { ctx.moveTo(x,y); st = true; } else ctx.lineTo(x,y);
  });
  ctx.stroke();
  // hum line
  ctx.strokeStyle = "#38bdf8"; ctx.lineWidth = 2; ctx.beginPath();
  st = false;
  tempHistory.forEach((h, i) => {
    if (h.hum == null) return;
    const x = 40 + i * xstep, y = yH(h.hum);
    if (!st) { ctx.moveTo(x,y); st = true; } else ctx.lineTo(x,y);
  });
  ctx.stroke();
  // legend
  ctx.font = "11px sans-serif";
  ctx.fillStyle = "#f87171"; ctx.fillRect(W - 150, 4, 10, 10);
  ctx.fillStyle = "#94a3b8"; ctx.fillText("temp °C", W - 136, 13);
  ctx.fillStyle = "#38bdf8"; ctx.fillRect(W - 80, 4, 10, 10);
  ctx.fillStyle = "#94a3b8"; ctx.fillText("RH %",   W - 66, 13);
}

function drawSpectrum() {
  const c = $("sensor-sp-chart");
  if (!c) return;
  const ctx = c.getContext("2d");
  const W = c.width, H = c.height;
  ctx.clearRect(0,0,W,H);
  if (!lastSpectrum || lastSpectrum.every(v => v == null)) {
    ctx.fillStyle = "#475569"; ctx.font = "13px sans-serif";
    ctx.fillText("waiting for spectrum…", 12, H/2);
    return;
  }
  const vals = lastSpectrum.map(v => v == null ? 0 : v);
  const vmax = Math.max(1, ...vals);
  const padL = 36, padR = 12, padT = 16, padB = 28;
  const innerW = W - padL - padR, innerH = H - padT - padB;
  const barW = innerW / SP_CHANNELS.length * 0.7;
  const gap  = innerW / SP_CHANNELS.length;
  // y grid
  ctx.strokeStyle = "#1e293b"; ctx.lineWidth = 1;
  ctx.font = "10px sans-serif"; ctx.fillStyle = "#475569";
  for (let i = 0; i <= 4; i++) {
    const y = padT + i * innerH/4;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    const v = vmax - i * vmax/4;
    ctx.fillStyle = "#475569"; ctx.fillText(v.toFixed(0), 2, y + 3);
  }
  // bars
  SP_CHANNELS.forEach((ch, i) => {
    const v = vals[i];
    const x = padL + i * gap + (gap - barW)/2;
    const h = v / vmax * innerH;
    const y = padT + innerH - h;
    ctx.fillStyle = SP_COLORS[ch];
    ctx.fillRect(x, y, barW, h);
    // label
    ctx.fillStyle = "#94a3b8"; ctx.font = "10px sans-serif";
    ctx.fillText(ch.replace("sp_",""), x, H - 14);
    ctx.fillStyle = "#e2e8f0";
    ctx.fillText(v.toFixed(0), x, y - 3);
  });
}

async function refreshSensorLatest() {
  try {
    const r = await fetch("/api/sensor");
    if (!r.ok) return;
    applySensorLatest(await r.json());
  } catch (_) {}
}

async function refreshSensorHistory() {
  try {
    const r = await fetch("/api/sensor/history?n=120");
    if (!r.ok) return;
    const data = await r.json();
    applySensorHistory(data.rows || []);
  } catch (_) {}
}

refreshSensorLatest();
refreshSensorHistory();
setInterval(refreshSensorLatest, 5000);
setInterval(refreshSensorHistory, 30000);
drawTempChart();
