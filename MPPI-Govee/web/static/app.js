// Govee MPPI dashboard frontend
const $ = (id) => document.getElementById(id);

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

function applyState(s) {
  if (!s) return;
  if ($("r-slider").matches(":not(:active)")) {
    $("r-slider").value = s.pwm_r;
  }
  if ($("b-slider").matches(":not(:active)")) {
    $("b-slider").value = s.pwm_b;
  }
  $("r-val").textContent = s.pwm_r;
  $("b-val").textContent = s.pwm_b;
  $("m-pred").textContent = fmt(s.predicted_ppfd, 2);
  $("m-pred2").textContent = fmt(s.predicted_ppfd, 2);
  $("m-meas").textContent = fmt(s.last_measured_ppfd, 2);
  $("m-measat").textContent = fmtTime(s.last_measured_at);
  $("m-disp").textContent = fmtTime(s.last_dispatch_at);
  $("m-lat").textContent = s.last_dispatch_latency_ms === null ? "—" : Math.round(s.last_dispatch_latency_ms);
  $("m-vcap").textContent = fmt(s.vcap_v, 3);

  // color preview based on PWM
  const r255 = Math.round((s.pwm_r / 100) * 255);
  const b255 = Math.round((s.pwm_b / 100) * 255);
  $("color-preview").style.background = `rgb(${r255}, 0, ${b255})`;
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
    hideWarn();
    const s = await r.json();
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

$("r-slider").addEventListener("input", scheduleDispatch);
$("b-slider").addEventListener("input", scheduleDispatch);

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

function showWarn(msg) {
  const el = $("warn");
  el.textContent = "⚠️ " + msg;
  el.style.display = "block";
}
function hideWarn() {
  $("warn").style.display = "none";
}

// initial + polling
fetchState();
setInterval(fetchState, 3000);
