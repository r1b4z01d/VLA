/* Robot page: a view + command channel for the teleop panel / eval running on the robot PC.
   State arrives as Server-Sent Events from the app (ur5e_lerobot/ui/webpanel.py); commands go
   back as POST /api/cmd and are executed by the app's own control loop. Cameras are <img> MJPEG
   streams, straight from the camera service when it runs (no load on the control loop). */
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const PAGE_API = 2;   // must match ur5e_lerobot/ui/webpanel.API_VERSION of the app serving this page
  let config = { mode: "teleop", cams: {} };
  let state = {};
  let lastStateAt = 0;

  // ---- commands -------------------------------------------------------------------------------
  async function cmd(name, args = {}) {
    try {
      const r = await fetch("/api/cmd", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, args }) });
      if (!r.ok) appendLog(`command ${name} rejected (${r.status})`);
    } catch (e) {
      appendLog(`command ${name} failed: ${e}`);
    }
  }
  $$("[data-cmd]").forEach(b => b.addEventListener("click", () => {
    if (b.dataset.confirm && !window.confirm(b.dataset.confirm)) return;
    let args = {};
    if (b.dataset.args) { try { args = JSON.parse(b.dataset.args); } catch (e) { /* ignore */ } }
    cmd(b.dataset.cmd, args);
  }));

  // Hold-to-ramp hand buttons: the panel ramps the grasp every tick while a direction is held (same
  // rate as the Stream Deck's finger keys) and stops on release — or by itself 0.6 s after the last
  // keepalive, so a lost pointerup can never leave the hand closing.
  $$("[data-hold]").forEach(b => {
    let timer = null;
    const dir = parseInt(b.dataset.hold, 10);
    const start = e => {
      e.preventDefault();
      if (timer) return;
      b.classList.add("active");
      cmd("grasp_hold", { dir });
      timer = setInterval(() => cmd("grasp_hold", { dir }), 250);
    };
    const stop = () => {
      if (!timer) return;
      clearInterval(timer); timer = null;
      b.classList.remove("active");
      cmd("grasp_hold", { dir: 0 });
    };
    b.addEventListener("pointerdown", start);
    ["pointerup", "pointerleave", "pointercancel"].forEach(ev => b.addEventListener(ev, stop));
    window.addEventListener("blur", stop);
  });

  // ---- cameras --------------------------------------------------------------------------------
  function setupCams() {
    $$(".cam").forEach(fig => {
      const name = fig.dataset.cam, img = $("img", fig), url = config.cams[name];
      if (!url) { fig.classList.add("hidden-cam"); return; }
      fig.classList.remove("hidden-cam");
      let holding = false, rawTimer = null;
      const load = () => { img.src = url + (url.includes("?") ? "&" : "?") + "t=" + Date.now(); };
      img.addEventListener("error", () => { if (holding) return; fig.classList.add("dead"); setTimeout(load, 3000); });
      img.addEventListener("load", () => { fig.classList.remove("dead"); if (holding) rawTimer = setTimeout(rawFrame, 150); });
      load();
      // Hold the tile to see the FULL uncropped frame with the crop square drawn (the camera service's
      // /<cam>.raw.jpg, one frame per request, ~6 fps); release to go back to the live cropped stream.
      const rawUrl = /\.mjpg$/.test(url) ? url.replace(/\.mjpg$/, ".raw.jpg") : null;
      if (!rawUrl || !fig.classList.contains("croppable")) return;   // the wrist is uncropped: nothing to show
      const rawFrame = () => { if (holding) img.src = rawUrl + "?t=" + Date.now(); };
      const startRaw = e => {
        if (holding || e.button > 0) return;
        e.preventDefault(); holding = true; fig.classList.add("raw"); rawFrame();
      };
      const stopRaw = () => {
        if (!holding) return;
        holding = false; clearTimeout(rawTimer); fig.classList.remove("raw"); load();
      };
      fig.addEventListener("pointerdown", startRaw);
      ["pointerup", "pointerleave", "pointercancel"].forEach(ev => fig.addEventListener(ev, stopRaw));
      fig.addEventListener("contextmenu", e => e.preventDefault());   // long-press on touch
      window.addEventListener("blur", stopRaw);
    });
  }

  // ---- rendering ------------------------------------------------------------------------------
  function pill(id, text, cls) {
    const el = $(id); el.textContent = text; el.className = "pill" + (cls ? " " + cls : "");
  }
  // Built ONCE, then only the lit state is updated: state arrives ~10x/s and rebuilding the buttons
  // on every update replaced the element between mouse-down and mouse-up, so clicks rarely landed.
  function stars(container, value, onPick) {
    if (container.childElementCount !== 5) {
      container.innerHTML = "";
      for (let i = 1; i <= 5; i++) {
        const b = document.createElement("button");
        b.textContent = "★"; b.title = `${i} star${i > 1 ? "s" : ""}`; b.dataset.n = String(i);
        b.addEventListener("click", () => onPick(i));
        container.appendChild(b);
      }
    }
    Array.from(container.children).forEach((b, k) => b.classList.toggle("lit", k + 1 <= (value || 0)));
  }
  function kv(el, pairs) {
    el.innerHTML = "";
    pairs.forEach(([k, v]) => {
      const s = document.createElement("span"); s.textContent = k;
      const b = document.createElement("b"); b.textContent = v == null ? "—" : String(v);
      el.appendChild(s); el.appendChild(b);
    });
  }
  const fmtTime = s => { s = Math.max(0, Math.round(s || 0)); return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`; };

  function render(st) {
    state = st;
    const ur = st.ur || {}, hand = st.hand || {}, tick = st.tick || {};
    pill("#pill-ur", ur.fault ? "UR FAULT" : ur.connected ? (ur.freedrive ? "UR freedrive" : ur.homing ? "UR homing" : "UR ok") : "UR —",
      ur.fault ? "bad" : ur.connected ? (ur.freedrive || ur.homing ? "warn" : "ok") : "");
    pill("#pill-hand", hand.ok === false ? "hand DOWN" : hand.ok ? "hand ok" : "hand —", hand.ok === false ? "bad" : hand.ok ? "ok" : "");
    if (tick.hz != null) pill("#pill-tick", `${tick.hz.toFixed(1)} tick/s · ${Math.round(tick.ms || 0)} ms`, tick.hz < 9 ? "warn" : "ok");
    if (st.servo) pill("#pill-servo", st.servo.short || "servo", st.servo.slow ? "warn" : "ok");
    const fault = $("#fault");
    fault.classList.toggle("hidden", !ur.fault);
    if (ur.fault) $("#fault-text").textContent = ur.fault;

    // exposure
    const ex = st.exposure || {};
    const lock = $("#expo-lock"); lock.textContent = ex.locked ? "LOCKED" : (ex.phase ? "metering" : "auto"); lock.classList.toggle("on", !!ex.locked);
    renderExpoCams(ex);

    // arm
    const p = ur.pose || [];
    kv($("#arm-kv"), [["x y z", p.length ? p.slice(0, 3).map(v => v.toFixed(3)).join("  ") : "—"],
      ["reach", ur.reach != null ? ur.reach.toFixed(2) + " m" : "—"], ["base", ur.base_deg != null ? ur.base_deg.toFixed(0) + "°" : "—"],
      ["grasp", hand.grasp != null ? Math.round(hand.grasp * 100) + " %" : "—"],
      ["input", st.input || "—"], ["mode", st.play_mode || (ur.freedrive ? "freedrive" : st.recording ? "recording" : "idle")]]);

    if (config.mode === "teleop") renderTeleop(st); else renderEval(st);
  }

  // one row per camera: exposure value, measured median brightness (target ~120), per-camera − / +
  function renderExpoCams(ex) {
    const box = $("#expo-cams"), vals = ex.values || {}, meds = ex.medians || {};
    const names = ["scene", "side", "wrist"].filter(n => n in vals);
    if (box.childElementCount !== names.length) {
      box.innerHTML = "";
      names.forEach(n => {
        const row = document.createElement("div"); row.className = "expo-cam"; row.dataset.cam = n;
        row.innerHTML = `<span>${n}</span><span class="val"><b class="v"></b> <span class="med"></span></span>` +
          `<button class="btn" data-ecam="${n}" data-eact="minus" title="darker: ${n} only">−</button><button class="btn" data-ecam="${n}" data-eact="plus" title="brighter: ${n} only">+</button>`;
        row.querySelectorAll("[data-ecam]").forEach(b => b.addEventListener("click", () => cmd("expo", { action: b.dataset.eact, cam: b.dataset.ecam })));
        box.appendChild(row);
      });
    }
    names.forEach(n => {
      const row = box.querySelector(`.expo-cam[data-cam="${n}"]`); if (!row) return;
      row.querySelector(".v").textContent = Math.round(vals[n]);
      const m = row.querySelector(".med"), md = meds[n];
      m.textContent = md != null ? `· median ${md}` : ""; m.className = "med " + (md == null ? "" : md < 80 ? "dark" : md > 170 ? "bright" : "");
    });
  }

  function renderTeleop(st) {
    $("#btn-rec").classList.toggle("on", !!st.recording);
    $("#btn-home").classList.toggle("on", !!(st.ur || {}).homing);
    $("#btn-free").classList.toggle("on", !!(st.ur || {}).freedrive);
    const rs = $("#rec-state"); rs.textContent = st.recording ? "● REC" : (st.save_in_flight ? "finalizing…" : "idle"); rs.className = st.recording ? "on" : "";
    $("#rec-time").textContent = st.recording ? fmtTime(st.rec_s) : "";
    $("#ep-frames").textContent = st.frames ?? 0;
    $("#ep-grasps").textContent = (st.grasps ?? 1) + (st.regrasp_open ? " (range open)" : "");
    $("#ep-count").textContent = st.episodes ?? 0;
    const ds = st.dataset || {};
    const dn = $("#dataset-name"); if (document.activeElement !== dn && ds.name != null && dn.value !== ds.name) dn.value = ds.name;
    const last = st.last || {};
    $("#last-ep").textContent = last.ep != null ? `episode ${last.ep + 1}` : "";
    stars($("#stars-teleop"), last.rating, n => cmd("rating", { stars: n }));
    $("#last-upload").textContent = last.upload || "";
    const notes = $("#notes"); if (document.activeElement !== notes && notes.value !== (last.notes || "")) notes.value = last.notes || "";
    const task = $("#task"); if (document.activeElement !== task && st.task != null && task.value !== st.task) task.value = st.task;
    // crop sliders: reflect the live value unless the user is dragging
    const crop = st.crop || {};
    $$("[data-crop]").forEach(inp => {
      const name = inp.dataset.crop, v = crop[name];
      const fig = $(`.cam[data-cam="${name}"] .crop-tag`); if (fig) fig.textContent = v == null ? "" : `crop ${Number(v).toFixed(2)}`;
      const val = inp.parentElement.querySelector(".val");
      if (document.activeElement !== inp && v != null) { inp.value = Math.round(v * 100); }
      val.textContent = v == null ? "off" : Number(v).toFixed(2);
      inp.disabled = !!st.crop_locked;
    });
    $("#crop-note").textContent = st.crop_locked ? "fixed (dataset exists)" : "live";
  }

  function renderEval(st) {
    const pm = st.play_mode || "paused";
    const play = $("#btn-play"); play.textContent = pm === "playing" ? "■ PAUSE" : "▶ PLAY"; play.classList.toggle("on", pm === "playing");
    $("#btn-efree").classList.toggle("on", pm === "freedrive");
    $("#btn-ehome").classList.toggle("on", !!(st.ur || {}).homing);
    const run = st.run || {};
    $("#run-id").textContent = run.name || "";
    const sg = st.stage;
    const rsel = $("#run-state");
    rsel.textContent = st.rating_pending ? "rate this run"
      : (sg && sg.phase === "wait_go") ? `▶ PLAY = go: ${sg.name} (${sg.index}/${sg.count})`
      : (sg ? `${pm} · ${sg.name} (${sg.index}/${sg.count})` : pm);
    rsel.className = pm === "playing" ? "on" : "";
    $("#run-time").textContent = fmtTime(run.duration_s);
    const t = st.timing || {};
    kv($("#run-kv"), [["steps", run.steps ?? 0], ["step", t.step_ms != null ? Math.round(t.step_ms) + " ms" : "—"],
      ["infer", t.infer_ms != null ? Math.round(t.infer_ms) + " ms" : "—"], ["obs", t.obs_ms != null ? Math.round(t.obs_ms) + " ms" : "—"],
      ["chunk waits", t.waited ?? "—"]]);
    stars($("#stars-eval"), run.rating, n => cmd("rating", { stars: n }));
    const m = st.model || {};
    kv($("#model-kv"), [["model", m.model_key || "—"], ["checkpoint", m.checkpoint || "—"], ["server", m.server || "—"],
      ["flags", m.flags || "—"], ["crop", m.capture || "—"], ["task", st.task || "—"]]);
    const runs = $("#runs"); runs.innerHTML = "";
    (st.runs || []).slice().reverse().forEach(r => {
      const d = document.createElement("div"); d.className = "run";
      d.innerHTML = `<span>${r.name}</span><span class="st">${"★".repeat(r.rating || 0)}</span><span class="muted">${r.steps ?? ""} steps</span>`;
      runs.appendChild(d);
    });
  }

  // ---- log ------------------------------------------------------------------------------------
  const logEl = $("#log"); let logCount = 0;
  function appendLog(line) {
    logCount++;
    logEl.textContent += (logEl.textContent ? "\n" : "") + line;
    const lines = logEl.textContent.split("\n"); if (lines.length > 400) logEl.textContent = lines.slice(-400).join("\n");
    logEl.scrollTop = logEl.scrollHeight;
    $("#log-count").textContent = String(logCount);
    $("#log-last").textContent = line;   // shown in the header while the log is collapsed
  }

  // collapsible log (state remembered per browser)
  const logWrap = $("#log-wrap");
  const setLogCollapsed = c => {
    logWrap.classList.toggle("collapsed", c); document.body.classList.toggle("log-collapsed", c);
    try { localStorage.setItem("robot.logCollapsed", c ? "1" : "0"); } catch (e) { /* ignore */ }
  };
  try { if (localStorage.getItem("robot.logCollapsed") === "1") setLogCollapsed(true); } catch (e) { /* ignore */ }
  $("#log-toggle").addEventListener("click", () => setLogCollapsed(!logWrap.classList.contains("collapsed")));

  // ---- inputs ---------------------------------------------------------------------------------
  $("#notes").addEventListener("keydown", e => { if (e.key === "Enter") { cmd("notes", { text: e.target.value }); e.target.blur(); } });
  $("#task").addEventListener("keydown", e => { if (e.key === "Enter") { cmd("task", { text: e.target.value }); e.target.blur(); } });
  $("#dataset-name").addEventListener("keydown", e => { if (e.key === "Enter") { cmd("dataset", { name: e.target.value }); e.target.blur(); } });
  $$("[data-crop]").forEach(inp => {
    const val = inp.parentElement.querySelector(".val");
    inp.addEventListener("input", () => { val.textContent = (inp.value / 100).toFixed(2); });
    inp.addEventListener("change", () => cmd("crop", { cam: inp.dataset.crop, at: inp.value / 100 }));
  });
  const KEYS = {
    teleop: { r: ["rec"], s: ["save"], d: ["discard"], h: ["home"], f: ["freedrive"], g: ["grasp_mark"],
      o: ["grasp", { delta: -0.06 }], c: ["grasp", { delta: 0.06 }], "1": ["rating", { stars: 1 }], "2": ["rating", { stars: 2 }],
      "3": ["rating", { stars: 3 }], "4": ["rating", { stars: 4 }], "5": ["rating", { stars: 5 }] },
    eval: { p: ["play"], " ": ["play"], f: ["freedrive"], r: ["reset"], h: ["home"], n: ["newrun"], o: ["open_hand"], c: ["close_hand"],
      "1": ["rating", { stars: 1 }], "2": ["rating", { stars: 2 }], "3": ["rating", { stars: 3 }], "4": ["rating", { stars: 4 }], "5": ["rating", { stars: 5 }] },
  };
  document.addEventListener("keydown", e => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
    const k = KEYS[config.mode] && KEYS[config.mode][e.key.toLowerCase()];
    if (k) { e.preventDefault(); cmd(k[0], k[1] || {}); }
  });

  // ---- connection -----------------------------------------------------------------------------
  function connect() {
    const es = new EventSource("/events");
    es.addEventListener("state", ev => { lastStateAt = Date.now(); try { render(JSON.parse(ev.data)); } catch (e) { /* ignore */ } });
    es.addEventListener("log", ev => { try { appendLog(JSON.parse(ev.data)); } catch (e) { /* ignore */ } });
    es.onopen = () => pill("#pill-sync", "link ok", "ok");
    es.onerror = () => pill("#pill-sync", "link lost — retrying", "bad");
  }
  setInterval(() => { if (lastStateAt && Date.now() - lastStateAt > 4000) pill("#pill-sync", "no state 4 s", "warn"); }, 2000);

  fetch("/api/config").then(r => r.json()).then(c => {
    config = c;
    document.body.classList.add("mode-" + c.mode);
    if ((c.api || 0) < PAGE_API) {
      const f = $("#fault"); f.classList.remove("hidden");
      $("#fault-text").textContent = `The ${c.mode === "eval" ? "eval" : "panel"} on the robot is running older code than this page (api ${c.api || 0} < ${PAGE_API}): some buttons will not work. Quit it and launch it again.`;
      $("#fault button").hidden = true;
    }
    if (c.mode === "eval") $("#pills-eval-slot").replaceWith($("#pills"));   // pills live in the active action row
    document.title = (c.mode === "eval" ? "Eval" : "Teleop") + " · " + (c.title || "robot");
    setupCams();
    connect();
  }).catch(e => appendLog("config failed: " + e));
})();
