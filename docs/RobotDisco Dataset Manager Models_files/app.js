"use strict";

const state = {
  datasets: [],
  selected: null,
  episodes: [],
  epSel: new Set(),
  merge: new Set(),
  signature: null,
  playing: null,
};

const $ = (s) => document.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v === true) n.setAttribute(k, "");
    else if (v !== false && v != null) n.setAttribute(k, v);
  }
  for (const kid of kids) if (kid != null) n.append(kid.nodeType ? kid : document.createTextNode(kid));
  return n;
};

// ---- net ----
async function api(path, opts) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `${r.status} ${r.statusText}`);
  return body;
}
const post = (path, payload) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload || {}) });

// ---- toast ----
let toastTimer = null;
function toast(msg, kind = "") {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast " + kind;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), kind === "err" ? 6000 : 3000);
}
const ok = (m) => toast(m, "ok");
const err = (e) => toast(typeof e === "string" ? e : e.message || String(e), "err");

// ---- modal ----
function modal({ title, fields, submit = "OK" }) {
  return new Promise((resolve) => {
    const inputs = {};
    const body = fields.map((f) => {
      let input;
      if (f.type === "select") {
        input = el("select", {}, ...f.options.map((o) => el("option", { value: o.value }, o.label)));
        if (f.value != null) input.value = f.value;
      } else if (f.type === "checkbox") {
        input = el("input", { type: "checkbox" });
        if (f.value) input.checked = true;
      } else {
        input = el("input", { type: f.type || "text", value: f.value ?? "", placeholder: f.placeholder || "" });
      }
      inputs[f.name] = input;
      const wrap = f.type === "checkbox"
        ? el("label", { style: "display:flex;gap:8px;align-items:center" }, input, f.label)
        : el("div", {}, el("label", {}, f.label), input);
      if (f.hint) wrap.append(el("div", { class: "hint" }, f.hint));
      return wrap;
    });
    const close = (val) => { back.remove(); resolve(val); };
    const back = el("div", { class: "modal-back", onclick: (e) => e.target === back && close(null) },
      el("div", { class: "modal" },
        el("h3", {}, title),
        ...body,
        el("div", { class: "modal-actions" },
          el("button", { class: "btn", onclick: () => close(null) }, "Cancel"),
          el("button", { class: "btn primary", onclick: () => {
            const out = {};
            for (const [k, i] of Object.entries(inputs)) out[k] = i.type === "checkbox" ? i.checked : i.value;
            close(out);
          } }, submit))));
    document.body.append(back);
    const first = back.querySelector("input,select"); if (first) first.focus();
  });
}

// ---- datasets ----
async function loadDatasets(fromMonitor = false) {
  let data;
  try { data = await api("/api/datasets"); } catch (e) { return err(e); }
  $("#datasets-dir").textContent = data.datasets_dir || "";
  const changed = state.signature && data.signature !== state.signature;
  state.signature = data.signature;
  state.datasets = data.datasets;
  renderDatasets();
  const mon = $("#monitor");
  if (fromMonitor && changed) {
    mon.textContent = "change detected"; mon.classList.add("changed");
    setTimeout(() => { mon.classList.remove("changed"); mon.textContent = "monitoring…"; }, 2500);
    if (state.selected) loadEpisodes(state.selected);
  } else if (!fromMonitor) {
    mon.textContent = "monitoring…";
  }
}

function renderDatasets() {
  const list = $("#dataset-list");
  list.replaceChildren();
  for (const d of state.datasets) {
    if (d.error) {
      list.append(el("div", { class: "ds-card" },
        el("div", { class: "ds-name" }, d.name), el("div", { class: "ds-err" }, d.error)));
      continue;
    }
    const check = el("input", { class: "ds-check", type: "checkbox", title: "select for merge",
      onclick: (e) => { e.stopPropagation(); e.target.checked ? state.merge.add(d.name) : state.merge.delete(d.name); syncMergeBtn(); } });
    check.checked = state.merge.has(d.name);
    const card = el("div", { class: "ds-card" + (state.selected === d.name ? " selected" : ""),
      onclick: () => selectDataset(d.name) },
      el("div", { class: "ds-top" }, check,
        el("span", { class: "ds-name" }, d.name),
        el("span", { class: "badge " + d.media }, d.media)),
      el("div", { class: "ds-meta" },
        `${d.episodes} eps · ${d.frames} frames · ${d.fps ?? "?"}fps · ${d.size_mb ?? "?"}MB`),
      el("div", { class: "ds-meta" }, "task: " + (d.tasks && d.tasks.length ? d.tasks.join(", ") : "—")),
      el("div", { class: "ds-foot" },
        el("button", { class: "btn tiny", onclick: (e) => { e.stopPropagation(); doRenameDataset(d.name); } }, "rename"),
        el("button", { class: "btn tiny", onclick: (e) => { e.stopPropagation(); doToVideo(d.name); } }, "Export video"),
        d.has_backup ? el("button", { class: "btn tiny", onclick: (e) => { e.stopPropagation(); doRestore(d.name); } }, "restore .bak") : null,
        el("button", { class: "btn tiny danger", onclick: (e) => { e.stopPropagation(); doDeleteDataset(d.name); } }, "delete")));
    list.append(card);
  }
  syncMergeBtn();
}
const syncMergeBtn = () => { $("#btn-merge").disabled = state.merge.size < 2; };

async function selectDataset(name) {
  state.selected = name;
  state.epSel.clear();
  renderDatasets();
  $("#episodes-title").textContent = `Episodes — ${name}`;
  $("#episode-toolbar").hidden = false;
  await loadEpisodes(name);
}

// ---- episodes ----
async function loadEpisodes(name) {
  let data;
  try { data = await api(`/api/datasets/${encodeURIComponent(name)}/episodes`); } catch (e) { return err(e); }
  state.episodes = data.episodes;
  renderEpisodes();
}

function stars(ep) {
  const cur = ep.rating || 0;
  const box = el("span", { class: "stars" });
  for (let i = 1; i <= 5; i++) {
    box.append(el("span", { class: i <= cur ? "on" : "", onclick: () => annotate(ep.episode, { rating: i === cur ? null : i }) }, "★"));
  }
  return box;
}

function renderEpisodes() {
  $("#episodes-empty").hidden = true;
  const tbl = $("#episodes-table"); tbl.hidden = false;
  const body = $("#episodes-body"); body.replaceChildren();
  for (const ep of state.episodes) {
    const sel = el("input", { type: "checkbox",
      onclick: (e) => { e.target.checked ? state.epSel.add(ep.episode) : state.epSel.delete(ep.episode); syncEpSel(); } });
    sel.checked = state.epSel.has(ep.episode);
    const notes = el("input", { class: "ep-input", value: ep.notes || "", placeholder: "notes…" });
    notes.addEventListener("change", () => annotate(ep.episode, { notes: notes.value }, true));
    const oper = el("input", { class: "ep-input", value: ep.operator || "", placeholder: "operator" });
    oper.addEventListener("change", () => annotate(ep.episode, { operator: oper.value }, true));
    const row = el("tr", { class: state.playing === ep.episode ? "playing" : "" },
      el("td", { class: "c-sel" }, sel),
      el("td", { class: "c-ep mono" }, String(ep.episode)),
      el("td", {}, el("div", { class: "task-cell", title: ep.task }, ep.task || "—")),
      el("td", { class: "c-len mono" }, String(ep.length)),
      el("td", { class: "c-len mono" }, ep.duration_s + "s"),
      el("td", { class: "c-rate" }, stars(ep)),
      el("td", {}, notes),
      el("td", { class: "c-op" }, oper),
      el("td", { class: "c-act" },
        el("button", { class: "btn tiny primary", onclick: () => play(ep.episode) }, "▶"),
        el("button", { class: "btn tiny", onclick: () => doTrim(ep) }, "trim"),
        el("button", { class: "btn tiny", onclick: () => doMove(ep) }, "move"),
        el("button", { class: "btn tiny danger", onclick: () => doDeleteEpisodes([ep.episode]) }, "del")));
    body.append(row);
  }
  syncEpSel();
  updateNavButtons();
  renderPlayerControls();
}
function syncEpSel() {
  $("#btn-del-selected").disabled = state.epSel.size === 0;
  const all = $("#sel-all"); all.checked = state.episodes.length > 0 && state.epSel.size === state.episodes.length;
}

async function annotate(episode, fields, silent = false) {
  try {
    await post(`/api/datasets/${encodeURIComponent(state.selected)}/annotate`, { episode, ...fields });
    const e = state.episodes.find((x) => x.episode === episode);
    if (e) Object.assign(e, fields);
    if ("rating" in fields) renderEpisodes();
    if (!silent) ok("saved");
  } catch (e) { err(e); }
}

// ---- player ----
async function play(episode) {
  const label = $("#player-label");
  label.textContent = `rendering ep ${episode}…`;
  $("#player-frame").replaceChildren(el("div", { class: "placeholder" },
    `Rendering episode ${episode} to rerun… the first play of an episode takes ~30s (cached afterwards).`));
  try {
    const r = await api(`/api/datasets/${encodeURIComponent(state.selected)}/episodes/${episode}/viewer`);
    $("#player-frame").replaceChildren(el("iframe", { src: r.url }));
    label.textContent = `${state.selected} · ep ${episode}  (rerun)`;
    $("#btn-player-close").hidden = false;
    state.playing = episode; renderEpisodes();
  } catch (e) { err(e); label.textContent = "Player"; $("#player-frame").replaceChildren(el("div", { class: "placeholder" }, "Player")); }
}
async function stopPlayer() {
  try { await post("/api/viewer/stop"); } catch (e) { /* ignore */ }
  $("#player-frame").replaceChildren(el("div", { class: "placeholder" }, "Press ▶ on an episode to play it in the rerun viewer."));
  $("#player-label").textContent = "Player";
  $("#btn-player-close").hidden = true;
  state.playing = null; renderEpisodes();
}

// ---- prev/next episode nav ----
function currentPlayingIndex() {
  return state.playing == null ? -1 : state.episodes.findIndex((e) => e.episode === state.playing);
}
function updateNavButtons() {
  const i = currentPlayingIndex();
  $("#btn-prev-ep").disabled = i <= 0;
  $("#btn-next-ep").disabled = i < 0 || i >= state.episodes.length - 1;
}
function playAdjacent(delta) {
  const i = currentPlayingIndex();
  if (i < 0) return;
  const j = i + delta;
  if (j >= 0 && j < state.episodes.length) play(state.episodes[j].episode);
}

// When the episode list is collapsed, mirror the playing episode's controls into the player bar.
// Reuses the same handlers as the table, so behavior is identical.
function renderPlayerControls() {
  const box = $("#player-ep-controls");
  const ep = state.playing == null ? null : state.episodes.find((e) => e.episode === state.playing);
  const show = state.epCollapsed && ep != null;
  box.hidden = !show;
  box.replaceChildren();
  if (!show) return;
  const notes = el("input", { class: "ep-input", value: ep.notes || "", placeholder: "notes…", title: "notes" });
  notes.addEventListener("change", () => annotate(ep.episode, { notes: notes.value }, true));
  const oper = el("input", { class: "ep-input", value: ep.operator || "", placeholder: "operator", title: "operator" });
  oper.addEventListener("change", () => annotate(ep.episode, { operator: oper.value }, true));
  box.append(
    el("span", { class: "ep-tag mono" }, `ep ${ep.episode} · ${ep.length}f`),
    stars(ep), notes, oper,
    el("button", { class: "btn tiny", onclick: () => doTrim(ep) }, "trim"),
    el("button", { class: "btn tiny", onclick: () => doMove(ep) }, "move"),
    el("button", { class: "btn tiny danger", onclick: () => doDeleteEpisodes([ep.episode]) }, "del"));
}

// ---- mutating actions ----
async function doDeleteEpisodes(eps) {
  if (!confirm(`Delete episode(s) ${eps.join(", ")} from "${state.selected}"?\nThe dataset is rebuilt; the previous version is kept as .bak.`)) return;
  try {
    const r = await post(`/api/datasets/${encodeURIComponent(state.selected)}/delete_episodes`, { episodes: eps, confirm: true });
    ok(`deleted ${eps.length} — now ${r.episodes} episodes`);
    if (state.playing != null && eps.includes(state.playing)) await stopPlayer();
    state.epSel.clear(); await loadDatasets(); await loadEpisodes(state.selected);
  } catch (e) { err(e); }
}

async function doTrim(ep) {
  const v = await modal({ title: `Trim episode ${ep.episode}`, submit: "Trim",
    fields: [
      { name: "cut_start_s", label: "cut start (seconds)", type: "number", placeholder: "0.0" },
      { name: "cut_end_s", label: "cut end (seconds)", type: "number", placeholder: String(ep.duration_s) },
    ] });
  if (!v) return;
  if (!confirm(`Drop frames from ${v.cut_start_s}s to ${v.cut_end_s}s of ep ${ep.episode}? (rebuild + .bak)`)) return;
  try {
    const r = await post(`/api/datasets/${encodeURIComponent(state.selected)}/trim`,
      { episode: ep.episode, cut_start_s: parseFloat(v.cut_start_s), cut_end_s: parseFloat(v.cut_end_s), confirm: true });
    ok(`trimmed ep ${ep.episode}: dropped ${r.dropped_frames}f, kept ${r.kept_frames}f`);
    await loadDatasets(); await loadEpisodes(state.selected);
    if (state.playing === ep.episode) await play(ep.episode);  // refresh the now-trimmed recording
  } catch (e) { err(e); }
}

async function doMove(ep) {
  const others = state.datasets.filter((d) => d.name !== state.selected && !d.error);
  const fields = [];
  if (others.length) {
    fields.push({ name: "dst", label: "move into an existing dataset", type: "select",
      options: [{ value: "", label: "— none (use a new dataset) —" },
                ...others.map((d) => ({ value: d.name, label: d.name }))] });
  }
  fields.push({ name: "new_name", label: "…or a NEW dataset (type a name)", type: "text",
    placeholder: "e.g. good_grasps",
    hint: "creates a new dataset from this episode (LeRobot datasets can't be empty)" });
  const v = await modal({ title: `Move episode ${ep.episode}`, submit: "Move", fields });
  if (!v) return;
  const dst = ((v.new_name || "").trim()) || v.dst;
  if (!dst) return err("pick an existing dataset or type a new name");
  try {
    const r = await post(`/api/datasets/${encodeURIComponent(state.selected)}/move`, { episode: ep.episode, dst });
    ok(`moved ep ${ep.episode} → ${r.dst}${r.created ? " (new dataset)" : ""} · src ${r.src_episodes}, dst ${r.dst_episodes}`);
    if (state.playing === ep.episode) stopPlayer();
    await loadDatasets(); await loadEpisodes(state.selected);
  } catch (e) { err(e); }
}

async function doRenameTask() {
  const nSel = state.epSel.size;
  const v = await modal({ title: "Rename task", submit: "Rename",
    fields: [
      { name: "new_task", label: "new task string", type: "text", placeholder: "e.g. put the block in the tray" },
      ...(nSel ? [{ name: "selected_only", label: `apply to the ${nSel} selected episode(s) only`, type: "checkbox", value: true }] : []),
    ] });
  if (!v || !v.new_task) return;
  const payload = v.selected_only
    ? { episode_tasks: Object.fromEntries([...state.epSel].map((e) => [e, v.new_task])) }
    : { new_task: v.new_task };
  try {
    await post(`/api/datasets/${encodeURIComponent(state.selected)}/rename_task`, payload);
    ok("task renamed"); await loadDatasets(); await loadEpisodes(state.selected);
  } catch (e) { err(e); }
}

async function doMerge() {
  const names = [...state.merge];
  const v = await modal({ title: `Merge ${names.length} datasets`, submit: "Merge",
    fields: [{ name: "out_name", label: "new dataset name", type: "text", placeholder: "merged_" + Date.now().toString(36),
      hint: "sources: " + names.join(", ") + " (left untouched)" }] });
  if (!v || !v.out_name) return;
  try {
    const r = await post("/api/datasets/merge", { names, out_name: v.out_name });
    ok(`merged → ${r.name} (${r.episodes} episodes)`); state.merge.clear(); await loadDatasets();
  } catch (e) { err(e); }
}

async function doToVideo(name) {
  const v = await modal({ title: `Convert "${name}" to video`, submit: "Convert",
    fields: [{ name: "out_name", label: "new dataset name", type: "text", value: name + "_video",
      hint: "encodes image frames as MP4 — much faster training loads. Source untouched." }] });
  if (!v) return;
  toast("converting to video… this can take a while");
  try {
    const r = await post(`/api/datasets/${encodeURIComponent(name)}/to_video`, { out_name: v.out_name });
    ok(`created ${r.name}`); await loadDatasets();
  } catch (e) { err(e); }
}

async function doRenameDataset(name) {
  const v = await modal({ title: `Rename dataset "${name}"`, submit: "Rename",
    fields: [{ name: "new_name", label: "new name", type: "text", value: name }] });
  if (!v || !v.new_name || v.new_name === name) return;
  try {
    const r = await post(`/api/datasets/${encodeURIComponent(name)}/rename`, { new_name: v.new_name });
    ok(`renamed → ${r.new}`);
    if (state.selected === name) state.selected = r.new;
    if (state.playing != null) stopPlayer();
    await loadDatasets();
    if (state.selected === r.new) await loadEpisodes(r.new);
  } catch (e) { err(e); }
}

async function doDeleteDataset(name) {
  if (!confirm(`Delete the ENTIRE dataset "${name}" (and its .bak)? This cannot be undone.`)) return;
  try { await post(`/api/datasets/${encodeURIComponent(name)}/delete`, { confirm: true }); ok(`deleted ${name}`);
    if (state.selected === name) { state.selected = null; $("#episodes-table").hidden = true; $("#episodes-empty").hidden = false; $("#episode-toolbar").hidden = true; }
    await loadDatasets();
  } catch (e) { err(e); }
}

async function doRestore(name) {
  if (!confirm(`Restore "${name}" from its .bak (swaps current ↔ backup)?`)) return;
  try { await post(`/api/datasets/${encodeURIComponent(name)}/restore`); ok(`restored ${name}`);
    await loadDatasets(); if (state.selected === name) await loadEpisodes(name);
  } catch (e) { err(e); }
}

async function doAdd() {
  const v = await modal({ title: "Add / import a dataset", submit: "Add",
    fields: [
      { name: "remote", label: "import from robot PC (.80) — dataset name", type: "text", placeholder: "e.g. manual_4",
        hint: "rsync from rd@192.168.11.80:~/VLA/outputs/datasets/<name>" },
      { name: "source", label: "…or copy from a local path", type: "text", placeholder: "/abs/path/to/dataset" },
      { name: "as_name", label: "save as (optional)", type: "text", placeholder: "leave blank to keep the name" },
    ] });
  if (!v || (!v.remote && !v.source)) return;
  toast("importing…");
  try {
    const r = await post("/api/datasets/add", { remote: v.remote || null, source: v.source || null, as_name: v.as_name || null });
    ok(`added ${r.name}`); await loadDatasets();
  } catch (e) { err(e); }
}

// ---- wire up ----
$("#btn-refresh").addEventListener("click", () => loadDatasets());
$("#btn-add").addEventListener("click", doAdd);
$("#btn-merge").addEventListener("click", doMerge);
$("#btn-rename-task").addEventListener("click", doRenameTask);
$("#btn-to-video").addEventListener("click", () => state.selected && doToVideo(state.selected));
$("#btn-del-selected").addEventListener("click", () => doDeleteEpisodes([...state.epSel]));
$("#btn-player-close").addEventListener("click", stopPlayer);
$("#btn-prev-ep").addEventListener("click", () => playAdjacent(-1));
$("#btn-next-ep").addEventListener("click", () => playAdjacent(1));
document.addEventListener("keydown", (e) => {   // ←/→ step episodes while a player is open
  if (state.playing == null) return;
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
  if (e.key === "ArrowLeft") { e.preventDefault(); playAdjacent(-1); }
  else if (e.key === "ArrowRight") { e.preventDefault(); playAdjacent(1); }
});
$("#sel-all").addEventListener("click", (e) => {
  state.epSel = e.target.checked ? new Set(state.episodes.map((x) => x.episode)) : new Set();
  renderEpisodes();
});

// collapse panels (persisted across reloads)
state.dsCollapsed = localStorage.getItem("dsCollapsed") === "1";
state.epCollapsed = localStorage.getItem("epCollapsed") === "1";
function applyCollapse() {
  $("#datasets-panel").classList.toggle("collapsed", state.dsCollapsed);
  $("#ds-collapse").textContent = state.dsCollapsed ? "»" : "«";
  $("#work-panel").classList.toggle("ep-collapsed", state.epCollapsed);
  $("#ep-collapse").textContent = state.epCollapsed ? "▸" : "▾";
  renderPlayerControls();
}
$("#ds-collapse").addEventListener("click", () => {
  state.dsCollapsed = !state.dsCollapsed;
  localStorage.setItem("dsCollapsed", state.dsCollapsed ? "1" : "0"); applyCollapse();
});
$("#ep-collapse").addEventListener("click", () => {
  state.epCollapsed = !state.epCollapsed;
  localStorage.setItem("epCollapsed", state.epCollapsed ? "1" : "0"); applyCollapse();
});
applyCollapse();

// ---- view switching (Datasets / Models / Evals) ----
const VIEWS = ["datasets", "models", "evals"];
function showView(v) {
  state.view = v;
  VIEWS.forEach((name) => { $("#view-" + name).hidden = name !== v; });
  document.querySelectorAll(".nav-btn").forEach((b) => b.classList.toggle("active", b.dataset.view === v));
  if (v === "models") loadModels();
  if (v === "evals") loadEvals();
  if (location.hash !== "#" + v) location.hash = "#" + v;
}
document.querySelectorAll(".nav-btn").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));
window.addEventListener("hashchange", () => { const v = location.hash.slice(1); if (VIEWS.includes(v)) showView(v); });
if (VIEWS.includes(location.hash.slice(1))) showView(location.hash.slice(1));

// ---- models ----
const fmtDate = (ts) => new Date(ts * 1000).toLocaleString();
const fmtSize = (mb) => (mb >= 1000 ? (mb / 1000).toFixed(1) + " GB" : Math.round(mb) + " MB");
async function loadModels() {
  let data; try { data = await api("/api/models"); } catch (e) { return err(e); }
  state.models = data.models;
  const training = data.models.filter((m) => m.status === "training").length;
  $("#models-summary").textContent = `${data.models.length} model(s)${training ? ` · ${training} training` : ""}`;
  const box = $("#models-list"); box.replaceChildren();
  if (!data.models.length) { box.append(el("div", { class: "placeholder" }, "No trained models yet.")); return; }
  for (const m of data.models) box.append(modelCard(m));
}
function modelCard(m) {
  const meta = el("div", { class: "card-meta" });
  const add = (label, val) => meta.append(el("span", {}, `${label}: `, el("b", {}, String(val))));
  add("policy", m.policy || "?");
  add("dataset", m.dataset || "?");
  add("steps", m.steps ?? "?");
  add("checkpoints", m.checkpoints && m.checkpoints.length ? m.checkpoints.join(", ") : "—");
  add("size", fmtSize(m.size_mb || 0));
  add("created", fmtDate(m.created));

  const card = el("div", { class: "card" + (m.status === "training" ? " training" : "") },
    el("div", { class: "card-top" },
      el("span", { class: "card-name" }, m.name),
      el("span", { class: "badge-status " + m.status }, m.status)),
    meta);

  if (m.status === "training") {
    if (m.step && m.total) {
      const pct = Math.round((100 * m.step) / m.total);
      card.append(
        el("div", { class: "card-meta live" }, `step ${m.step}/${m.total} (${pct}%) · loss ${m.loss ?? "?"} · ${m.rate ?? "?"} step/s · ETA ${m.eta ?? "?"}`),
        el("div", { class: "progress" }, el("span", { style: `width:${pct}%` })));
    } else {
      card.append(el("div", { class: "card-meta live" }, `training… loss ${m.loss ?? "?"}`));
    }
  }

  const foot = el("div", { class: "card-foot" });
  if (m.status === "training") {
    foot.append(el("span", { class: "muted" }, "actions available once training finishes"));
  } else {
    foot.append(
      el("a", { class: "btn tiny", href: `/api/models/${encodeURIComponent(m.name)}/download` }, "⬇ download"),
      el("button", { class: "btn tiny", onclick: () => doRenameModel(m.name) }, "rename"),
      el("button", { class: "btn tiny danger", onclick: () => doDeleteModel(m.name) }, "delete"));
  }
  card.append(foot);
  return card;
}
async function doRenameModel(name) {
  const v = await modal({ title: `Rename model "${name}"`, submit: "Rename",
    fields: [{ name: "new_name", label: "new name", type: "text", value: name }] });
  if (!v || !v.new_name || v.new_name === name) return;
  try { await post(`/api/models/${encodeURIComponent(name)}/rename`, { new_name: v.new_name }); ok("renamed"); loadModels(); }
  catch (e) { err(e); }
}
async function doDeleteModel(name) {
  if (!confirm(`Delete model "${name}" and all its checkpoints? This cannot be undone.`)) return;
  try { await post(`/api/models/${encodeURIComponent(name)}/delete`, { confirm: true }); ok(`deleted ${name}`); loadModels(); }
  catch (e) { err(e); }
}
$("#btn-models-refresh").addEventListener("click", loadModels);

// ---- evals ----
async function loadEvals() {
  let data; try { data = await api("/api/evals"); } catch (e) { return err(e); }
  state.evals = data.evals;
  const box = $("#evals-wrap"); box.replaceChildren();
  if (!data.evals.length) {
    box.append(el("div", { class: "placeholder" },
      "No evals yet. Bridge evals are auto-captured here; or add one with “+ New eval”."));
    return;
  }
  for (const ev of data.evals) box.append(evalCard(ev));
}
function evalStars(ev) {
  const cur = ev.rating || 0;
  const box = el("span", { class: "stars" });
  for (let i = 1; i <= 5; i++)
    box.append(el("span", { class: i <= cur ? "on" : "", onclick: () => annotateEval(ev.name, { rating: i === cur ? null : i }) }, "★"));
  return box;
}
function evalCard(ev) {
  const meta = el("div", { class: "card-meta" });
  const add = (l, v) => { if (v != null && v !== "") meta.append(el("span", {}, `${l}: `, el("b", {}, String(v)))); };
  add("model", ev.model); add("checkpoint", ev.checkpoint); add("task", ev.task);
  add("episodes", ev.episodes);
  if (ev.success_rate != null) add("success", Math.round(ev.success_rate * 100) + "%");
  else if (ev.successes != null && ev.episodes) add("success", `${ev.successes}/${ev.episodes}`);
  add("date", fmtDate(ev.created));

  const notes = el("input", { class: "ep-input", value: ev.notes || "", placeholder: "notes…" });
  notes.addEventListener("change", () => annotateEval(ev.name, { notes: notes.value }, true));
  const oper = el("input", { class: "ep-input", value: ev.operator || "", placeholder: "operator" });
  oper.addEventListener("change", () => annotateEval(ev.name, { operator: oper.value }, true));

  const foot = el("div", { class: "card-foot" }, el("span", { class: "muted" }, "rating"), evalStars(ev), notes, oper);
  if (ev.videos && ev.videos.length)
    foot.append(el("button", { class: "btn tiny primary", onclick: () => playEvalVideo(ev.name) }, "▶ video"));
  foot.append(
    el("a", { class: "btn tiny", href: `/api/evals/${encodeURIComponent(ev.name)}/download` }, "⬇ download"),
    el("button", { class: "btn tiny", onclick: () => doRenameEval(ev.name) }, "rename"),
    el("button", { class: "btn tiny danger", onclick: () => doDeleteEval(ev.name) }, "delete"));

  return el("div", { class: "card" },
    el("div", { class: "card-top" },
      el("span", { class: "card-name" }, ev.name),
      el("span", { class: "badge-status " + (ev.source === "auto" ? "done" : "stopped") }, ev.source)),
    meta, foot);
}
async function annotateEval(name, fields, silent) {
  try {
    await post(`/api/evals/${encodeURIComponent(name)}/annotate`, fields);
    const e = state.evals.find((x) => x.name === name); if (e) Object.assign(e, fields);
    if ("rating" in fields) loadEvals();
    if (!silent) ok("saved");
  } catch (e) { err(e); }
}
function playEvalVideo(name) {
  const back = el("div", { class: "modal-back", onclick: (e) => e.target === back && back.remove() },
    el("div", { class: "modal", style: "width:auto;max-width:92vw" },
      el("h3", {}, `${name} — video`),
      el("video", { src: `/api/evals/${encodeURIComponent(name)}/video`, controls: true, autoplay: true,
        style: "max-width:86vw;max-height:70vh;display:block;background:#000" }),
      el("div", { class: "modal-actions" }, el("button", { class: "btn", onclick: () => back.remove() }, "Close"))));
  document.body.append(back);
}
async function doRenameEval(name) {
  const v = await modal({ title: `Rename eval "${name}"`, submit: "Rename",
    fields: [{ name: "new_name", label: "new name", type: "text", value: name }] });
  if (!v || !v.new_name || v.new_name === name) return;
  try { await post(`/api/evals/${encodeURIComponent(name)}/rename`, { new_name: v.new_name }); ok("renamed"); loadEvals(); } catch (e) { err(e); }
}
async function doDeleteEval(name) {
  if (!confirm(`Delete eval "${name}"?`)) return;
  try { await post(`/api/evals/${encodeURIComponent(name)}/delete`, { confirm: true }); ok(`deleted ${name}`); loadEvals(); } catch (e) { err(e); }
}
async function newEval() {
  const models = (state.models || []).filter((m) => m.status !== "training").map((m) => ({ value: m.name, label: m.name }));
  const v = await modal({ title: "New eval", submit: "Create", fields: [
    { name: "name", label: "eval name", type: "text", placeholder: "e.g. hd_bucket_smolvla_run1" },
    (models.length
      ? { name: "model", label: "model", type: "select", options: [{ value: "", label: "—" }, ...models] }
      : { name: "model", label: "model", type: "text" }),
    { name: "task", label: "task", type: "text" },
    { name: "episodes", label: "episodes", type: "number" },
    { name: "successes", label: "successes", type: "number" },
    { name: "operator", label: "operator", type: "text" },
    { name: "notes", label: "notes", type: "text" },
  ] });
  if (!v || !v.name) return;
  if (v.episodes && v.successes) v.success_rate = Number(v.successes) / Number(v.episodes);
  try { await post("/api/evals", v); ok(`created ${v.name}`); loadEvals(); } catch (e) { err(e); }
}
$("#btn-evals-refresh").addEventListener("click", loadEvals);
$("#btn-eval-new").addEventListener("click", newEval);

loadDatasets();
setInterval(() => loadDatasets(true), 4000);          // folder monitor (datasets view)
setInterval(() => { if (state.view === "models") loadModels(); }, 5000);  // live training refresh
