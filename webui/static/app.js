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
  if (!r.ok) {
    const d = body.detail;                     // FastAPI: a string, or a list of validation errors
    throw new Error(typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => `${(x.loc || []).join(".")}: ${x.msg}`).join("; ") : `${r.status} ${r.statusText}`);
  }
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

function captureStr(c) {
  const sq = c.squares || {};
  const parts = ["scene", "wrist", "side"].filter((k) => k in sq)
    .map((k) => `${k} ${sq[k] == null ? "none" : Number(sq[k]).toFixed(2)}`);
  const rec = c.recorded || {};
  const f = (rec.first || "").slice(0, 10), l = (rec.last || rec.first || "").slice(0, 10);
  return "crop: " + (parts.join(" · ") || "—") + (f ? ` · recorded ${f}${l && l !== f ? " .. " + l : ""}` : "");
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
      // camera geometry + recording dates from meta/capture.json (written by the teleop panel).
      // Datasets with different crops cannot be merged or share a checkpoint — that's why it's here.
      el("div", { class: "ds-meta", title: d.capture ? "meta/capture.json" : "recorded before capture.json existed" },
        d.capture ? captureStr(d.capture) : "crop: unknown (pre-capture.json)"),
      el("div", { class: "ds-foot" },
        el("button", { class: "btn tiny primary", onclick: (e) => { e.stopPropagation(); doTrain(d.name); } }, "Train"),
        el("button", { class: "btn tiny", onclick: (e) => { e.stopPropagation(); doRenameDataset(d.name); } }, "rename"),
        el("button", { class: "btn tiny", onclick: (e) => { e.stopPropagation(); doToVideo(d.name); } }, "Export video"),
        el("button", { class: "btn tiny", title: "copy the teleop launch line that appends to this dataset (same crop, storage and task)",
          onclick: (e) => { e.stopPropagation(); copyRecordCmd(d); } }, "record cmd"),
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

// Read-only cells for what the teleop panel recorded. Both are per-episode facts about how the take
// was captured, not opinions like rating/notes, so they are shown rather than edited here.
function graspsCell(ep) {
  const n = ep.grasps;
  // absent = recorded before the flag existed; NOT the same as a clean single grasp, so it reads "—"
  if (!n) return el("span", { class: "muted" }, "—");
  if (n === 1) return el("span", { class: "mono" }, "1");
  const spans = (ep.regrasp_ranges || []).map((r) => `${r[0]}-${r[1]}`).join(", ");
  return el("span", { class: "tag warn", title: spans ? `re-grasp frames ${spans}` : "" }, String(n));
}

function exposureCell(ep) {
  const e = ep.exposure;
  if (!e) return el("span", { class: "muted" }, "auto");
  const parts = Object.keys(e).sort().map((k) => `${k}=${Math.round(e[k])}`);
  // collapse to one number when every camera matched, else show the spread on hover
  const vals = Object.values(e).map(Math.round);
  const same = vals.every((v) => v === vals[0]);
  return el("span", { class: "mono", title: parts.join("  ") }, same ? String(vals[0]) : `${Math.min(...vals)}–${Math.max(...vals)}`);
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
      el("td", { class: "c-num" }, graspsCell(ep)),
      el("td", { class: "c-num" }, exposureCell(ep)),
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

async function doTrain(name) {
  const v = await modal({ title: `Train a policy on "${name}"`, submit: "Start training",
    fields: [
      { name: "policy", label: "policy", type: "select", value: "smolvla", options: [
        { value: "smolvla", label: "SmolVLA (fine-tune pretrained)" },
        { value: "act", label: "ACT (from scratch)" }] },
      { name: "steps", label: "steps", type: "number", value: "20000" },
      { name: "batch_size", label: "batch size", type: "number", value: "64" },
      { name: "save_freq", label: "save every N steps", type: "number", value: "2500",
        hint: "Runs on this host's GPU. One run at a time — watch progress in the Models tab." },
    ] });
  if (!v) return;
  try {
    const r = await post(`/api/datasets/${encodeURIComponent(name)}/train`,
      { policy: v.policy, steps: Number(v.steps), batch_size: Number(v.batch_size),
        save_freq: Number(v.save_freq) });
    ok(`training started: ${r.run}`);
    showView("models");
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
const VIEWS = ["datasets", "models", "evals", "robot", "routes", "map", "services"];
function showView(v) {
  state.view = v;
  VIEWS.forEach((name) => { $("#view-" + name).hidden = name !== v; });
  document.querySelectorAll(".nav-btn").forEach((b) => b.classList.toggle("active", b.dataset.view === v));
  if (v === "models") loadModels();
  if (v === "evals") loadEvals();
  if (v === "robot") loadRobot();
  if (v === "routes") loadRoutesView();
  if (v === "map") loadMapView();
  if (v === "services") loadServicesView();
  // the header's right side belongs to the robot page while that view is up; the dataset
  // monitor / refresh controls only make sense on the dataset views
  const robot = v === "robot", embedded = robot || v === "routes" || v === "map";
  ["#monitor", "#datasets-dir", "#btn-refresh"].forEach((s) => { const e = $(s); if (e) e.hidden = embedded; });
  $("#hdr-robot").hidden = !robot;
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
  // Grouped by model (run/step the server resolved), newest activity first, unassigned last.
  // Each header carries the COMBINED rating: mean stars over the rated runs in the group.
  let groups = []; try { groups = (await api("/api/evals/groups")).groups; } catch (e) { groups = []; }
  const byKey = new Map(); for (const ev of data.evals) { const k = ev.model_key || "unassigned"; if (!byKey.has(k)) byKey.set(k, []); byKey.get(k).push(ev); }
  const order = groups.map((g) => g.model_key).filter((k) => byKey.has(k));
  for (const k of byKey.keys()) if (!order.includes(k)) order.push(k);
  state.evalCollapsed = state.evalCollapsed || {};
  for (const k of order) {
    const g = groups.find((x) => x.model_key === k) || { runs: byKey.get(k).length, rated: 0, rating: null };
    const runs = byKey.get(k);
    const collapsed = !!state.evalCollapsed[k];
    const title = k === "unassigned" ? "unassigned model" : k;
    const stars = g.rating != null
      ? el("span", { class: "group-rating", title: `${g.rated} of ${g.runs} runs rated` },
          "★".repeat(Math.round(g.rating)) + "☆".repeat(5 - Math.round(g.rating)), " ", el("b", {}, g.rating.toFixed(2)))
      : el("span", { class: "muted" }, "no ratings yet");
    const stats = el("span", { class: "group-stats" },
      `${g.runs} run${g.runs === 1 ? "" : "s"} · ${g.rated} rated`,
      g.success_rate != null ? ` · success ${Math.round(g.success_rate * 100)}%` : "",
      g.inferred ? ` · ${g.inferred} inferred` : "");
    const head = el("div", { class: "eval-group-head", onclick: () => { state.evalCollapsed[k] = !collapsed; loadEvals(); } },
      el("span", { class: "group-toggle" }, collapsed ? "▸" : "▾"),
      el("span", { class: "group-title" }, title), stars, stats);
    if (k === "unassigned") {
      head.append(el("button", { class: "btn tiny", onclick: (e) => { e.stopPropagation(); assignGroup(runs); } },
        "assign model to all…"));
    }
    box.append(el("div", { class: "eval-group" + (collapsed ? " collapsed" : "") }, head,
      collapsed ? "" : el("div", { class: "eval-group-body" }, ...runs.map(evalCard))));
  }
}
async function assignGroup(runs) {
  const models = (state.models || []).map((m) => m.name);
  const v = await modal({ title: `Assign a model to ${runs.length} run${runs.length === 1 ? "" : "s"}`, submit: "Assign",
    fields: [
      { name: "checkpoint", label: "checkpoint (train/<run>/checkpoints/<step>)", type: "text",
        placeholder: models.length ? `e.g. train/${models[0]}/checkpoints/020000` : "train/<run>/checkpoints/020000",
        hint: "Sets the same checkpoint on every run in this group. Marked as inferred, so it stays distinguishable from a server-reported value." },
    ] });
  if (!v || !v.checkpoint) return;
  try {
    await post("/api/evals/bulk_annotate", { names: runs.map((r) => r.name),
      fields: { checkpoint: v.checkpoint, checkpoint_inferred: true } });
    ok(`assigned ${runs.length} runs`); loadEvals();
  } catch (e) { err(e); }
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
  // checkpoint is EDITABLE: runs recorded before the server reported what it served carry only an
  // address, and the only way to attribute them is for the operator to say which model ran.
  const ck = el("input", { class: "ep-input mono", value: ev.checkpoint || "", placeholder: "train/<run>/checkpoints/<step>",
    title: ev.checkpoint_inferred ? "inferred/assigned by hand, not reported by the server" : "as reported by the inference server",
    style: "min-width:260px" });
  ck.addEventListener("change", () => annotateEval(ev.name, { checkpoint: ck.value, checkpoint_inferred: !!ck.value }, false));
  meta.append(el("span", {}, "checkpoint: ", ck, ev.checkpoint_inferred ? el("span", { class: "tag warn", title: "assigned by hand" }, "inferred") : ""));
  if (ev.server) add("server", ev.server);
  add("task", ev.task);
  add("episodes", ev.episodes);
  if (ev.success_rate != null) add("success", Math.round(ev.success_rate * 100) + "%");
  else if (ev.successes != null && ev.episodes) add("success", `${ev.successes}/${ev.episodes}`);
  // hardware-run facts from eval_hw.py — read-only, and the reason a run is or isn't comparable
  add("steps", ev.steps);
  if (ev.duration_s != null) add("duration", ev.duration_s + "s");
  if (ev.fps != null) add("fps", ev.fps);
  if (ev.n_action_steps != null) add("n-act", ev.n_action_steps);
  if (ev.exposure && Object.keys(ev.exposure).length) {
    const vals = Object.values(ev.exposure).map(Math.round);
    const same = vals.every((v) => v === vals[0]);
    add("exposure", same ? vals[0] : `${Math.min(...vals)}–${Math.max(...vals)}`);
  }
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
    if ("rating" in fields || "checkpoint" in fields) loadEvals();   // both change the grouping/roll-up
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
// Live registry: the eval uploads each finished run; re-read while the tab is visible and nobody is typing.
setInterval(() => {
  if (state.view !== "evals" || !$("#evals-auto").checked || document.visibilityState !== "visible") return;
  const a = document.activeElement;
  if (a && a.closest && a.closest("#view-evals") && ["INPUT", "TEXTAREA", "SELECT"].includes(a.tagName)) return;
  loadEvals();
}, 15000);

// ---- robot view: the page the robot PC serves (teleop panel / eval), embedded ----
let robotTimer = null;
function robotStatus(r) {
  const s = $("#robot-status"), c = r.config || {};
  if (r.reachable) {
    s.textContent = `● ${c.mode || "up"} — ${c.title || r.url}`; s.className = "mono ok";
  } else {
    s.textContent = "○ not running"; s.className = "mono muted";
  }
  $("#robot-hint").hidden = r.reachable;
  $("#robot-frame").hidden = !r.reachable;
}
async function loadRobot() {
  let r; try { r = await api("/api/robot"); } catch (e) { return err(e); }
  const inp = $("#robot-url"); if (!inp.value) inp.value = r.url;
  const url = inp.value || r.url, fr = $("#robot-frame");
  if (r.reachable && fr.dataset.src !== url) { fr.src = url; fr.dataset.src = url; }
  robotStatus(r);
  clearInterval(robotTimer);
  robotTimer = setInterval(async () => {
    if (state.view !== "robot") return;
    try {
      const rr = await api("/api/robot");
      const was = $("#robot-frame").hidden;
      robotStatus(rr);
      if (rr.reachable && was) { fr.src = url; fr.dataset.src = url; }   // came up: load it
    } catch (e) { /* keep the last status */ }
  }, 5000);
}
$("#robot-url").addEventListener("change", () => { const fr = $("#robot-frame"); fr.src = $("#robot-url").value; fr.dataset.src = fr.src; });

// ---- routes view: the route editor the robot PC serves (map + live pose + waypoints), embedded ----
let routeTimer = null;
function routeStatus(r) {
  const s = $("#route-status"), i = r.info || {};
  if (r.reachable) {
    const p = i.pose ? `robot (${i.pose.x.toFixed(2)}, ${i.pose.y.toFixed(2)})` : `no robot pose: ${i.ros || "—"}`;
    s.textContent = `● route editor — map ${i.map || "?"} · ${p}`; s.className = "mono ok";
  } else {
    s.textContent = "○ route editor not running"; s.className = "mono muted";
  }
  $("#route-hint").hidden = r.reachable;
  $("#route-frame").hidden = !r.reachable;
}
async function loadRoutesView() {
  let r; try { r = await api("/api/route_editor"); } catch (e) { return err(e); }
  const inp = $("#route-url"); if (!inp.value) inp.value = r.url;
  const url = inp.value || r.url, fr = $("#route-frame");
  $("#route-open").href = url;
  if (r.reachable && fr.dataset.src !== url) { fr.src = url; fr.dataset.src = url; }
  routeStatus(r);
  clearInterval(routeTimer);
  routeTimer = setInterval(async () => {
    if (state.view !== "routes") return;
    try {
      const rr = await api("/api/route_editor");
      const was = $("#route-frame").hidden;
      routeStatus(rr);
      if (rr.reachable && was) { fr.src = url; fr.dataset.src = url; }
    } catch (e) { /* keep the last status */ }
  }, 5000);
}
$("#route-url").addEventListener("change", () => { const fr = $("#route-frame"); fr.src = $("#route-url").value; fr.dataset.src = fr.src; $("#route-open").href = fr.src; });

// ---- map view: the route editor's map mode (map + robot + lidar scans), embedded ----
let mapTimer = null;
async function loadMapView() {
  let r; try { r = await api("/api/route_editor?scope=map"); } catch (e) { return err(e); }
  const url = r.url + "3d", fr = $("#map-frame"), s = $("#map-status");   // the 3D view by default; its "2D map" button switches
  document.querySelectorAll('input[name="map-source"]').forEach((x) => { x.checked = x.value === r.source; });
  $("#map-open").href = url;
  const show = (rr) => {
    const i = rr.info || {};
    if (rr.reachable) { s.textContent = `● ${rr.source === "sim" ? "SIM · " : ""}map ${i.map || "?"} · ${i.pose ? `robot (${i.pose.x.toFixed(2)}, ${i.pose.y.toFixed(2)})` : "no robot pose: " + (i.ros || "—")}`; s.className = "mono ok"; }
    else { s.textContent = "○ map page not running"; s.className = "mono muted"; }
    $("#map-hint").hidden = rr.reachable; fr.hidden = !rr.reachable;
    if (rr.reachable && fr.dataset.src !== url) { fr.src = url; fr.dataset.src = url; }
  };
  show(r);
  clearInterval(mapTimer);
  mapTimer = setInterval(async () => { if (state.view !== "map") return; try { show(await api("/api/route_editor?scope=map")); } catch (e) { /* keep */ } }, 5000);
}

// the Map tab's live / sim switch: which route editor the Map tab embeds (its 3D button stays inside it)
document.querySelectorAll('input[name="map-source"]').forEach((r) => r.addEventListener("change", async () => {
  if (!r.checked) return;
  const s = $("#map-status");
  if (r.value === "sim") { s.textContent = "● starting the simulator on the manager's host…"; s.className = "mono muted"; }
  let d; try { d = await api("/api/map_source", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ mode: r.value }) }); } catch (e) { return err(e); }
  if (d.start && d.start.error) err(new Error(d.start.error));
  { const fr = $("#map-frame"); fr.removeAttribute("src"); delete fr.dataset.src; fr.hidden = true; }
  if (state.view === "map") loadMapView();
}));

// ---- Services view: supervisor's units on the robot PC, native (XML-RPC through the manager) ----
let servicesTimer = null;
const svcOpenLogs = {};                      // name -> "stdout" | "stderr" (log rows kept open across refreshes)
function svcUptime(s) { if (s == null) return ""; const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60); return h ? `${h}h ${m}m` : m ? `${m}m ${s % 60}s` : `${s}s`; }
async function svcAction(name, action) {
  try { await post(`/api/services/${encodeURIComponent(name)}`, { action }); toast(`${action}: ${name}`, "ok"); }
  catch (e) { err(e); }
  renderServices();
}
async function svcLog(name, stream, pre) {
  pre.textContent = "loading…";
  try { const d = await api(`/api/services/${encodeURIComponent(name)}/log?stream=${stream}&lines=120`); pre.textContent = d.lines.join("\n") || "(empty)"; pre.scrollTop = pre.scrollHeight; }
  catch (e) { pre.textContent = String(e); }
}
async function renderServices() {
  let d; try { d = await api("/api/services"); } catch (e) { return err(e); }
  const s = $("#services-status");
  $("#services-open").href = d.url.replace(/\/RPC2$/, "/");
  const job = $("#services-job");
  if (d.job && (d.job.running || (d.job.log || []).length)) { job.hidden = false; job.textContent = (d.job.running ? "restarting base stack: " : "last base restart: ") + (d.job.log || []).join(" · "); job.classList.toggle("on", d.job.running); }
  else job.hidden = true;
  $("#btn-restart-base").disabled = !d.reachable || !!(d.job && d.job.running);
  if (!d.reachable) { s.textContent = "○ supervisor not reachable (robot off?)"; s.className = "mono muted"; $("#services-wrap").replaceChildren(el("div", { class: "muted", style: "padding: 14px" }, d.error || "")); return; }
  const n = d.services.length, run = d.services.filter((x) => x.state === "RUNNING").length;
  s.textContent = `● ${run} of ${n} running`; s.className = "mono ok";
  const tbody = el("tbody");
  for (const x of d.services) {
    const st = x.state.toLowerCase();
    const can = { start: x.state !== "RUNNING" && x.state !== "STARTING", stop: x.state === "RUNNING" || x.state === "STARTING", restart: true };
    const logBtn = el("button", { class: "btn", title: "tail of the unit's log" }, svcOpenLogs[x.name] ? "hide log" : "log");
    const row = el("tr", { class: "svc-row" },
      el("td", { class: "name" }, x.name),
      el("td", {}, el("span", { class: `badge-status ${st}` }, x.state)),
      el("td", { class: "mono muted" }, x.pid ? `pid ${x.pid} · ${svcUptime(x.uptime_s)}` : (x.description || "")),
      el("td", { class: "acts" },
        el("button", { class: "btn", disabled: !can.start, onclick: () => svcAction(x.name, "start") }, "start"),
        el("button", { class: "btn", disabled: !can.stop, onclick: () => svcAction(x.name, "stop") }, "stop"),
        el("button", { class: "btn", disabled: !can.restart, onclick: () => svcAction(x.name, "restart") }, "restart"),
        logBtn));
    tbody.appendChild(row);
    if (svcOpenLogs[x.name]) {
      const pre = el("pre", { class: "svc-log" });
      const bar = el("div", { class: "svc-logbar" },
        ...["stdout", "stderr"].map((sname) => el("button", { class: "btn" + (svcOpenLogs[x.name] === sname ? " active" : ""), onclick: () => { svcOpenLogs[x.name] = sname; svcLog(x.name, sname, pre); renderServices(); } }, sname)),
        el("button", { class: "btn", onclick: () => svcLog(x.name, svcOpenLogs[x.name], pre) }, "↻"),
        el("span", { class: "muted" }, x[svcOpenLogs[x.name]] || ""));
      tbody.appendChild(el("tr", { class: "svc-log" }, el("td", { colspan: 4 }, bar, pre)));
      svcLog(x.name, svcOpenLogs[x.name], pre);
    }
    logBtn.onclick = () => { if (svcOpenLogs[x.name]) delete svcOpenLogs[x.name]; else svcOpenLogs[x.name] = "stdout"; renderServices(); };
  }
  const table = el("table", { class: "svc" }, el("thead", {}, el("tr", {}, el("th", {}, "unit"), el("th", {}, "state"), el("th", {}, "pid · uptime"), el("th", {}, ""))), tbody);
  $("#services-wrap").replaceChildren(table);
}
async function loadServicesView() {
  await renderServices();
  clearInterval(servicesTimer);
  servicesTimer = setInterval(() => { if (state.view === "services" && !Object.keys(svcOpenLogs).length) renderServices(); }, 4000);
}
$("#btn-services-refresh").onclick = renderServices;
$("#btn-restart-base").onclick = async () => { if (!confirm("Restart robot_disco, then localization, navigation and the collision monitor, in order (about 45 s)? The robot must be stationary.")) return; try { await post("/api/services/restart_base", {}); } catch (e) { err(e); } renderServices(); };

// ---- "record cmd": the exact teleop launch line that appends to a dataset ----
function recordCmd(d) {
  const sq = d.capture && d.capture.squares ? d.capture.squares : null;
  const task = d.tasks && d.tasks.length ? d.tasks[0] : "";
  let cmd = `./run.sh -m ur5e_lerobot.teleop.manual_panel --engine hardware --cameras udp --input spacemouse --resume` +
    ` --root outputs/datasets/${d.name} --repo_id local/${d.name} --task "${task.replace(/"/g, '\\"')}"`;
  if (d.media === "video") cmd += " --use-videos";
  if (sq) { if (sq.scene != null) cmd += ` --scene-square ${sq.scene}`; if (sq.side != null) cmd += ` --side-square ${sq.side}`; }
  return cmd;
}
async function copyRecordCmd(d) {
  const cmd = recordCmd(d);
  try { await navigator.clipboard.writeText(cmd); ok("record command copied"); }
  catch (e) { window.prompt("copy the launch line:", cmd); }
}

loadDatasets();
setInterval(() => loadDatasets(true), 4000);          // folder monitor (datasets view)
setInterval(() => { if (state.view === "models") loadModels(); }, 5000);  // live training refresh
