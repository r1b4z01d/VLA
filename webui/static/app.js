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
  state.datasetsDir = data.datasets_dir || "";          // the header no longer shows it
  const changed = state.signature && data.signature !== state.signature;
  state.signature = data.signature;
  state.datasets = data.datasets;
  renderDatasets();
  if (fromMonitor && changed && state.selected) loadEpisodes(state.selected);   // the folder monitor still refreshes silently
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
        hint: "rsync from rd@192.168.11.80:~/rd_ws/VLA/outputs/datasets/<name>" },
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

// (declared BEFORE the view switcher below: it calls loadHandView() synchronously at load when the URL hash
//  is #hand, and these consts must already exist — `let`/`const` are not hoisted like the functions are)
// ---- Hand view: the AmazingHand over UDP, through the manager (/api/hand*) ----
// Sliders are by PHYSICAL finger. The four curl channels the API takes (the dataset / schema order, named
// index, middle, ring, thumb) do not drive the fingers their names suggest on this hand — measured 2026-10-26
// for the Stream Deck keys (ur5e_lerobot/teleop/manual_panel.py): channel 0 moves the middle finger, 1 the
// pinky, 2 the index, 3 the thumb. The channel order itself stays: recorded datasets replay through it.
const HAND_FINGERS = [["thumb", 3], ["index", 2], ["middle", 0], ["pinky", 1]];   // [label, curl channel]
const HAND_CH = Object.fromEntries(HAND_FINGERS);                                // label -> channel
const handPoseOf = (open) => { const c = [1, 1, 1, 1]; for (const [f, v] of Object.entries(open)) c[HAND_CH[f]] = v; return c; };
const HAND_POSES = { open: [0, 0, 0, 0], half: [0.5, 0.5, 0.5, 0.5], close: [1, 1, 1, 1],
                     point: handPoseOf({ index: 0 }), pinch: handPoseOf({ index: 0.75, thumb: 0.8 }), thumbs: handPoseOf({ thumb: 0 }) };
const handS = { curls: [0, 0, 0, 0], abduct: [0, 0, 0, 0], timer: null, pending: null, lastSend: 0, gesturesBuilt: false };
const handHost = () => $("#hand-host").value.trim();
function handLog(line) {
  const pre = $("#hand-log");
  pre.textContent = [new Date().toLocaleTimeString() + "  " + line, ...pre.textContent.split("\n")].filter((x) => x).slice(0, 10).join("\n");
}
function handRow(name, idx) {
  // idx: the finger's curl channel (see HAND_FINGERS); null: the "all" row, which drives every finger at once
  const row = el("div", { class: "hand-r" + (idx === null ? " all" : "") });
  const curl = el("input", { type: "range", min: 0, max: 100, value: 0, title: "curl: 0 open … 1 closed" });
  const cval = el("span", { class: "val" }, "0.00");
  const abd = idx === null ? el("span", { class: "muted", style: "font-size: 11px" }, "—") : el("input", { type: "range", min: -100, max: 100, value: 0, title: "abduction: -1 … 1" });
  const aval = el("span", { class: "val" }, idx === null ? "" : "+0.00");
  curl.addEventListener("input", () => {
    const c = curl.value / 100;
    if (idx === null) handS.curls = [c, c, c, c]; else handS.curls[idx] = c;
    handSyncSliders(); handSendCurls();
  });
  curl.addEventListener("change", () => handSendCurls(true));
  if (idx !== null) {
    abd.addEventListener("input", () => { handS.abduct[idx] = abd.value / 100; handSyncSliders(); handSendCurls(); });
    abd.addEventListener("change", () => handSendCurls(true));
  }
  row.append(el("span", { class: "fname" }, name), curl, cval, abd, aval);
  row._set = (c, a) => {
    curl.value = Math.round(c * 100); cval.textContent = c.toFixed(2);
    if (idx !== null) { abd.value = Math.round(a * 100); aval.textContent = (a >= 0 ? "+" : "") + a.toFixed(2); }
  };
  return row;
}
function buildHandGrid() {
  const g = $("#hand-grid");
  if (g.childElementCount) return;
  g.append(el("div", { class: "hand-r hdr" }, el("span"), el("span", {}, "curl"), el("span"), el("span", {}, "abduction"), el("span")));
  handS.rows = [handRow("all", null), ...HAND_FINGERS.map(([f, ch]) => handRow(f, ch))];
  handS.rows.forEach((r) => g.append(r));
  handSyncSliders();
}
function handSyncSliders() {
  if (!handS.rows) return;
  const same = handS.curls.every((c) => Math.abs(c - handS.curls[0]) < 1e-6);
  handS.rows[0]._set(same ? handS.curls[0] : handS.curls.reduce((a, b) => a + b, 0) / 4, 0);
  HAND_FINGERS.forEach(([f, ch], i) => handS.rows[i + 1]._set(handS.curls[ch], handS.abduct[ch]));
}
async function handSendCurls(now = false) {
  // while dragging, at most ~14 lines/s (the teleop stream's order of magnitude); the last value always goes out
  const dt = performance.now() - handS.lastSend;
  if (!now && dt < 70) { clearTimeout(handS.pending); handS.pending = setTimeout(() => handSendCurls(true), 70 - dt); return; }
  clearTimeout(handS.pending); handS.lastSend = performance.now();
  const body = { curls: handS.curls.slice(), abduct: handS.abduct.slice(), speed: +$("#hand-speed").value || 200, host: handHost() || undefined };
  try { const r = await post("/api/hand/curls", body); handLog(r.line); } catch (e) { err(e); }
}
function handPose(name) { handS.curls = HAND_POSES[name].slice(); handS.abduct = [0, 0, 0, 0]; handSyncSliders(); handSendCurls(true); }
async function handGesture(c, label) {
  try { await post("/api/hand/gesture", { line: c, host: handHost() || undefined }); handLog(`${c}  (${label})`); } catch (e) { err(e); }
}
function handUptime(ms) { if (ms == null) return ""; const s = Math.floor(ms / 1000), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60); return h ? `${h}h ${m}m` : m ? `${m}m ${s % 60}s` : `${s}s`; }
async function handStatus() {
  const st = $("#hand-status");
  let d; try { d = await api("/api/hand" + (handHost() ? "?host=" + encodeURIComponent(handHost()) : "")); } catch (e) { st.textContent = "○ " + e.message; st.className = "mono muted"; return; }
  $("#hand-console").href = d.console_url;
  $("#hand-host").placeholder = d.host;
  const recent = !d.alive && d.last_alive_s != null && d.last_alive_s < 90;   // one lost probe on a lossy link is not a dead hand
  st.textContent = d.alive ? `● ${d.host} · up ${handUptime(d.uptime_ms)} · ${d.state.sent} sent${d.state.errors ? ` · ${d.state.errors} failed` : ""}`
    : recent ? `◐ ${d.host}: probe lost (answered ${Math.round(d.last_alive_s)} s ago)` : `○ ${d.host}: ${d.error}`;
  st.className = d.alive ? "mono ok" : recent ? "mono warn" : "mono muted";
  if (!handS.gesturesBuilt) {
    handS.gesturesBuilt = true;
    $("#hand-gestures").replaceChildren(...Object.entries(d.gestures).map(([c, label]) => el("button", { class: "btn", title: `sends "${c}"`, onclick: () => handGesture(c, label) }, `${c} · ${label}`)));
    const sp = $("#hand-speed"); sp.min = d.speed_range[0]; sp.max = d.speed_range[1];
  }
  if (!handS.seeded && d.state.last) { handS.seeded = true; handS.curls = d.state.curls.slice(); handS.abduct = d.state.abduct.slice(); $("#hand-speed").value = d.state.speed; handSyncSliders(); }
}
async function loadHandView() {
  if (!$("#hand-host").value) $("#hand-host").value = localStorage.getItem("handHost") || "";
  buildHandGrid();
  await handStatus();
  clearInterval(handS.timer);
  handS.timer = setInterval(() => { if (state.view === "hand") handStatus(); }, 5000);
}
$("#hand-host").addEventListener("change", () => { localStorage.setItem("handHost", handHost()); handStatus(); });
$("#btn-hand-probe").onclick = handStatus;
document.querySelectorAll("#view-hand [data-pose]").forEach((b) => b.addEventListener("click", () => handPose(b.dataset.pose)));
$("#btn-hand-send").onclick = async () => {
  const line = $("#hand-raw").value.trim(); if (!line) return;
  try { const r = await post("/api/hand/raw", { line, host: handHost() || undefined }); handLog(line + (r.reply ? "  →  " + r.reply : "")); }
  catch (e) { err(e); }
};
$("#hand-raw").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#btn-hand-send").click(); });

// ---- Landmarks view: named places the robot can go to (the route editor on the robot keeps them; /api/landmarks* relays) ----
// (above the view switcher for the same reason as the Hand block: loadLandmarksView() can run at load)
const lmS = { last: null, sig: "", map: null, mapImg: null, mapSrc: null, models: null };
let lmTimer = null;
const lmDeg = (yaw) => `${(yaw * 180 / Math.PI).toFixed(0)}°`;
const lmPoseText = (p) => `${p.x.toFixed(2)}, ${p.y.toFixed(2)}, ${lmDeg(p.yaw)}`;
function lmMapThumb(lm) {
  // a 4 m window of the nav map around the landmark, the landmark drawn at the centre with its heading
  const c = el("canvas", { width: 160, height: 160, class: "lm-map", title: `map ${lm.map} around (${lm.x.toFixed(2)}, ${lm.y.toFixed(2)})` });
  const draw = () => {
    const m = lmS.map, im = lmS.mapImg, g = c.getContext("2d");
    if (!m || !im || !im.complete || !im.naturalWidth) return;
    const px = (lm.x - m.origin[0]) / m.resolution, py = m.height - (lm.y - m.origin[1]) / m.resolution;   // map pixel of the landmark
    const half = 2.0 / m.resolution;                                                                  // 2 m each way
    g.fillStyle = "#15171c"; g.fillRect(0, 0, 160, 160);
    g.imageSmoothingEnabled = false;
    g.drawImage(im, px - half, py - half, 2 * half, 2 * half, 0, 0, 160, 160);
    g.strokeStyle = "#4da3ff"; g.lineWidth = 3; g.lineCap = "round";
    g.beginPath(); g.moveTo(80, 80); g.lineTo(80 + Math.cos(lm.yaw) * 30, 80 - Math.sin(lm.yaw) * 30); g.stroke();
    g.fillStyle = "#4da3ff"; g.beginPath(); g.arc(80, 80, 6, 0, 2 * Math.PI); g.fill();
    g.strokeStyle = "#fff"; g.lineWidth = 1.5; g.stroke();
  };
  if (lmS.mapImg && lmS.mapImg.complete) draw(); else if (lmS.mapImg) lmS.mapImg.addEventListener("load", draw);
  return c;
}
function lmCard(lm, d) {
  const here = lm.map === d.map;
  const t = encodeURIComponent(lm.updated || "");
  const photo = (file, label) => file
    ? el("a", { href: `/api/landmarks/img/${file}?t=${t}`, target: "_blank", title: `${label} — open full size` },
        el("img", { class: "lm-img", src: `/api/landmarks/img/${file}?t=${t}`, alt: label }))
    : el("div", { class: "lm-img lm-none muted" }, `no ${label} photo`);
  const cams = lm.cameras || [];
  return el("div", { class: "card lm-card" + (here ? "" : " other-map") },
    el("div", { class: "card-top" },
      el("span", { class: "card-name" }, lm.name),
      el("span", { class: "tag" + (here ? "" : " warn"), title: here ? "the map the robot runs now" : `the robot runs map ${d.map}` }, lm.map || "?"),
      el("span", { class: "mono muted lm-pose", title: "x, y (m) and yaw in the map frame" }, lmPoseText(lm))),
    el("div", { class: "lm-imgs" }, lmMapThumb(lm), photo(lm.images && lm.images.stitched, "side + scene"), photo(lm.images && lm.images.realsense, "RealSense")),
    lmTasks(lm, d),
    el("div", { class: "card-foot" },
      el("button", { class: "btn primary", disabled: !here || !d.ros_up, title: here ? "send the robot there (nav2 goal)" : "not on the map the robot runs", onclick: () => lmGoto(lm) }, "Go"),
      el("button", { class: "btn", onclick: () => lmRename(lm) }, "Rename…"),
      el("button", { class: "btn", title: "take the photos again from where the robot is now", onclick: () => lmRecapture(lm, false) }, "↻ Photos"),
      el("button", { class: "btn", title: "update this landmark to where the robot is now (pose and photos)", onclick: () => lmRecapture(lm, true) }, "⌖ Update"),
      el("span", { class: "muted lm-when" }, `${lm.created || ""}${cams.length && cams.length < 3 ? " · photos: " + cams.join(", ") : ""}`),
      el("button", { class: "btn danger", onclick: () => lmDelete(lm) }, "Delete")));
}
// ---- a landmark's tasks: "pick up tigger" = drive here, then run this model with this instruction (an eval).
// The voice assistant on the robot reads the same file; ▶ Run hands it the job (it drives first, then runs).
const LM_DONE = { none: "time / stillness", closed: "hand closed (a pick)", open: "hand opened (a place)" };
function lmTaskSummary(t) {
  const bits = [`${t.model}`, `“${t.task}”`, `${t.seconds || 120} s`];
  if (t.still) bits.push(`still ${t.still} s`);
  if (t.done_when && t.done_when !== "none") bits.push(`ends: ${LM_DONE[t.done_when] || t.done_when}`);
  if (t.keep_grasp) bits.push("keep grasp");
  if (t.home === false) bits.push("no home");
  if (t.still_mode === "fingers") bits.push("fingers still");
  if (t.extra) bits.push(t.extra);
  return bits.join(" · ");
}
function lmTasks(lm, d) {
  const here = lm.map === d.map, canRun = here && d.ros_up && (d.voice || d.source === "sim");
  const runTitle = !here ? "not on the map the robot runs" : !d.ros_up ? "the editor's ROS node is down"
    : !d.voice && d.source !== "sim" ? "the voice assistant (robot_voice) is not running — Services tab" : "drive there, then run the policy (the voice assistant does both)";
  const rows = (lm.tasks || []).map((t) => el("div", { class: "lm-task" },
    el("button", { class: "btn tiny primary", disabled: !canRun, title: runTitle, onclick: () => lmTaskRun(lm, t) }, "▶ Run"),
    el("span", { class: "lm-task-name", title: "what you say: “Robot Disco, " + t.name + "”" }, t.name),
    el("span", { class: "mono muted lm-task-sum", title: lmTaskSummary(t) }, lmTaskSummary(t)),
    el("button", { class: "btn tiny", title: "edit", onclick: () => lmTaskEdit(lm, t) }, "✎"),
    el("button", { class: "btn tiny danger", title: "remove this task", onclick: () => lmTaskDelete(lm, t) }, "×")));
  return el("div", { class: "lm-tasks" },
    el("div", { class: "lm-tasks-head" },
      el("span", { class: "muted" }, rows.length ? `Tasks here (say “Robot Disco, <task>”)` : "No tasks here yet"),
      el("button", { class: "btn tiny", title: "tie a policy run to this place", onclick: () => lmTaskEdit(lm, null) }, "+ Task")),
    ...rows);
}
async function lmModels() {
  if (lmS.models) return lmS.models;
  try { lmS.models = ((await api("/api/models")).models || []).map((m) => m.name).sort(); } catch (e) { lmS.models = []; }
  return lmS.models;
}
async function lmTaskEdit(lm, t) {
  const models = await lmModels();
  const cur = t || { seconds: 60, still: 3, done_when: "none", keep_grasp: false, home: true };
  const modelField = models.length && (!cur.model || models.includes(cur.model))
    ? { name: "model", label: "Model (training run to serve)", type: "select", value: cur.model || models[0], options: models.map((m) => ({ value: m, label: m })) }
    : { name: "model", label: "Model (training run, run@step or checkpoint path)", value: cur.model || "", placeholder: "tigger_1_merged_smolvla_1" };
  const r = await modal({
    title: t ? `Edit task “${t.name}” at ${lm.name}` : `New task at ${lm.name}`,
    fields: [
      { name: "name", label: "What you say", value: cur.name || "", placeholder: "pick up tigger", hint: "“Robot Disco, pick up tigger” drives here and runs this." },
      modelField,
      { name: "task", label: "Instruction (the policy's task string)", value: cur.task || "", placeholder: "Grab and hold tigger" },
      { name: "seconds", label: "Max seconds", type: "number", value: cur.seconds ?? 60 },
      { name: "still", label: "Still seconds (0 = off)", type: "number", value: cur.still ?? 0, hint: "the run ends once nothing has moved for this long" },
      { name: "done_when", label: "Ends when", type: "select", value: cur.done_when || "none",
        options: [{ value: "none", label: "time or stillness only" }, { value: "closed", label: "the hand has closed and stays closed (a pick)" }, { value: "open", label: "the hand has opened after a grasp (a place)" }] },
      { name: "keep_grasp", label: "Keep the grasp (carry the object; UR stays powered)", type: "checkbox", value: !!cur.keep_grasp },
      { name: "home", label: "Return the arm home at the end", type: "checkbox", value: cur.home !== false },
      { name: "extra", label: "Extra vla_action.py arguments", value: cur.extra || "", placeholder: "--fps 12" },
    ],
    submit: t ? "Save" : "Add task",
  });
  if (!r) return;
  const body = { name: r.name.trim(), model: r.model.trim(), task: r.task.trim(), seconds: parseFloat(r.seconds) || 120,
    still: parseFloat(r.still) || 0, done_when: r.done_when, keep_grasp: !!r.keep_grasp, home: !!r.home, extra: r.extra.trim() };
  try {
    await post(`/api/landmarks/${encodeURIComponent(lm.id)}/tasks${t ? "/" + encodeURIComponent(t.id) : ""}`, body);
    ok(`${t ? "saved" : "added"}: ${body.name} at ${lm.name}`);
  } catch (e) { err(e); }
  lmRefresh();
}
async function lmTaskRun(lm, t) {
  if (!confirm(`Run “${t.name}”?\n\nThe robot drives to “${lm.name}” (${lmPoseText(lm)}) by itself, then serves ${t.model} and runs “${t.task}” on the arm. Be ready to HALT (Map tab) or say “Robot Disco, stop”.`)) return;
  try {
    const r = await post(`/api/landmarks/${encodeURIComponent(lm.id)}/tasks/${encodeURIComponent(t.id)}/run`, {});
    ok(r.sim ? `simulation: ${t.name} would run at ${lm.name}` : `sent to the voice assistant: ${t.name} at ${lm.name} — it drives there first`);
  } catch (e) { err(e); }
}
async function lmTaskDelete(lm, t) {
  if (!confirm(`Remove task “${t.name}” from ${lm.name}?`)) return;
  try { await api(`/api/landmarks/${encodeURIComponent(lm.id)}/tasks/${encodeURIComponent(t.id)}`, { method: "DELETE" }); ok(`removed ${t.name}`); } catch (e) { err(e); }
  lmRefresh();
}
function lmStatus(d) {
  const s = $("#lm-status"), n = (d.landmarks || []).length, sim = d.source === "sim" ? "SIM · " : "";
  const voice = d.source === "sim" ? "" : d.voice ? " · voice ✓" : " · voice assistant off";
  if (d.pose) { s.textContent = `● ${sim}map ${d.map} · robot ${lmPoseText(d.pose)} · ${n} landmark${n === 1 ? "" : "s"}${voice}`; s.className = "mono ok"; }
  else { s.textContent = `◐ ${sim}map ${d.map} · no robot pose (${d.ros || "—"}) · ${n} landmark${n === 1 ? "" : "s"}${voice}`; s.className = "mono warn"; }
  const cams = d.cameras || {}, dead = Object.keys(cams).filter((k) => !cams[k]), b = $("#btn-lm-add");
  b.disabled = !d.pose;
  b.title = !d.pose ? "no robot pose: localization not running or not initialised"
    : dead.length ? `no frame from ${dead.join(", ")} right now — the photos will lack them` : "save the robot's current map pose with photos from the cameras";
}
function lmRender(d) {
  const sig = JSON.stringify([d.map, d.ros_up, d.voice, d.source, d.landmarks]);
  if (sig === lmS.sig) return;
  lmS.sig = sig;
  const items = d.landmarks || [];
  $("#lm-list").replaceChildren(...(items.length ? items.map((lm) => lmCard(lm, d))
    : [el("div", { class: "placeholder" }, "No landmarks yet. Drive the robot somewhere, type a name and press “+ Add at robot”.")]));
}
async function lmLoadMap(source) {
  if (lmS.mapSrc === source && lmS.map) return;
  try { lmS.map = await api("/api/landmarks/map"); } catch (e) { lmS.map = null; return; }
  lmS.mapSrc = source;
  const im = new Image();
  im.addEventListener("load", () => { lmS.sig = ""; if (lmS.last) lmRender(lmS.last); });
  im.src = `/api/landmarks/map.png?src=${source}&t=${Date.now()}`;
  lmS.mapImg = im;
}
async function lmRefresh() {
  let d;
  try { d = await api("/api/landmarks"); }
  catch (e) { const s = $("#lm-status"); s.textContent = "○ " + e.message; s.className = "mono muted"; $("#btn-lm-add").disabled = true; return; }
  lmS.last = d;
  await lmLoadMap(d.source);
  lmStatus(d); lmRender(d);
}
function loadLandmarksView() {
  lmRefresh();
  clearInterval(lmTimer);
  lmTimer = setInterval(() => { if (state.view === "landmarks") lmRefresh(); }, 5000);
}
async function lmAdd() {
  const name = $("#lm-name").value.trim();
  if (!name) { $("#lm-name").focus(); return err("give the landmark a name"); }
  const b = $("#btn-lm-add"); b.disabled = true; b.textContent = "capturing…";
  try {
    const r = await post("/api/landmarks", { name });
    ok(`added ${r.landmark.name} at ${lmPoseText(r.landmark)}${r.missing && r.missing.length ? " — no frame from " + r.missing.join(", ") : ""}`);
    $("#lm-name").value = "";
  } catch (e) { err(e); }
  b.textContent = "+ Add at robot"; b.disabled = false;
  lmRefresh();
}
async function lmGoto(lm) {
  if (!confirm(`Send the robot to “${lm.name}” (${lmPoseText(lm)})? It drives by itself — be ready to HALT (Map tab).`)) return;
  try { await post(`/api/landmarks/${encodeURIComponent(lm.id)}/goto`, {}); ok(`goal sent: ${lm.name}`); } catch (e) { err(e); }
}
async function lmRename(lm) {
  const r = await modal({ title: `Rename “${lm.name}”`, fields: [{ name: "name", label: "Name", value: lm.name }], submit: "Rename" });
  if (!r || !r.name.trim() || r.name.trim() === lm.name) return;
  try { await post(`/api/landmarks/${encodeURIComponent(lm.id)}`, { name: r.name.trim() }); ok("renamed"); } catch (e) { err(e); }
  lmRefresh();
}
async function lmRecapture(lm, pose) {
  if (pose && !confirm(`Update “${lm.name}” to where the robot stands now, and retake its photos?`)) return;
  try {
    const r = await post(`/api/landmarks/${encodeURIComponent(lm.id)}/recapture`, { pose, images: true });
    ok(`${pose ? "updated" : "photos retaken"}: ${lm.name}${r.missing && r.missing.length ? " — no frame from " + r.missing.join(", ") : ""}`);
  } catch (e) { err(e); }
  lmRefresh();
}
async function lmDelete(lm) {
  if (!confirm(`Delete landmark “${lm.name}”?`)) return;
  try { await api(`/api/landmarks/${encodeURIComponent(lm.id)}`, { method: "DELETE" }); ok(`deleted ${lm.name}`); } catch (e) { err(e); }
  lmRefresh();
}
$("#btn-lm-add").addEventListener("click", lmAdd);
$("#btn-lm-refresh").addEventListener("click", () => { lmS.sig = ""; lmRefresh(); });
$("#lm-name").addEventListener("keydown", (e) => { if (e.key === "Enter") lmAdd(); });

// ---- view switching (Datasets / Models / Evals) ----
const VIEWS = ["datasets", "models", "evals", "robot", "hand", "routes", "landmarks", "map", "maps", "services", "folks"];
function showView(v) {
  state.view = v;
  VIEWS.forEach((name) => { $("#view-" + name).hidden = name !== v; });
  document.querySelectorAll(".nav-btn").forEach((b) => b.classList.toggle("active", b.dataset.view === v));
  if (v === "models") loadModels();
  if (v === "evals") loadEvals();
  if (v === "robot") loadRobot();
  if (v === "routes") loadRoutesView();
  if (v === "map") { loadMapView(); if (typeof actlog !== "undefined" && actlog) actlogRefresh(); }
  if (v === "services") loadServicesView();
  if (v === "folks") loadFolksView();
  if (v === "hand") loadHandView();
  if (v === "landmarks") loadLandmarksView();
  if (v === "maps") loadMapsView();
  // the header's right side belongs to the robot page while that view is up; the dataset
  // monitor / refresh controls only make sense on the dataset views
  const robot = v === "robot";
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
  if (ev.landmark) meta.append(el("span", { title: "the landmark (named place) the robot drove to for this run" }, "at: ", el("b", {}, "⌖ " + ev.landmark)));
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

// ---- folks view: the people the robot knows (robot_face's Folks page on the robot PC, port 8096), embedded ----
let folksTimer = null;
function folksStatus(r) {
  const s = $("#folks-status"), i = r.info || {};
  if (r.reachable) {
    const live = i.live && i.live.id ? ` · in view: ${i.live.name || i.live.id}` : "";
    s.textContent = `● face node — ${i.count} known${live}`; s.className = "mono ok";
  } else {
    s.textContent = "○ face node not running"; s.className = "mono muted";
  }
  $("#folks-hint").hidden = r.reachable;
  $("#folks-frame").hidden = !r.reachable;
}
async function loadFolksView() {
  let r; try { r = await api("/api/folks"); } catch (e) { return err(e); }
  const inp = $("#folks-url"); if (!inp.value) inp.value = r.url;
  const url = inp.value || r.url, fr = $("#folks-frame");
  $("#folks-open").href = url;
  if (r.reachable && fr.dataset.src !== url) { fr.src = url; fr.dataset.src = url; }
  folksStatus(r);
  clearInterval(folksTimer);
  folksTimer = setInterval(async () => {
    if (state.view !== "folks") return;
    try {
      const rr = await api("/api/folks");
      const was = $("#folks-frame").hidden;
      folksStatus(rr);
      if (rr.reachable && was) { fr.src = url; fr.dataset.src = url; }
    } catch (e) { /* keep the last status */ }
  }, 5000);
}
$("#folks-url").addEventListener("change", () => { const fr = $("#folks-frame"); fr.src = $("#folks-url").value; fr.dataset.src = fr.src; $("#folks-open").href = fr.src; });

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

// ---- Live view: collapsible activity log (what the robot's units are doing / waiting for) ----
// /api/activity merges the supervisor log tails of the nav, collision monitor, localization, route, voice, face,
// eval and inference units, noise removed. Fetched only while the pane is open and the Live tab is up.
var actlog = { open: false, timer: null, data: null, units: new Set(), allUnits: [], busy: false };
try { actlog.open = localStorage.getItem("actlog.open") === "1"; } catch (e) { /* storage blocked */ }
const alTime = (t) => { if (!t) return "        "; const d = new Date(t * 1000); return d.toTimeString().slice(0, 8); };
function actlogRender() {
  const body = $("#actlog-body"), d = actlog.data;
  if (!d) return;
  const sum = $("#actlog-summary");
  if (!d.reachable) { sum.textContent = "○ supervisor not reachable (robot off?)"; sum.className = "mono muted"; body.replaceChildren(); return; }
  const down = d.units.filter((u) => ["FATAL", "BACKOFF", "EXITED"].includes(u.state));
  const run = d.units.filter((u) => u.state === "RUNNING").length;
  sum.textContent = `● ${run} units running` + (down.length ? ` · down: ${down.map((u) => u.unit).join(", ")}` : "");
  sum.className = "mono " + (down.length ? "warn" : "ok");
  const chips = $("#actlog-units");
  if (chips.dataset.sig !== d.units.map((u) => u.unit + u.state).join()) {
    chips.dataset.sig = d.units.map((u) => u.unit + u.state).join();
    chips.replaceChildren(...d.units.map((u) => el("span", {
      class: "chip" + (actlog.units.has(u.unit) ? " on" : "") + (u.state === "RUNNING" ? "" : " down"),
      title: `${u.name}: ${u.state.toLowerCase()}`,
      onclick: () => { actlog.units.has(u.unit) ? actlog.units.delete(u.unit) : actlog.units.add(u.unit); chips.dataset.sig = ""; actlogRender(); },
    }, u.unit)));
  }
  const rank = { INFO: 0, WARN: 1, ERROR: 2, FATAL: 2 }, min = rank[$("#actlog-level").value] || 0;
  const waitOnly = $("#actlog-wait").checked;
  const rows = d.lines.filter((r) => actlog.units.has(r.unit) && (rank[r.level] || 0) >= min && (!waitOnly || r.wait || r.level !== "INFO"));
  const atEnd = body.scrollHeight - body.scrollTop - body.clientHeight < 30;
  body.replaceChildren(...rows.map((r) => el("div", { class: `al-row ${r.level}${r.wait ? " wait" : ""}` },
    el("span", { class: "al-t" }, alTime(r.t)), el("span", { class: "al-u" }, r.unit),
    el("span", { class: "al-m" }, (r.node && r.node !== r.unit ? `[${r.node}] ` : "") + r.msg),
    r.count > 1 ? el("span", { class: "al-x" }, `×${r.count}`) : null)));
  if ($("#actlog-follow").checked && (atEnd || body.dataset.fresh !== "1")) body.scrollTop = body.scrollHeight;
  body.dataset.fresh = "1";
}
async function actlogRefresh() {
  if (!actlog.open || state.view !== "map" || document.visibilityState !== "visible" || actlog.busy) return;
  actlog.busy = true;
  try {
    actlog.data = await api("/api/activity?lines=400");
    if (!actlog.allUnits.length && actlog.data.units) {      // first fetch: every unit on, except the chatty infra ones
      actlog.allUnits = actlog.data.units.map((u) => u.unit);
      actlog.units = new Set(actlog.allUnits.filter((u) => !["base", "tag"].includes(u)));
    }
    actlogRender();
  } catch (e) { $("#actlog-summary").textContent = "○ " + e.message; $("#actlog-summary").className = "mono muted"; }
  finally { actlog.busy = false; }
}
function actlogApply() {
  $("#actlog").classList.toggle("collapsed", !actlog.open);
  $("#actlog-toggle").textContent = actlog.open ? "▾" : "▸";
  $("#actlog-body").hidden = !actlog.open;
  $("#actlog-ctl").hidden = !actlog.open;
  clearInterval(actlog.timer);
  if (actlog.open) { actlog.timer = setInterval(actlogRefresh, 3000); actlogRefresh(); }
}
$("#actlog-toggle").addEventListener("click", () => {
  actlog.open = !actlog.open;
  try { localStorage.setItem("actlog.open", actlog.open ? "1" : "0"); } catch (e) { /* storage blocked */ }
  actlogApply();
});
["#actlog-level", "#actlog-wait", "#actlog-follow"].forEach((s) => $(s).addEventListener("change", actlogRender));
actlogApply();

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

// ---- Maps view: the robot's nav maps (rd_ws/src/robot_disco/maps, kept by the route editor) ----
// list · activate (localization + navigation restart on it) · new map / re-map with slam_toolbox ·
// a drawing editor over the map image · delete. Routes, landmarks and tasks are tagged with the map
// they were made on; the editor serves the active map and tags new ones with it.
var mapsTimer = null, mapsData = null;   // var: showView() may run for #maps before this block is reached (TDZ)
const fmtMapSize = (m) => m.metres ? `${m.metres[0]} × ${m.metres[1]} m` : "?";
const fmtMB = (b) => b >= 1e6 ? (b / 1e6).toFixed(1) + " MB" : Math.round(b / 1e3) + " kB";
// one badge per unit, all alike. Mapping needs localization (AMCL) and navigation OUT of the way, so for
// those two RUNNING is the no-go (red) and STOPPED is green; for SLAM it is the other way round.
function mapsUnitBadge(units, key, label, runningIsGood) {
  const st = units[key] || "—", running = st === "RUNNING";
  const cls = st === "—" ? "" : (running === runningIsGood ? " ok" : " warn");
  return el("span", { class: "tag" + cls, title: key }, `${label}: ${st.toLowerCase()}`);
}
function mapCard(m, d) {
  const busy = d.mapping || (d.job && d.job.running);
  const act = m.active;
  return el("div", { class: "card map-card" + (act ? " active" : "") + (m.error ? " warn" : "") },
    el("div", { class: "card-top" },
      el("span", { class: "card-name" }, m.name),
      act ? el("span", { class: "tag ok", title: "the map the robot runs: new routes and landmarks are tagged with it" }, "active") : null,
      m.posegraph ? el("span", { class: "tag", title: "made with SLAM: its posegraph is kept, so it can be continued (Edit via SLAM)" }, "SLAM") : null),
    m.error ? el("div", { class: "muted" }, m.error)
      : el("a", { href: `/api/maps/${m.name}.png?t=${Math.round(m.mtime || 0)}`, target: "_blank", title: "open the map image full size" },
          el("img", { class: "map-thumb", src: `/api/maps/${m.name}.png?t=${Math.round(m.mtime || 0)}`, alt: m.name })),
    el("div", { class: "card-meta" },
      el("span", {}, el("b", {}, fmtMapSize(m)), ` · ${m.width || "?"} × ${m.height || "?"} cells @ ${m.resolution} m`),
      el("span", {}, `origin ${m.origin ? m.origin.map((v) => v.toFixed(2)).join(", ") : "?"}`),
      el("span", {}, `${m.image || ""} · ${fmtMB(m.bytes || 0)}`),
      el("span", {}, m.mtime ? new Date(m.mtime * 1000).toLocaleString() : "")),
    el("div", { class: "card-foot" },
      el("button", { class: "btn primary", disabled: act || busy || !!m.error, title: act ? "already the robot's map" : "make this the robot's map (localization + navigation restart on it)", onclick: () => mapActivate(m) }, act ? "Active" : "Activate"),
      el("button", { class: "btn", disabled: busy || !!m.error, title: "draw on the map: walls, free space, unknown", onclick: () => mapEdit(m) }, "✎ Edit"),
      el("button", { class: "btn", disabled: busy || !m.posegraph, title: m.posegraph ? "continue mapping this map with slam_toolbox (drive around, then save)" : "no posegraph: only maps made with SLAM here can be continued", onclick: () => mapSlamStart("continue", m.name) }, "Edit via SLAM"),
      el("span", { class: "muted lm-when" }, ""),
      el("button", { class: "btn danger", disabled: act || busy, title: act ? "activate another map first" : "delete the map files", onclick: () => mapDelete(m) }, "Delete")));
}
function mapsRender(d) {
  mapsData = d;
  const units = d.units || {};
  const st = $("#maps-status");
  st.replaceChildren(el("span", {}, `● active: ${d.active || "?"} · ${(d.maps || []).length} maps`), " ",
    ...(d.source === "sim" ? [el("span", { class: "tag warn" }, "sim")] : [mapsUnitBadge(units, "robot_disco_slam", "SLAM", true), " ", mapsUnitBadge(units, "robot_disco_localization", "localization", false), " ", mapsUnitBadge(units, "robot_disco_navigation", "navigation", false)]));
  st.className = "mono ok";
  $("#btn-maps-new").disabled = !!d.mapping || (d.job && d.job.running);
  const slam = $("#maps-slam"), req = (d.slam && d.slam.request) || {};
  if (d.mapping) {
    slam.hidden = false;
    const name = req.name || "";
    slam.replaceChildren(
      el("span", {}, el("b", {}, req.mode === "continue" ? `Re-mapping ${name}` : `Mapping a new map: ${name}`), " — drive the robot around (gamepad or VLA Teleop); the Live tab shows the map grow. Save when it looks right, then Stop and Activate it."),
      el("span", { class: "muted" }, units.robot_disco_slam ? `slam unit: ${units.robot_disco_slam.toLowerCase()}` : ""),
      d.slam && d.slam.last_save ? el("span", { class: "tag ok" }, `saved ${d.slam.last_save.name} at ${new Date(d.slam.last_save.t * 1000).toLocaleTimeString()}`) : null,
      el("input", { id: "maps-savename", class: "robot-url", value: name, placeholder: "save as", maxlength: 40 }),
      el("button", { class: "btn primary", onclick: mapSlamSave }, "Save"),
      el("button", { class: "btn danger", onclick: mapSlamStop }, "Stop mapping"));
  } else { slam.hidden = true; slam.replaceChildren(); }
  const list = $("#maps-list");
  list.replaceChildren(...(d.maps || []).map((m) => mapCard(m, d)));
  if (d.job && (d.job.running || (d.job.log || []).length)) list.prepend(el("div", { class: "maps-job" + (d.job.running ? " on" : "") }, (d.job.running ? "switching map: " : "last switch: ") + (d.job.log || []).join(" · ")));
}
async function mapsRefresh() {
  let d; try { d = await api("/api/maps"); } catch (e) { $("#maps-status").textContent = "○ " + e.message; $("#maps-status").className = "mono muted"; $("#maps-list").replaceChildren(); $("#btn-maps-new").disabled = true; return; }
  mapsRender(d);
}
function loadMapsView() {
  mapsRefresh();
  clearInterval(mapsTimer);
  mapsTimer = setInterval(() => { if (state.view === "maps" && !document.querySelector(".mapedit-back")) mapsRefresh(); }, 4000);
}
$("#btn-maps-refresh").onclick = mapsRefresh;
$("#btn-maps-new").onclick = () => {
  const name = $("#maps-newname").value.trim();
  if (!name) return err("give the new map a name first");
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(name)) return err("map names: letters, digits, _ and - only");
  if (mapsData && (mapsData.maps || []).some((m) => m.name === name)) return err(`a map named ${name} exists`);
  mapSlamStart("new", name);
};
$("#maps-newname").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#btn-maps-new").click(); });
async function mapSlamStart(mode, name) {
  const what = mode === "continue" ? `Re-map ${name} with SLAM?` : `Start a new map named ${name}?`;
  if (!confirm(`${what}\n\nLocalization stops and slam_toolbox starts; the robot will not navigate until a map is activated again. Drive it around by hand, then Save and Stop.`)) return;
  try { const d = await post("/api/maps/slam/start", { mode, name }); ok(`SLAM started (${mode}): ${(d.units || []).join(", ") || "editor only"}`); $("#maps-newname").value = ""; }
  catch (e) { return err(e); }
  mapsRefresh();
}
async function mapSlamSave() {
  const name = ($("#maps-savename") || {}).value?.trim();
  if (!name) return err("give the map a name");
  toast("saving the map through slam_toolbox…");
  try { const d = await post("/api/maps/slam/save", { name }); ok(`saved ${name}: map ${d.saved && d.saved.save_map === 0 ? "ok" : d.saved && d.saved.save_map}, posegraph ${d.saved && d.saved.serialize === 0 ? "ok" : d.saved && d.saved.serialize}`); }
  catch (e) { return err(e); }
  mapsRefresh();
}
async function mapSlamStop() {
  if (!confirm("Stop mapping? Anything not saved is lost. The file map comes back; activate the saved map to run on it.")) return;
  try { await post("/api/maps/slam/stop", {}); ok("SLAM stopped"); } catch (e) { return err(e); }
  mapsRefresh();
}
async function mapActivate(m) {
  if (!confirm(`Activate ${m.name}?\n\nLocalization and navigation restart on it (the robot stops what it is doing); set the initial pose afterwards. New routes and landmarks will be tagged with it.`)) return;
  try { await post(`/api/maps/${m.name}/activate`, { restart: true }); ok(`${m.name} activated — restarting localization and navigation`); } catch (e) { return err(e); }
  mapsRefresh();
}
async function mapDelete(m) {
  if (!confirm(`Delete map ${m.name} and its files? Routes and landmarks made on it stay, tagged with its name.`)) return;
  try { const d = await api(`/api/maps/${m.name}`, { method: "DELETE" }); ok(`deleted ${(d.deleted || []).join(", ")}`); } catch (e) { return err(e); }
  mapsRefresh();
}

// ---- the drawing editor: paint on the map image (black = wall, white = free, grey = unknown) ----
function mapEdit(m) {
  const back = el("div", { class: "mapedit-back" });
  const view = el("div", { class: "mapedit-view" });
  const canvas = el("canvas"); view.append(canvas);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const tools = { wall: "#000000", free: "#ffffff", unknown: "#cdcdcd" };
  let tool = "wall", shape = "brush", size = 3, scale = 1, tx = 0, ty = 0, drawing = null, undo = [], redo = [];
  const stat = el("span", { class: "mono muted" }, "");
  const sizeIn = el("input", { type: "range", min: 1, max: 40, value: size, title: "brush size (cells)" });
  const saveAs = el("input", { type: "text", placeholder: "save as (new name)", maxlength: 40, title: "leave empty to overwrite " + m.name });
  const toolBtn = (key, label, title) => el("button", { class: "btn" + (tool === key ? " on" : ""), "data-tool": key, title, onclick: (e) => { tool = key; bar.querySelectorAll("[data-tool]").forEach((b) => b.classList.toggle("on", b.dataset.tool === key)); view.classList.toggle("pan", key === "pan"); } }, label);
  const shapeBtn = (key, label, title) => el("button", { class: "btn" + (shape === key ? " on" : ""), "data-shape": key, title, onclick: () => { shape = key; bar.querySelectorAll("[data-shape]").forEach((b) => b.classList.toggle("on", b.dataset.shape === key)); } }, label);
  const bar = el("div", { class: "mapedit-bar" },
    el("b", {}, `Edit map ${m.name}`), el("span", { class: "muted" }, `${m.width} × ${m.height} cells @ ${m.resolution} m`),
    toolBtn("wall", "■ wall", "paint occupied (black)"), toolBtn("free", "□ free", "paint free space (white)"), toolBtn("unknown", "▒ unknown", "paint unknown (grey)"), toolBtn("pan", "✥ pan", "drag to pan (wheel zooms anywhere)"),
    el("span", { class: "muted" }, "|"), shapeBtn("brush", "brush", "freehand"), shapeBtn("line", "line", "drag a straight line"), shapeBtn("rect", "rect", "drag a filled rectangle"),
    el("label", {}, "size", sizeIn),
    el("button", { class: "btn", title: "undo (Ctrl/⌘+Z)", onclick: () => doUndo() }, "↶ undo"), el("button", { class: "btn", title: "redo", onclick: () => doRedo() }, "↷ redo"),
    el("button", { class: "btn", title: "fit the map in the window", onclick: () => fit() }, "fit"),
    stat, el("span", { style: "flex:1" }),
    saveAs, el("button", { class: "btn primary", onclick: () => save() }, "Save"), el("button", { class: "btn", onclick: () => close() }, "Close"));
  back.append(bar, view); document.body.append(back);
  sizeIn.oninput = () => { size = parseInt(sizeIn.value, 10) || 1; };
  const img = new Image();
  img.onload = () => { canvas.width = img.naturalWidth; canvas.height = img.naturalHeight; ctx.drawImage(img, 0, 0); fit(); snapshot(); undo.length = 0; };
  img.onerror = () => err("could not load the map image");
  img.src = `/api/maps/${m.name}.png?t=${Date.now()}`;
  const apply = () => { canvas.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`; stat.textContent = `${Math.round(scale * 100)}%`; };
  function fit() { const r = view.getBoundingClientRect(); scale = Math.min(r.width / canvas.width, r.height / canvas.height) * 0.96; tx = (r.width - canvas.width * scale) / 2; ty = (r.height - canvas.height * scale) / 2; apply(); }
  function snapshot() { undo.push(ctx.getImageData(0, 0, canvas.width, canvas.height)); if (undo.length > 25) undo.shift(); redo.length = 0; }
  function doUndo() { if (undo.length < 2) return; redo.push(undo.pop()); ctx.putImageData(undo[undo.length - 1], 0, 0); }
  function doRedo() { if (!redo.length) return; const d = redo.pop(); undo.push(d); ctx.putImageData(d, 0, 0); }
  const toCell = (e) => { const r = view.getBoundingClientRect(); return [(e.clientX - r.left - tx) / scale, (e.clientY - r.top - ty) / scale]; };
  const dot = (x, y) => { ctx.fillStyle = tools[tool]; const h = size / 2; ctx.fillRect(Math.round(x - h), Math.round(y - h), size, size); };
  const line = (x0, y0, x1, y1) => { const n = Math.ceil(Math.hypot(x1 - x0, y1 - y0)) + 1; for (let i = 0; i <= n; i++) dot(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n); };
  view.addEventListener("wheel", (e) => { e.preventDefault(); const r = view.getBoundingClientRect(); const mx = e.clientX - r.left, my = e.clientY - r.top; const f = e.deltaY < 0 ? 1.15 : 1 / 1.15; const ns = Math.min(40, Math.max(0.05, scale * f)); tx = mx - (mx - tx) * ns / scale; ty = my - (my - ty) * ns / scale; scale = ns; apply(); }, { passive: false });
  view.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 && e.button !== 1) return;
    view.setPointerCapture(e.pointerId);
    if (tool === "pan" || e.button === 1 || e.shiftKey) { drawing = { pan: true, x: e.clientX, y: e.clientY, tx, ty }; return; }
    const [x, y] = toCell(e); drawing = { x0: x, y0: y, x, y, base: ctx.getImageData(0, 0, canvas.width, canvas.height) };
    if (shape === "brush") dot(x, y);
  });
  view.addEventListener("pointermove", (e) => {
    if (!drawing) return;
    if (drawing.pan) { tx = drawing.tx + (e.clientX - drawing.x); ty = drawing.ty + (e.clientY - drawing.y); apply(); return; }
    const [x, y] = toCell(e);
    if (shape === "brush") { line(drawing.x, drawing.y, x, y); drawing.x = x; drawing.y = y; return; }
    ctx.putImageData(drawing.base, 0, 0);
    if (shape === "line") line(drawing.x0, drawing.y0, x, y);
    else { ctx.fillStyle = tools[tool]; ctx.fillRect(Math.round(Math.min(drawing.x0, x)), Math.round(Math.min(drawing.y0, y)), Math.round(Math.abs(x - drawing.x0)) + 1, Math.round(Math.abs(y - drawing.y0)) + 1); }
  });
  const end = () => { if (drawing && !drawing.pan) snapshot(); drawing = null; };
  view.addEventListener("pointerup", end); view.addEventListener("pointercancel", end);
  const keys = (e) => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); e.shiftKey ? doRedo() : doUndo(); } if (e.key === "Escape") close(); };
  window.addEventListener("keydown", keys);
  function close() { if (undo.length > 1 && !confirm("Close without saving?")) return; window.removeEventListener("keydown", keys); back.remove(); }
  async function save() {
    const name = saveAs.value.trim();
    if (name && !/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(name)) return err("map names: letters, digits, _ and - only");
    if (!name && !confirm(`Overwrite ${m.name}'s image? (routes and landmarks keep their coordinates; nav uses it after the next activation / restart)`)) return;
    toast("saving…");
    try { const d = await post(`/api/maps/${m.name}/draw`, { png: canvas.toDataURL("image/png"), save_as: name || null }); ok(`saved ${d.map.name}`); undo.length = 1; }
    catch (e) { return err(e); }
    window.removeEventListener("keydown", keys); back.remove(); mapsRefresh();
  }
}

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
$("#btn-restart-base").onclick = async () => { if (!confirm("Restart robot_disco, then localization, navigation, the collision monitor and the route editor, in order (about 45 s)? The robot must be stationary.")) return; try { await post("/api/services/restart_base", {}); } catch (e) { err(e); } renderServices(); };


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
// ---- robot speaker volume (header, every tab): the face node on the robot serves /api/volume
// next to its Folks page; the manager relays. Polled slowly, never while the slider is being dragged.
const vol = { timer: null, dragging: false, last: null, muted: false, pending: null };
function volShow(v) {
  const r = $("#vol-range"), p = $("#vol-pct"), m = $("#vol-mute");
  if (!v || v.volume == null) {
    r.disabled = true; p.textContent = "—"; p.className = "mono muted"; m.textContent = "🔊"; m.classList.remove("off");
    m.title = v && v.error ? v.error : "robot speaker: face node not reachable";
    return;
  }
  r.disabled = false;
  if (!vol.dragging) r.value = v.volume;
  vol.muted = !!v.muted;
  p.textContent = vol.muted ? "mute" : `${v.volume}%`;
  p.className = vol.muted ? "mono warn" : "mono muted";
  m.textContent = vol.muted ? "🔇" : v.volume === 0 ? "🔈" : v.volume < 50 ? "🔉" : "🔊";
  m.classList.toggle("off", vol.muted);
  m.title = `${vol.muted ? "unmute" : "mute"} the robot speaker (${v.sink || "default sink"})`;
}
async function volRefresh() {
  if (vol.dragging || document.visibilityState !== "visible") return;
  try { volShow(await api("/api/volume")); } catch (e) { volShow({ error: e.message }); }
}
function volSend(payload) {
  // one request in flight at a time; the latest value wins (a drag fires dozens of input events)
  vol.pending = payload;
  if (vol.last) return;
  vol.last = (async () => {
    while (vol.pending) {
      const p = vol.pending; vol.pending = null;
      try { volShow(await post("/api/volume", p)); } catch (e) { err(e); }
    }
    vol.last = null;
  })();
}
$("#vol-range").addEventListener("pointerdown", () => { vol.dragging = true; });
$("#vol-range").addEventListener("pointerup", () => { vol.dragging = false; });
$("#vol-range").addEventListener("input", (e) => { $("#vol-pct").textContent = `${e.target.value}%`; volSend({ volume: Number(e.target.value) }); });
$("#vol-range").addEventListener("change", (e) => { vol.dragging = false; volSend({ volume: Number(e.target.value) }); });
$("#vol-mute").addEventListener("click", () => volSend({ muted: !vol.muted }));
volRefresh();
vol.timer = setInterval(volRefresh, 10000);

setInterval(() => loadDatasets(true), 4000);          // folder monitor (datasets view)
setInterval(() => { if (state.view === "models") loadModels(); }, 5000);  // live training refresh
