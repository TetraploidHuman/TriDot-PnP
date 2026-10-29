"use strict";

window.addEventListener("error", (evt) => {
  // Surface script errors in the UI (headless screenshots have no console).
  const msg = evt && evt.message ? evt.message : "unknown script error";
  document.title = `JS ERROR: ${msg}`;
  const t = document.getElementById("toast");
  if (t) {
    t.textContent = `JS 错误：${msg}`;
    t.hidden = false;
    t.classList.add("error");
  }
});
window.addEventListener("unhandledrejection", (evt) => {
  const reason = evt && evt.reason;
  const msg = reason && reason.message ? reason.message : String(reason);
  document.title = `JS REJECTION: ${msg}`;
  const t = document.getElementById("toast");
  if (t) {
    t.textContent = `JS 错误：${msg}`;
    t.hidden = false;
    t.classList.add("error");
  }
});

/* TriDot real-video annotation front-end.
 *
 * Talks to the recovered label_tool server (research/label_tool, loaded from
 * .pyc).  Endpoints used:
 *   GET  /api/health | /api/videos
 *   POST /api/session                     {video, stride}
 *   GET  /api/session/<sid>/summary
 *   GET  /api/session/<sid>/frame/<i>     -> frame JSON (pred/manual/status)
 *   GET  /api/session/<sid>/frame/<i>.jpg -> frame image
 *   POST /api/session/<sid>/action        {action, index, points?}
 *        actions: goto | accept | reject | set_points
 *   POST /api/session/<sid>/export
 *   POST /api/upload                      multipart field "upload.mp4"
 *
 * Server semantics worth knowing: every action addresses the frame by `index`
 * (session frame index, NOT the source video frame number); `set_points`
 * requires all three colours in one request.
 */

const COLORS = ["red", "green", "blue"];
const COLOR_HEX = { red: "#ff4d4f", green: "#35c46a", blue: "#4e7dff" };
const COLOR_LABEL = { red: "R", green: "G", blue: "B" };

const $ = (id) => document.getElementById(id);
const els = {
  videoSelect: $("videoSelect"),
  strideInput: $("strideInput"),
  newSessionBtn: $("newSessionBtn"),
  uploadBtn: $("uploadBtn"),
  uploadInput: $("uploadInput"),
  sessionInfo: $("sessionInfo"),
  counts: $("counts"),
  exportBtn: $("exportBtn"),
  frameList: $("frameList"),
  canvas: $("canvas"),
  placeholder: $("placeholder"),
  hint: $("hint"),
  prevBtn: $("prevBtn"),
  nextBtn: $("nextBtn"),
  frameLabel: $("frameLabel"),
  acceptBtn: $("acceptBtn"),
  rejectBtn: $("rejectBtn"),
  frameStatus: $("frameStatus"),
  pointInfo: $("pointInfo"),
  toast: $("toast"),
};

const ctx = els.canvas.getContext("2d");

const state = {
  videos: [],
  sid: null,
  nFrames: 0,
  width: 1280,
  height: 720,
  fps: 30,
  cur: 0,
  frames: new Map(), // index -> frame JSON
  images: new Map(), // index -> HTMLImageElement
  pts: {}, // current editing points {red:[u,v], ...}
  lockedColor: null,
  counts: { pending: 0, accepted: 0, rejected: 0 },
  loadingImage: -1,
  drag: null,
};

/* ---------------------------------------------------------------- helpers */

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let body = null;
  try {
    body = await res.json();
  } catch (err) {
    body = { error: `HTTP ${res.status}` };
  }
  if (!res.ok || (body && body.error)) {
    throw new Error((body && body.error) || `HTTP ${res.status}`);
  }
  return body;
}

function postJSON(path, payload) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

let toastTimer = null;
function toast(msg, isError = false) {
  els.toast.textContent = msg;
  els.toast.hidden = false;
  els.toast.classList.toggle("error", isError);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    els.toast.hidden = true;
  }, isError ? 6000 : 2600);
}

function setHint(msg) {
  els.hint.textContent = msg;
}

function fmtTime(sec) {
  return `${sec.toFixed(2)}s`;
}

/* ------------------------------------------------------------- bootstrap */

async function loadVideos() {
  const { videos } = await api("/api/videos");
  state.videos = videos || [];
  els.videoSelect.innerHTML = "";
  for (const name of state.videos) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    els.videoSelect.appendChild(opt);
  }
  const preferred = state.videos.find((n) => !n.startsWith("syn_"));
  if (preferred) els.videoSelect.value = preferred;
}

async function health() {
  try {
    const h = await api("/api/health");
    console.info("label_tool health", h);
  } catch (err) {
    toast(`无法连接标注服务：${err.message}`, true);
  }
}

/* ---------------------------------------------------------- session flow */

async function createSession() {
  const video = els.videoSelect.value;
  if (!video) {
    toast("请先选择视频", true);
    return;
  }
  const stride = Math.max(1, Number(els.strideInput.value) || 10);
  try {
    const info = await postJSON("/api/session", { video, stride });
    state.sid = info.id;
    state.nFrames = info.n_frames;
    state.width = info.width || state.width;
    state.height = info.height || state.height;
    state.frames.clear();
    state.images.clear();
    state.pts = {};
    state.lockedColor = null;
    state.cur = 0;
    state.counts = info.counts || { pending: info.n_frames, accepted: 0, rejected: 0 };

    els.canvas.width = state.width;
    els.canvas.height = state.height;
    els.placeholder.hidden = true;

    const summary = await api(`/api/session/${state.sid}/summary`);
    state.fps = summary.fps || 30;
    els.sessionInfo.textContent = `${summary.seq_id} · ${summary.n_frames} 帧 · 步长 ${summary.stride} · ${summary.width}×${summary.height}@${state.fps.toFixed(0)}fps`;
    buildFrameList();
    renderCounts();
    await gotoFrame(0);
    prefetchFrames();
    toast(`会话 ${info.id} 已创建（${info.n_frames} 帧）`);
  } catch (err) {
    toast(`创建会话失败：${err.message}`, true);
  }
}

function buildFrameList() {
  els.frameList.innerHTML = "";
  for (let i = 0; i < state.nFrames; i += 1) {
    const el = document.createElement("div");
    el.className = "frame-item pending";
    el.dataset.index = String(i);
    el.innerHTML = `<span class="dot"></span><span class="idx">#${i}</span><span class="status-txt"></span>`;
    el.addEventListener("click", () => gotoFrame(i));
    els.frameList.appendChild(el);
  }
}

function frameItem(i) {
  return els.frameList.querySelector(`.frame-item[data-index="${i}"]`);
}

function applyFrameState(index, frame) {
  state.frames.set(index, frame);
  const item = frameItem(index);
  if (!item) return;
  item.className = `frame-item ${frame.status}`;
  const txt = item.querySelector(".status-txt");
  if (txt) txt.textContent = frame.manual ? "手动" : frame.pred && frame.pred.method === "b0" ? "自动" : "";
  if (frame.manual) item.dataset.manual = "1";
  else delete item.dataset.manual;
}

function updateCounts() {
  const counts = { pending: 0, accepted: 0, rejected: 0 };
  for (const frame of state.frames.values()) {
    counts[frame.status] = (counts[frame.status] || 0) + 1;
  }
  state.counts = counts;
  renderCounts();
}

function renderCounts() {
  const c = state.counts;
  els.counts.innerHTML =
    `<span>接受 <b style="color:var(--ok)">${c.accepted || 0}</b></span>` +
    `<span>拒绝 <b style="color:var(--bad)">${c.rejected || 0}</b></span>` +
    `<span>待标 <b style="color:var(--warn)">${c.pending || 0}</b></span>`;
}

async function fetchFrame(i, { force = false } = {}) {
  if (!force && state.frames.has(i)) return state.frames.get(i);
  const frame = await api(`/api/session/${state.sid}/frame/${i}`);
  applyFrameState(i, frame);
  updateCounts();
  return frame;
}

async function loadImage(i) {
  if (state.images.has(i)) return state.images.get(i);
  const img = new Image();
  img.src = `/api/session/${state.sid}/frame/${i}.jpg`;
  await img.decode();
  state.images.set(i, img);
  return img;
}

async function gotoFrame(i) {
  if (!state.sid) return;
  if (i < 0 || i >= state.nFrames) return;
  state.cur = i;
  els.placeholder.hidden = true;
  setHint("加载中…");
  try {
    const [frame, img] = await Promise.all([fetchFrame(i), loadImage(i)]);
    state.pts = frame.manual ? { ...frame.manual } : frame.pred && frame.pred.ok ? { ...frame.pred.points } : {};
    drawScene(img);
    updateFramePanel(frame);
    for (const el of els.frameList.querySelectorAll(".frame-item.active")) el.classList.remove("active");
    const item = frameItem(i);
    if (item) {
      item.classList.add("active");
      item.scrollIntoView({ block: "nearest" });
    }
    setHint(frame.manual ? "该帧已手动标注（自动检测点已被覆盖）" : "接受自动检测，或点击/拖动修正三点后接受");
  } catch (err) {
    setHint("");
    toast(`加载帧失败：${err.message}`, true);
  }
}

function updateFramePanel(frame) {
  els.frameLabel.textContent = `#${frame.index} / ${state.nFrames - 1}`;
  const t = frame.frame_index != null ? `源帧 ${frame.frame_index} · ${fmtTime(frame.timestamp_sec || 0)}` : "";
  els.frameStatus.textContent = `${frame.status} · ${t}`;
  els.acceptBtn.disabled = false;
  els.rejectBtn.disabled = false;
  els.prevBtn.disabled = frame.index <= 0;
  els.nextBtn.disabled = frame.index >= state.nFrames - 1;
  renderPointInfo();
}

function renderPointInfo() {
  const rows = [];
  for (const c of COLORS) {
    const p = state.pts[c];
    rows.push(`<span>${COLOR_LABEL[c]}</span><span>${p ? `${p[0].toFixed(1)}, ${p[1].toFixed(1)}` : "—"}</span>`);
  }
  els.pointInfo.innerHTML = rows.join("");
  for (const btn of document.querySelectorAll(".color-row button")) {
    btn.classList.toggle("active", btn.dataset.color === state.lockedColor);
  }
}

/* ------------------------------------------------------------- rendering */

function drawScene(img) {
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  if (img) ctx.drawImage(img, 0, 0, els.canvas.width, els.canvas.height);

  const frame = state.frames.get(state.cur);
  const isManual = Boolean(frame && frame.manual);
  for (const c of COLORS) {
    const p = state.pts[c];
    if (!p) continue;
    drawPoint(p, c, isManual ? "manual" : "auto");
  }
}

function drawPoint(p, color, kind) {
  const [u, v] = p;
  const r = kind === "manual" ? 11 : 9;
  ctx.save();
  ctx.lineWidth = kind === "manual" ? 2.5 : 2;
  ctx.strokeStyle = COLOR_HEX[color];
  ctx.fillStyle = kind === "manual" ? "rgba(255,255,255,.12)" : "rgba(255,255,255,.05)";
  ctx.beginPath();
  ctx.arc(u, v, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = COLOR_HEX[color];
  ctx.font = "bold 13px ui-monospace, monospace";
  ctx.fillText(COLOR_LABEL[color], u + r + 3, v - r);
  ctx.restore();
}

/* ------------------------------------------------------------- interaction */

function canvasToImage(evt) {
  const rect = els.canvas.getBoundingClientRect();
  const x = ((evt.clientX - rect.left) / rect.width) * els.canvas.width;
  const y = ((evt.clientY - rect.top) / rect.height) * els.canvas.height;
  return [Math.max(0, Math.min(els.canvas.width, x)), Math.max(0, Math.min(els.canvas.height, y))];
}

function nearestPoint(u, v) {
  let best = null;
  let bestDist = 18;
  for (const c of COLORS) {
    const p = state.pts[c];
    if (!p) continue;
    const d = Math.hypot(p[0] - u, p[1] - v);
    if (d < bestDist) {
      bestDist = d;
      best = c;
    }
  }
  return best;
}

function nextColor() {
  if (state.lockedColor) return state.lockedColor;
  for (const c of COLORS) if (!state.pts[c]) return c;
  return null;
}

function canSavePoints() {
  return COLORS.every((c) => state.pts[c]);
}

async function savePoints() {
  if (!canSavePoints()) {
    setHint("三点未齐：请点齐 R/G/B 后再保存（服务端要求三点同时提交）");
    return false;
  }
  const frame = state.frames.get(state.cur);
  if (!frame) return false;
  if (frame.status === "accepted") {
    // keep accepted status; server settles manual points via set_points anyway
  }
  try {
    const updated = await postJSON(`/api/session/${state.sid}/action`, {
      action: "set_points",
      index: state.cur,
      points: {
        red: state.pts.red,
        green: state.pts.green,
        blue: state.pts.blue,
      },
    });
    applyFrameState(state.cur, updated);
    state.pts = { ...updated.pred.points };
    updateCounts();
    drawScene(state.images.get(state.cur));
    renderPointInfo();
    setHint("已保存该帧三点");
    return true;
  } catch (err) {
    toast(`保存失败：${err.message}`, true);
    return false;
  }
}

function bindCanvas() {
  els.canvas.addEventListener("pointerdown", (evt) => {
    if (!state.sid) return;
    const [u, v] = canvasToImage(evt);
    const hit = nearestPoint(u, v);
    const color = hit || nextColor();
    if (!color) {
      setHint("三点都已存在：拖动圆点微调，或先选中要覆盖的颜色");
      return;
    }
    state.pts[color] = [u, v];
    state.drag = { color, moved: false };
    els.canvas.setPointerCapture(evt.pointerId);
    drawScene(state.images.get(state.cur));
    renderPointInfo();
  });

  els.canvas.addEventListener("pointermove", (evt) => {
    if (!state.drag) return;
    const [u, v] = canvasToImage(evt);
    state.pts[state.drag.color] = [u, v];
    state.drag.moved = true;
    drawScene(state.images.get(state.cur));
    renderPointInfo();
  });

  els.canvas.addEventListener("pointerup", async () => {
    if (!state.drag) return;
    const moved = state.drag.moved;
    state.drag = null;
    if (moved) await savePoints();
    else await savePoints();
  });
}

/* ---------------------------------------------------------------- actions */

async function acceptFrame() {
  if (!state.sid) return;
  try {
    const frame = await postJSON(`/api/session/${state.sid}/action`, {
      action: "accept",
      index: state.cur,
    });
    applyFrameState(state.cur, frame);
    updateCounts();
    toast(`已接受 #${state.cur}`);
    if (state.cur < state.nFrames - 1) await gotoFrame(state.cur + 1);
  } catch (err) {
    toast(`接受失败：${err.message}`, true);
  }
}

async function rejectFrame() {
  if (!state.sid) return;
  try {
    const frame = await postJSON(`/api/session/${state.sid}/action`, {
      action: "reject",
      index: state.cur,
    });
    applyFrameState(state.cur, frame);
    updateCounts();
    toast(`已拒绝 #${state.cur}`);
    if (state.cur < state.nFrames - 1) await gotoFrame(state.cur + 1);
  } catch (err) {
    toast(`拒绝失败：${err.message}`, true);
  }
}

async function exportSession() {
  if (!state.sid) return;
  els.exportBtn.disabled = true;
  try {
    const res = await api(`/api/session/${state.sid}/export`, { method: "POST" });
    console.info("export", res);
    const detail = res.n_accepted != null ? `已接受 ${res.n_accepted} 帧` : "";
    toast(`导出成功：${detail} → ${res.annotation_sparse || ""}`);
    if (res.train_hint) console.info("训练提示:", res.train_hint);
  } catch (err) {
    toast(`导出失败：${err.message}`, true);
  } finally {
    els.exportBtn.disabled = false;
  }
}

async function uploadVideo(file) {
  const form = new FormData();
  form.append("upload.mp4", file);
  try {
    const res = await api("/api/upload", { method: "POST", body: form });
    toast(`上传成功：${res.video_name || res.video || file.name}`);
    await loadVideos();
    if (res.video_name || res.video) els.videoSelect.value = res.video_name || res.video;
  } catch (err) {
    toast(`上传失败：${err.message}`, true);
  }
}

/* ------------------------------------------------------------------ prefetch */

async function prefetchFrames() {
  const total = state.nFrames;
  let cursor = 0;
  const worker = async () => {
    while (cursor < total) {
      const i = cursor;
      cursor += 1;
      if (i === state.cur || state.frames.has(i)) continue;
      try {
        await fetchFrame(i);
      } catch (err) {
        // ignore individual prefetch failures; they will retry on demand
      }
    }
  };
  await Promise.all([worker(), worker(), worker(), worker()]);
}

/* ------------------------------------------------------------------- wiring */

function bindControls() {
  els.newSessionBtn.addEventListener("click", createSession);
  els.exportBtn.addEventListener("click", exportSession);
  els.prevBtn.addEventListener("click", () => gotoFrame(state.cur - 1));
  els.nextBtn.addEventListener("click", () => gotoFrame(state.cur + 1));
  els.acceptBtn.addEventListener("click", acceptFrame);
  els.rejectBtn.addEventListener("click", rejectFrame);
  els.uploadBtn.addEventListener("click", () => els.uploadInput.click());
  els.uploadInput.addEventListener("change", () => {
    const file = els.uploadInput.files && els.uploadInput.files[0];
    if (file) uploadVideo(file);
    els.uploadInput.value = "";
  });
  els.videoSelect.addEventListener("change", () => {
    if (state.sid) toast("已切换视频：请点击『新建会话』生效");
  });

  for (const btn of document.querySelectorAll(".color-row button")) {
    btn.addEventListener("click", () => {
      const c = btn.dataset.color;
      state.lockedColor = state.lockedColor === c ? null : c;
      renderPointInfo();
      setHint(state.lockedColor ? `已锁定 ${c} 颜色：点击画布放置该点` : "颜色锁定已解除");
    });
  }

  document.addEventListener("keydown", (evt) => {
    if (evt.target && ["INPUT", "SELECT", "TEXTAREA"].includes(evt.target.tagName)) return;
    if (evt.key === "ArrowLeft") {
      evt.preventDefault();
      gotoFrame(state.cur - 1);
    } else if (evt.key === "ArrowRight") {
      evt.preventDefault();
      gotoFrame(state.cur + 1);
    } else if (evt.key === "a" || evt.key === "A") {
      acceptFrame();
    } else if (evt.key === "r" || evt.key === "R") {
      rejectFrame();
    } else if (evt.key === "1") {
      state.lockedColor = "red";
      renderPointInfo();
    } else if (evt.key === "2") {
      state.lockedColor = "green";
      renderPointInfo();
    } else if (evt.key === "3") {
      state.lockedColor = "blue";
      renderPointInfo();
    } else if (evt.key === "Escape") {
      state.lockedColor = null;
      renderPointInfo();
    }
  });

  bindCanvas();
}

async function main() {
  bindControls();
  await health();
  try {
    await loadVideos();
  } catch (err) {
    toast(`加载视频列表失败：${err.message}`, true);
  }
  // Deep link: /?video=<name>&stride=10 auto-creates the session (handy for
  // jumping straight into a specific capture, and for smoke tests).
  const qs = new URLSearchParams(window.location.search);
  if (qs.get("video")) {
    if (state.videos.includes(qs.get("video"))) els.videoSelect.value = qs.get("video");
    if (qs.get("stride")) els.strideInput.value = qs.get("stride");
    await createSession();
  }
}

main();
