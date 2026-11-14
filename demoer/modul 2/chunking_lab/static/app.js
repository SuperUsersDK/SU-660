"use strict";

/* Chunking-laboratoriet: klip → embed → gem → søg, animeret. */

const $ = (id) => document.getElementById(id);
const els = {
  textTabs: $("textTabs"), strategyTabs: $("strategyTabs"), strategyDesc: $("strategyDesc"), params: $("params"),
  runBtn: $("runBtn"), skipBtn: $("skipBtn"), speed: $("speed"),
  doc: $("doc"), docInput: $("docInput"), docInfo: $("docInfo"), stats: $("stats"),
  modelBox: $("modelBox"), modelName: $("modelName"), vectorBars: $("vectorBars"), vectorNums: $("vectorNums"),
  dimsText: $("dimsText"), modelBadge: $("modelBadge"),
  db: $("db"), dbSvg: $("dbSvg"), dbDots: $("dbDots"), dbLines: $("dbLines"), dbQuery: $("dbQuery"), dbEmpty: $("dbEmpty"), dbInfo: $("dbInfo"),
  chips: $("questionChips"), searchForm: $("searchForm"), question: $("question"), searchBtn: $("searchBtn"),
  compareBtn: $("compareBtn"), searchStatus: $("searchStatus"), results: $("results"), compare: $("compare"),
  tooltip: $("tooltip"),
};

const SVG_NS = "http://www.w3.org/2000/svg";
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const state = {
  setup: null,
  textId: null,          // null = egen tekst
  strategy: "fixed",
  params: {},            // strategi-id → { navn: værdi }
  preview: null,         // chunks uden embeddings (til live-visning)
  run: null,             // seneste indekserede kørsel
  expected: null,        // forventet svar for et eksempelspørgsmål
  busy: false,
  skip: false,
  chunkSpans: new Map(), // chunk-indeks → [span, ...]
};

/* ------------------------------------------------------------------ hjælpere */

const color = (i, light = false) => `hsl(${(i * 137.508) % 360} ${light ? "85% 88%" : "68% 48%"})`;
const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const speed = () => (reducedMotion ? Infinity : Number(els.speed.value) * (state.run && state.run.chunks.length > 25 ? 2 : 1));
const dur = (ms) => (state.skip || reducedMotion ? 0 : ms / speed());
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, dur(ms)));

async function api(path, body) {
  const res = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const error = new Error(data.error || data.detail || res.statusText);
    error.notApplicable = Boolean(data.not_applicable);
    throw error;
  }
  return data;
}

function currentText() {
  return state.textId ? state.setup.texts.find((t) => t.id === state.textId).text : els.docInput.value;
}

function textRequest() {
  return state.textId ? { text_id: state.textId } : { text: els.docInput.value };
}

function strategyInfo(id = state.strategy) {
  return state.setup.strategies.find((s) => s.id === id);
}

/* ------------------------------------------------------------------ opsætning */

async function init() {
  state.setup = await api("/api/setup");
  const { model } = state.setup;
  els.modelBadge.textContent = `${model.name} · ${model.dims} dimensioner`;
  els.modelName.textContent = model.name;
  els.dimsText.textContent = model.dims;
  els.dbInfo.textContent = `Qdrant · ${state.setup.collection}`;
  for (const s of state.setup.strategies) {
    state.params[s.id] = Object.fromEntries(s.params.map((p) => [p.name, p.default]));
  }

  const tabs = [...state.setup.texts.map((t) => ({ id: t.id, title: t.title })), { id: null, title: "✏️ Egen tekst" }];
  for (const tab of tabs) {
    const b = document.createElement("button");
    b.className = "tab"; b.textContent = tab.title; b.dataset.id = tab.id ?? "";
    b.onclick = () => selectText(tab.id);
    els.textTabs.append(b);
  }
  for (const s of state.setup.strategies) {
    const b = document.createElement("button");
    b.className = "tab"; b.textContent = s.name; b.dataset.id = s.id;
    b.onclick = () => selectStrategy(s.id);
    els.strategyTabs.append(b);
  }

  renderBars(els.vectorBars, null);
  els.runBtn.onclick = runIndex;
  els.skipBtn.onclick = () => { state.skip = true; };
  els.searchForm.onsubmit = (e) => { e.preventDefault(); runSearch(); };
  els.compareBtn.onclick = runCompare;
  els.question.oninput = () => { state.expected = null; };
  let typing;
  els.docInput.oninput = () => { clearTimeout(typing); typing = setTimeout(() => { invalidate(); refreshPreview(); }, 250); };
  window.addEventListener("resize", () => drawDots(state.run));

  selectText(state.setup.texts[0].id);
}

function selectText(id) {
  if (state.busy) return;
  state.textId = id;
  for (const b of els.textTabs.children) b.classList.toggle("active", (b.dataset.id || null) === id);
  const custom = id === null;
  els.docInput.hidden = !custom;
  els.doc.parentElement.classList.toggle("custom", custom);
  // Vælg en strategi, der passer til teksttypen.
  const kind = custom ? "text" : state.setup.texts.find((t) => t.id === id).kind;
  if (kind === "markdown" && state.strategy === "code") state.strategy = "markdown";
  if (kind === "python" && state.strategy === "markdown") state.strategy = "code";
  if (kind === "text" && ["markdown", "code"].includes(state.strategy)) state.strategy = "fixed";
  renderChips();
  els.question.value = ""; state.expected = null;
  invalidate();
  selectStrategy(state.strategy);
}

function selectStrategy(id) {
  if (state.busy) return;
  state.strategy = id;
  for (const b of els.strategyTabs.children) b.classList.toggle("active", b.dataset.id === id);
  const s = strategyInfo();
  els.strategyDesc.textContent = s.description;
  els.params.innerHTML = "";
  for (const p of s.params) {
    const label = document.createElement("label");
    label.className = "param";
    label.innerHTML = `${escapeHtml(p.label)} <input type="range" min="${p.minimum}" max="${p.maximum}" step="${p.step}" value="${state.params[id][p.name]}"> <output>${state.params[id][p.name]}</output>`;
    const input = label.querySelector("input"); const out = label.querySelector("output");
    input.oninput = () => { state.params[id][p.name] = Number(input.value); out.textContent = input.value; invalidate(); refreshPreview(); };
    els.params.append(label);
  }
  if (!s.params.length) els.params.innerHTML = `<span class="hint">Ingen parametre: grænserne følger tekstens struktur.</span>`;
  invalidate();
  refreshPreview();
}

function renderChips() {
  els.chips.innerHTML = "";
  if (!state.textId) return;
  for (const q of state.setup.texts.find((t) => t.id === state.textId).questions) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "chip"; b.textContent = q.question;
    b.onclick = () => { els.question.value = q.question; state.expected = q.expected; if (state.run) runSearch(); };
    els.chips.append(b);
  }
}

/* ------------------------------------------------------------------ dokument */

function invalidate() {
  state.run = null;
  els.searchBtn.disabled = true;
  els.compareBtn.disabled = !currentText().trim();
  els.results.innerHTML = ""; els.compare.innerHTML = ""; els.searchStatus.textContent = "";
  clearDb();
}

let previewToken = 0;
async function refreshPreview() {
  const token = ++previewToken;
  const text = currentText();
  if (!text.trim()) { renderDoc([], 0); renderStats(null); return; }
  try {
    const data = await api("/api/chunk", { ...textRequest(), strategy: state.strategy, params: state.params[state.strategy] });
    if (token !== previewToken || state.busy) return;
    state.preview = data.chunks;
    renderDoc(data.chunks, data.chunks.length);
    renderStats(data.stats, data.chunks);
    els.runBtn.disabled = false;
  } catch (err) {
    if (token !== previewToken) return;
    state.preview = null;
    renderDoc([], 0);
    renderStats(null, null, err.notApplicable ? `⚠️ ${err.message}` : `Fejl: ${err.message}`);
    els.runBtn.disabled = true;
  }
}

/* Teksten vises som segmenter mellem alle chunk-grænser. Et segment, der ligger i to
   chunks (overlap), får begge farver som striber. */
function renderDoc(chunks, upto, cutting = -1) {
  const text = currentText();
  const visible = chunks.slice(0, upto);
  const cuts = new Set([0, text.length]);
  for (const c of chunks.slice(0, Math.max(upto, cutting + 1))) { cuts.add(c.start); cuts.add(c.end); }
  const points = [...cuts].sort((a, b) => a - b);

  els.doc.textContent = "";
  state.chunkSpans = new Map();
  const frag = document.createDocumentFragment();
  for (let k = 0; k < points.length - 1; k++) {
    const a = points[k], b = points[k + 1];
    const covering = visible.filter((c) => c.start <= a && c.end >= b).map((c) => c.index);
    const isCutting = cutting >= 0 && chunks[cutting].start <= a && chunks[cutting].end >= b;
    if (!covering.length && !isCutting) { frag.append(text.slice(a, b)); continue; }
    const span = document.createElement("span");
    span.className = "seg" + (isCutting ? " cutting" : "");
    span.textContent = text.slice(a, b);
    if (covering.length === 1) span.style.background = color(covering[0], true);
    else if (covering.length > 1) {
      const stripes = covering.map((i, n) => `${color(i, true)} ${n * 6}px ${(n + 1) * 6}px`).join(", ");
      span.style.background = `repeating-linear-gradient(135deg, ${stripes})`;
      span.title = "Overlap: teksten ligger i flere chunks";
    }
    const owners = isCutting ? [...covering, cutting] : covering;
    span.dataset.chunks = owners.join(",");
    for (const i of owners) {
      if (!state.chunkSpans.has(i)) state.chunkSpans.set(i, []);
      state.chunkSpans.get(i).push(span);
    }
    frag.append(span);
  }
  els.doc.append(frag);
}

function renderStats(stats, chunks, message) {
  if (message) { els.stats.innerHTML = `<span class="warn">${escapeHtml(message)}</span>`; els.docInfo.textContent = ""; return; }
  if (!stats) { els.stats.innerHTML = ""; els.docInfo.textContent = ""; return; }
  els.docInfo.textContent = `${currentText().length.toLocaleString("da-DK")} tegn`;
  const max = Math.max(...chunks.map((c) => c.length), 1);
  const bars = chunks.map((c) => `<div style="height:${Math.max(6, (c.length / max) * 100)}%;background:${color(c.index)}" title="Chunk ${c.index + 1}: ${c.length} tegn"></div>`).join("");
  const truncated = stats.truncated ? ` · <span class="warn">${stats.truncated} chunks er længere end modellens ${state.setup.model.max_tokens} tokens og klippes af</span>` : "";
  els.stats.innerHTML =
    `<span><b>${stats.chunks}</b> chunks</span><span>gns. <b>${stats.avg}</b> tegn</span>` +
    `<span>min <b>${stats.min}</b> / maks <b>${stats.max}</b></span>${truncated}` +
    `<div class="lengths" title="Længden af hver chunk">${bars}</div>`;
}

function scrollDocTo(index) {
  const spans = state.chunkSpans.get(index);
  if (!spans || !spans.length) return;
  const top = spans[0].offsetTop;
  if (top < els.doc.scrollTop + 20 || top > els.doc.scrollTop + els.doc.clientHeight - 60) {
    els.doc.scrollTop = Math.max(0, top - els.doc.clientHeight / 3);
  }
}

/* ------------------------------------------------------------------ vektor-database */

function clearDb() {
  els.dbDots.innerHTML = ""; els.dbLines.innerHTML = ""; els.dbQuery.innerHTML = "";
  els.dbEmpty.hidden = false;
}

const toSvg = (pos) => [6 + pos[0] * 88, 94 - pos[1] * 88];

function svgToClient(x, y) {
  const pt = els.dbSvg.createSVGPoint(); pt.x = x; pt.y = y;
  const p = pt.matrixTransform(els.dbSvg.getScreenCTM());
  return { x: p.x, y: p.y };
}

function addDot(chunk) {
  const [x, y] = toSvg(chunk.pos);
  const g = document.createElementNS(SVG_NS, "g");
  g.dataset.index = chunk.index;
  const c = document.createElementNS(SVG_NS, "circle");
  c.setAttribute("class", "dot"); c.setAttribute("cx", x); c.setAttribute("cy", y); c.setAttribute("r", 2.6);
  c.setAttribute("fill", color(chunk.index));
  const t = document.createElementNS(SVG_NS, "text");
  t.setAttribute("class", "dot-label"); t.setAttribute("x", x); t.setAttribute("y", y); t.textContent = chunk.index + 1;
  g.append(c, t);
  g.onmouseenter = (e) => showTooltip(e, chunk);
  g.onmousemove = (e) => moveTooltip(e);
  g.onmouseleave = hideTooltip;
  g.onclick = () => focusChunk(chunk.index);
  els.dbDots.append(g);
  els.dbEmpty.hidden = true;
  c.animate([{ r: 0 }, { r: 4.2 }, { r: 2.6 }], { duration: dur(450), easing: "ease-out" });
  return g;
}

function drawDots(run) {
  els.dbDots.innerHTML = "";
  if (!run) return;
  for (const chunk of run.chunks) addDot(chunk);
}

function showTooltip(event, chunk) {
  const text = currentText().slice(chunk.start, chunk.end);
  els.tooltip.textContent = `Chunk ${chunk.index + 1} · ${chunk.length} tegn\n\n${text.slice(0, 260)}${text.length > 260 ? " …" : ""}`;
  els.tooltip.hidden = false; moveTooltip(event);
}
function moveTooltip(event) {
  els.tooltip.style.left = Math.min(event.clientX + 14, innerWidth - 340) + "px";
  els.tooltip.style.top = Math.min(event.clientY + 14, innerHeight - 160) + "px";
}
function hideTooltip() { els.tooltip.hidden = true; }

function focusChunk(index) {
  for (const [i, spans] of state.chunkSpans) for (const s of spans) s.classList.toggle("hit1", i === index);
  scrollDocTo(index);
}

/* ------------------------------------------------------------------ animation */

function renderBars(container, values) {
  container.innerHTML = "";
  const list = values || Array.from({ length: 32 }, () => 0);
  for (const v of list) {
    const d = document.createElement("div");
    d.style.height = `${Math.max(3, Math.abs(v) * 100)}%`;
    d.style.background = v >= 0 ? "#2a9d8f" : "#e76f51";
    d.style.opacity = values ? 1 : 0.25;
    container.append(d);
  }
}

function showVector(values) {
  renderBars(els.vectorBars, values);
  const nums = values.slice(0, 4).map((v) => (v >= 0 ? " " : "") + v.toFixed(2)).join(", ");
  els.vectorNums.textContent = `[${nums}, …]\n${state.setup.model.dims} tal`;
}

/* Flyt et element fra ét rektangel til et punkt i en bue. */
function fly(el, from, to, ms, { scaleTo = 1, lift = 80 } = {}) {
  document.body.append(el);
  const w = el.offsetWidth, h = el.offsetHeight;
  const sx = from.x - w / 2, sy = from.y - h / 2, ex = to.x - w / 2, ey = to.y - h / 2;
  const mx = (sx + ex) / 2, my = Math.min(sy, ey) - lift;
  const anim = el.animate([
    { transform: `translate(${sx}px, ${sy}px) scale(1)`, opacity: 0.2 },
    { transform: `translate(${sx}px, ${sy - 10}px) scale(1.04)`, opacity: 1, offset: 0.12 },
    { transform: `translate(${mx}px, ${my}px) scale(${(1 + scaleTo) / 2})`, opacity: 1, offset: 0.55 },
    { transform: `translate(${ex}px, ${ey}px) scale(${scaleTo})`, opacity: 0.9 },
  ], { duration: dur(ms), easing: "cubic-bezier(.45,.05,.35,1)", fill: "forwards" });
  return anim.finished.then(() => el.remove());
}

const center = (rect) => ({ x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 });

function vectorPill(values) {
  const pill = document.createElement("div");
  pill.className = "fly fly-vec";
  renderBars(pill, values.slice(0, 24));
  return pill;
}

function setBusy(busy) {
  state.busy = busy;
  els.runBtn.disabled = busy;
  els.skipBtn.disabled = !busy;
  els.searchBtn.disabled = busy || !state.run;
  els.compareBtn.disabled = busy;
  for (const b of [...els.textTabs.children, ...els.strategyTabs.children]) b.disabled = busy;
  for (const i of els.params.querySelectorAll("input")) i.disabled = busy;
  if (!busy) state.skip = false;
}

/* ------------------------------------------------------------------ chunk og indeksér */

async function runIndex() {
  if (state.busy || !currentText().trim()) return;
  setBusy(true);
  invalidate();
  els.modelBox.classList.add("working");
  els.vectorNums.textContent = "embedder …";
  let run;
  try {
    run = await api("/api/index", { ...textRequest(), strategy: state.strategy, params: state.params[state.strategy] });
  } catch (err) {
    els.modelBox.classList.remove("working");
    els.vectorNums.textContent = "tekst → tal";
    renderStats(null, null, err.notApplicable ? `⚠️ ${err.message}` : `Fejl: ${err.message}`);
    setBusy(false);
    return;
  }
  els.modelBox.classList.remove("working");
  state.run = run;
  els.dbInfo.textContent = `Qdrant · ${run.collection} · ${run.cached ? "genbrugt" : `embeddet på ${run.embed_ms} ms`}`;

  renderDoc(run.chunks, 0);
  const scissors = document.createElement("div");
  scissors.className = "scissors"; scissors.textContent = "✂️";
  els.doc.append(scissors);

  for (const chunk of run.chunks) {
    if (state.skip) break;
    const i = chunk.index;
    // 1. Klip: chunken markeres, og saksen flytter sig til slutningen af den.
    renderDoc(run.chunks, i, i);
    els.doc.append(scissors);
    scrollDocTo(i);
    const cutSpans = state.chunkSpans.get(i) || [];
    if (cutSpans.length) {
      const rects = cutSpans[cutSpans.length - 1].getClientRects();
      const end = rects[rects.length - 1];
      const box = els.doc.getBoundingClientRect();
      scissors.style.left = `${end.right - box.left + els.doc.scrollLeft}px`;
      scissors.style.top = `${end.bottom - box.top + els.doc.scrollTop}px`;
    }
    await sleep(350);
    // 2. Chunken flyver til modellen.
    const card = document.createElement("div");
    card.className = "fly fly-card";
    card.style.setProperty("--c", color(i));
    card.textContent = currentText().slice(chunk.start, chunk.end).slice(0, 180);
    const spans = state.chunkSpans.get(i) || [];
    const fromRect = spans.length ? spans[0].getClientRects()[0] : els.doc.getBoundingClientRect();
    const modelRect = els.modelBox.getBoundingClientRect();
    els.modelBox.classList.add("working");
    await fly(card, { x: Math.min(fromRect.left + 115, innerWidth - 120), y: fromRect.top + 30 }, center(modelRect), 750, { scaleTo: 0.45 });
    // 3. Modellen laver vektoren.
    showVector(chunk.preview);
    await sleep(380);
    // 4. Vektoren flyver til databasen.
    const [x, y] = toSvg(chunk.pos);
    const target = svgToClient(x, y);
    const pill = vectorPill(chunk.preview);
    els.modelBox.classList.remove("working");
    renderDoc(run.chunks, i + 1);
    await fly(pill, center(modelRect), target, 650, { scaleTo: 0.5, lift: 50 });
    addDot(chunk);
  }
  scissors.remove();
  els.modelBox.classList.remove("working");
  renderDoc(run.chunks, run.chunks.length);
  drawDots(run);
  renderStats(run.stats, run.chunks);
  setBusy(false);
  els.searchBtn.disabled = false;
  els.searchStatus.textContent = `${run.chunks.length} chunks ligger i databasen. Stil et spørgsmål, eller vælg et eksempel ovenfor.`;
  if (els.question.value.trim()) runSearch();
}

/* ------------------------------------------------------------------ søg */

function markExpected(text, expected) {
  const safe = escapeHtml(text);
  if (!expected || !text.includes(expected)) return safe;
  return safe.replace(escapeHtml(expected), `<mark>${escapeHtml(expected)}</mark>`);
}

async function runSearch() {
  const question = els.question.value.trim();
  if (!question || !state.run || state.busy) return;
  setBusy(true);
  els.results.innerHTML = ""; els.dbLines.innerHTML = ""; els.dbQuery.innerHTML = "";
  for (const g of els.dbDots.children) { g.querySelector("circle").classList.remove("dim"); g.querySelector("circle").setAttribute("r", 2.6); }
  for (const spans of state.chunkSpans.values()) for (const s of spans) s.classList.remove("dim", "hit", "hit1");

  let result;
  try {
    result = await api("/api/search", { run_id: state.run.run_id, question, limit: 3 });
  } catch (err) {
    els.searchStatus.textContent = err.message; setBusy(false); return;
  }

  // Spørgsmålet flyver til modellen, bliver til en vektor og flyver til databasen.
  const card = document.createElement("div");
  card.className = "fly fly-card fly-query"; card.textContent = `❓ ${question}`;
  const modelRect = els.modelBox.getBoundingClientRect();
  els.modelBox.classList.add("working");
  await fly(card, center(els.question.getBoundingClientRect()), center(modelRect), 750, { scaleTo: 0.5, lift: 120 });
  showVector(result.preview);
  await sleep(380);
  els.modelBox.classList.remove("working");
  const [qx, qy] = toSvg(result.pos.map((v) => Math.min(1.05, Math.max(-0.05, v))));
  await fly(vectorPill(result.preview), center(modelRect), svgToClient(qx, qy), 650, { scaleTo: 0.5, lift: 50 });

  const star = document.createElementNS(SVG_NS, "path");
  star.setAttribute("class", "qstar");
  star.setAttribute("d", starPath(qx, qy, 4.2, 1.8));
  star.style.transformBox = "fill-box";
  star.style.transformOrigin = "center";
  els.dbQuery.append(star);
  star.animate([{ opacity: 0, transform: "scale(.3)" }, { opacity: 1, transform: "scale(1)" }], { duration: dur(300) });

  // Linjer til de nærmeste chunks; de øvrige tones ned.
  const hitIdx = new Set(result.hits.map((h) => h.index));
  for (const g of els.dbDots.children) g.querySelector("circle").classList.toggle("dim", !hitIdx.has(Number(g.dataset.index)));
  for (const [rank, hit] of result.hits.entries()) {
    const chunk = state.run.chunks[hit.index];
    const [x, y] = toSvg(chunk.pos);
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("class", "qline");
    line.setAttribute("x1", qx); line.setAttribute("y1", qy); line.setAttribute("x2", x); line.setAttribute("y2", y);
    els.dbLines.append(line);
    const length = Math.hypot(x - qx, y - qy);
    line.animate([{ strokeDashoffset: length, opacity: 0 }, { strokeDashoffset: 0, opacity: 1 }], { duration: dur(400) });
    const dot = els.dbDots.querySelector(`g[data-index="${hit.index}"] circle`);
    if (dot) dot.setAttribute("r", rank === 0 ? 4.4 : 3.6);
    await sleep(160);
  }

  // Fremhæv hits i dokumentet.
  for (const [i, spans] of state.chunkSpans) {
    const rank = result.hits.findIndex((h) => h.index === i);
    for (const s of spans) {
      s.classList.toggle("dim", rank < 0);
      s.classList.toggle("hit", rank > 0);
      s.classList.toggle("hit1", rank === 0);
    }
  }
  if (result.hits.length) scrollDocTo(result.hits[0].index);

  renderResults(result.hits);
  setBusy(false);
  els.searchBtn.disabled = false;
}

function starPath(cx, cy, outer, inner) {
  let d = "";
  for (let k = 0; k < 10; k++) {
    const r = k % 2 ? inner : outer;
    const a = (Math.PI / 5) * k - Math.PI / 2;
    d += `${k ? "L" : "M"}${cx + r * Math.cos(a)},${cy + r * Math.sin(a)}`;
  }
  return d + "Z";
}

function renderResults(hits) {
  const text = currentText();
  const expected = state.expected;
  let verdict = "";
  if (expected) {
    const rank = hits.findIndex((h) => text.slice(state.run.chunks[h.index].start, state.run.chunks[h.index].end).includes(expected));
    const intact = state.run.chunks.some((c) => text.slice(c.start, c.end).includes(expected));
    if (rank === 0) verdict = `<span class="verdict good">✅ Svaret ligger i den øverste chunk.</span>`;
    else if (rank > 0) verdict = `<span class="verdict mid">🟡 Svaret er nr. ${rank + 1}; en anden chunk scorer højere.</span>`;
    else if (intact) verdict = `<span class="verdict bad">❌ Svaret findes i en chunk, men ikke i top 3.</span>`;
    else verdict = `<span class="verdict bad">✂️ Svaret er klippet over: det ligger ikke helt i nogen chunk.</span>`;
  }
  els.searchStatus.innerHTML = verdict || "De nærmeste chunks (cosine-lighed):";
  els.results.innerHTML = "";
  for (const [rank, hit] of hits.entries()) {
    const chunk = state.run.chunks[hit.index];
    const div = document.createElement("div");
    div.className = "result";
    div.style.animationDelay = `${rank * 80}ms`;
    div.innerHTML =
      `<div class="rank">#${rank + 1}</div>` +
      `<div><b>${hit.score.toFixed(3)}</b><div class="score-bar"><div style="width:${Math.max(4, (hit.score - 0.7) / 0.3 * 100)}%;background:${color(hit.index)}"></div></div>` +
      `<div class="hint">chunk ${hit.index + 1} · ${chunk.length} tegn</div></div>` +
      `<div class="result-text">${markExpected(text.slice(chunk.start, chunk.end), expected)}</div>`;
    div.onclick = () => focusChunk(hit.index);
    els.results.append(div);
  }
}

/* ------------------------------------------------------------------ sammenlign */

async function runCompare() {
  const question = els.question.value.trim();
  if (!question) { els.searchStatus.textContent = "Skriv et spørgsmål, eller vælg et eksempel, først."; return; }
  setBusy(true);
  els.compare.innerHTML = `<p class="hint">Chunker, embedder og søger med alle strategier …</p>`;
  try {
    const data = await api("/api/compare", { ...textRequest(), question, expected: state.expected, params: state.params });
    renderCompare(data);
  } catch (err) {
    els.compare.innerHTML = `<p class="verdict bad">${escapeHtml(err.message)}</p>`;
  }
  setBusy(false);
  els.searchBtn.disabled = !state.run;
}

function renderCompare(data) {
  const rows = data.rows;
  const usable = rows.filter((r) => r.hits);
  const best = usable.filter((r) => r.answer_rank === 1).length;
  const head = data.expected
    ? `<p><b>Svaret ligger øverst for ${best} af ${usable.length} strategier.</b> <span class="hint">Forventet svar: “${escapeHtml(data.expected)}”</span></p>`
    : `<p class="hint">Eget spørgsmål: der er intet facit, så sammenlign selv de øverste chunks.</p>`;
  const body = rows.map((r) => {
    if (r.not_applicable || r.error) {
      return `<tr class="na"><td>${escapeHtml(r.name)}</td><td colspan="5">${escapeHtml(r.not_applicable || r.error)}</td></tr>`;
    }
    let verdict = "–";
    if (data.expected) {
      if (r.answer_rank === 1) verdict = `<span class="verdict good">✅ #1</span>`;
      else if (r.answer_rank) verdict = `<span class="verdict mid">🟡 #${r.answer_rank}</span>`;
      else if (r.answer_intact) verdict = `<span class="verdict bad">❌ ikke i top 3</span>`;
      else verdict = `<span class="verdict bad">✂️ klippet over</span>`;
    }
    const top = r.hits[0];
    return `<tr class="${r.strategy === state.strategy ? "current" : ""}">` +
      `<td><b>${escapeHtml(r.name)}</b><br><button class="ghost" data-show="${r.strategy}">Vis animation</button></td>` +
      `<td>${r.chunks}</td><td>${r.avg}</td><td>${top.score.toFixed(3)}</td><td>${verdict}</td>` +
      `<td class="snippet">${markExpected(top.text.slice(0, 400), data.expected)}${top.text.length > 400 ? " …" : ""}</td></tr>`;
  }).join("");
  els.compare.innerHTML = head +
    `<table><thead><tr><th>Strategi</th><th>Chunks</th><th>Gns. tegn</th><th>Top-1 score</th><th>Svar</th><th>Øverste chunk</th></tr></thead><tbody>${body}</tbody></table>`;
  for (const b of els.compare.querySelectorAll("button[data-show]")) {
    b.onclick = async () => {
      selectStrategy(b.dataset.show);
      await runIndex();
      els.compare.innerHTML = compareHtml;
      renderCompare(data);
      window.scrollTo({ top: document.getElementById("stage").offsetTop - 10, behavior: reducedMotion ? "auto" : "smooth" });
    };
  }
}

init().catch((err) => { els.modelBadge.textContent = `Fejl: ${err.message}`; });
