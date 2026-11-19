// Front end logic. All data goes through `api` (pywebview js_api, or the mock in demo mode).
const CLASSES = ["glioma", "meningioma", "notumor", "pituitary"];
// Okabe-Ito, the same colour for the same class on every screen
const CLASS_COLOURS = { glioma: "#E69F00", meningioma: "#0072B2", notumor: "#009E73", pituitary: "#CC79A7" };

const $ = (s, r) => (r || document).querySelector(s);
const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
const state = { lang: "en", theme: "light", view: "predict", mode: "register-off", demo: false, status: null,
                file: null, result: null, saved: false, queue: [], qi: 0, sel: null, busy: false, poll: null, armed: null };
let api = null;

// ---------- i18n ----------
function t(key, vars) {
  let s = (STRINGS[state.lang] && STRINGS[state.lang][key]) || STRINGS.en[key] || key;
  for (const k in (vars || {})) s = s.replace("{" + k + "}", vars[k]);
  return s;
}
function applyI18n() {
  document.documentElement.lang = state.lang;
  $$("[data-i18n]").forEach(el => { el.textContent = t(el.dataset.i18n); });
  $$("[data-i18n-html]").forEach(el => { el.innerHTML = t(el.dataset.i18nHtml); });
  $$("[data-i18n-aria]").forEach(el => { el.setAttribute("aria-label", t(el.dataset.i18nAria)); el.title = t(el.dataset.i18nAria); });
  $$("[data-i18n-ph]").forEach(el => { el.placeholder = t(el.dataset.i18nPh); });
  $$("[data-lang]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.lang === state.lang)));
  document.title = t("app.title");
  setLoginMode(state.mode);
}
function errText(code) { return t("err." + code); }
const fmt = (x) => Number(x).toLocaleString(state.lang === "pt" ? "pt-PT" : "en-GB", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const cname = (c) => t("class." + c);

// ---------- small helpers ----------
function toast(msg, isError) {
  const el = $("#toast");
  el.textContent = msg; el.className = "toast" + (isError ? " error" : ""); el.hidden = false;
  clearTimeout(toast.t1); clearTimeout(toast.t2);
  toast.t1 = setTimeout(() => { el.classList.add("out"); }, 3200);
  toast.t2 = setTimeout(() => { el.hidden = true; }, 3450);
}
function busy(btn, on) { btn.classList.toggle("is-busy", on); btn.disabled = on; }
function ripple(e) {
  const b = e.target.closest(".btn"); if (!b || b.disabled) return;
  const r = b.getBoundingClientRect(), size = Math.max(r.width, r.height) * 2;
  const s = document.createElement("span"); s.className = "ripple";
  s.style.cssText = "width:" + size + "px;height:" + size + "px;left:" + (e.clientX - r.left - size / 2) + "px;top:" + (e.clientY - r.top - size / 2) + "px";
  b.appendChild(s); setTimeout(() => s.remove(), 600);
}
function dot(c, large) { return '<span class="dot' + (large ? " dot-lg" : "") + '" style="background:' + CLASS_COLOURS[c] + '"></span>'; }
function sortedProbs(p) { return Object.entries(p).sort((a, b) => b[1] - a[1]); }
function renderBars(el, probs) {
  el.innerHTML = sortedProbs(probs).map(([c, v]) =>
    '<div class="bar"><span class="bar-name">' + cname(c) + '</span><div class="bar-track"><div class="bar-fill" data-w="' + (v * 100) +
    '" style="background:' + CLASS_COLOURS[c] + '"></div></div><span class="bar-val">' + fmt(v) + '</span></div>').join("");
  requestAnimationFrame(() => requestAnimationFrame(() => $$(".bar-fill", el).forEach(f => { f.style.width = f.dataset.w + "%"; })));
}
async function call(fn, ...args) {
  const r = await fn.apply(api, args);
  if (r && r.ok === false && r.error === "not_logged_in") { showLogin(); }
  return r;
}

// ---------- theme and language ----------
function setTheme(th) {
  state.theme = th; document.documentElement.dataset.theme = th;
  try { localStorage.setItem("theme", th); } catch (e) {}
  $$("[data-theme-toggle]").forEach(b => { b.innerHTML = icon(th === "dark" ? "sun" : "moon");
    const lab = th === "dark" ? "Light theme" : "Dark theme"; b.setAttribute("aria-label", lab); b.title = lab; });
}
function setLang(l) {
  state.lang = l; try { localStorage.setItem("lang", l); } catch (e) {}
  applyI18n(); if (!$("#app").hidden) { refreshAll(); }
}

// ---------- login ----------
function setLoginMode(mode) {
  state.mode = mode;
  const reg = mode === "register";
  $("#login-submit .btn-label").textContent = t(reg ? "login.register" : "login.signin");
  $("#login-switch-q").textContent = t(reg ? "login.haveaccount" : "login.noaccount");
  $("#login-switch").textContent = t(reg ? "login.signin_link" : "login.create");
  $("#l-pass").autocomplete = reg ? "new-password" : "current-password";
}
function showLogin() {
  stopPoll(); $("#app").hidden = true; $("#login").hidden = false;
  $("#login").style.animation = "none"; void $("#login").offsetWidth; $("#login").style.animation = "";
  $("#l-pass").value = ""; $("#login-error").hidden = true;
  drawBrain(); setTimeout(() => $("#l-user").focus(), 50);
}
async function submitLogin(e) {
  e.preventDefault();
  const u = $("#l-user").value.trim(), p = $("#l-pass").value, err = $("#login-error"), btn = $("#login-submit");
  err.hidden = true; busy(btn, true);
  try {
    if (state.mode === "register") {
      const r = await api.register(u, p);
      if (!r.ok) { err.textContent = errText(r.error); err.hidden = false; err.style.animation = "none"; void err.offsetWidth; err.style.animation = ""; return; }
      setLoginMode("login"); toast(t("login.created")); return;
    }
    const r = await api.login(u, p);
    if (!r.ok) { err.textContent = errText(r.error); err.hidden = false; err.style.animation = "none"; void err.offsetWidth; err.style.animation = ""; return; }
    enterApp(r.username);
  } finally { busy(btn, false); }
}
async function enterApp(username) {
  $("#login").hidden = true; $("#app").hidden = false; $("#username").textContent = username;
  state.file = null; state.result = null; resetPredict();
  switchView("predict", true); await refreshAll();
}

// ---------- views ----------
function placeInk() {
  const tab = $(".tab[aria-selected='true']"), ink = $(".tab-ink"); if (!tab) return;
  ink.style.left = tab.offsetLeft + "px"; ink.style.width = tab.offsetWidth + "px";
}
function switchView(name, instant) {
  const cur = $(".view:not([hidden])"), next = $("#view-" + name);
  state.view = name;
  $$(".tab").forEach(b => b.setAttribute("aria-selected", String(b.dataset.view === name)));
  placeInk();
  if (cur === next) return;
  const show = () => { if (cur) { cur.hidden = true; cur.classList.remove("leaving"); } next.hidden = false;
    $(".main").scrollTop = 0; if (name === "support") loadQueue(); if (name === "history") loadHistory(); };
  if (instant || !cur) show(); else { cur.classList.add("leaving"); setTimeout(show, 120); }
}
async function refreshStatus() {
  const s = await call(api.status);
  if (!s || !s.ok) return;
  state.status = s;
  const b = $("#badge"); b.hidden = !s.badge; if (s.badge && b.textContent !== String(s.badge)) { b.textContent = s.badge; b.style.animation = "none"; void b.offsetWidth; b.style.animation = ""; }
  $("#foot-mode").textContent = state.demo ? t("foot.demo") : (s.model_version ? t("foot.model", { v: s.model_version }) : "");
  $("#setup-path").textContent = s.weights_dir || "";
  if (!s.weights_installed) { $$(".view").forEach(v => { v.hidden = v.id !== "view-setup"; }); }
  else if ($("#view-setup") && !$("#view-setup").hidden) { $("#view-setup").hidden = true; $("#view-" + state.view).hidden = false; }
  renderModelCard(); renderProgress();
  requestAnimationFrame(placeInk);
}
async function refreshAll() {
  await refreshStatus(); fillFilters();
  if (state.view === "support") await loadQueue(); if (state.view === "history") await loadHistory();
  if (state.result) renderResult(); else renderPredictEmpty();
}

// ---------- prediction ----------
function resetPredict() {
  state.file = null; state.result = null; state.saved = false;
  $("#preview").hidden = true; $("#preview").removeAttribute("src"); $("#drop").classList.add("is-empty");
  $("#file-name").textContent = ""; $("#run").disabled = true; $("#predict-error").hidden = true; renderPredictEmpty();
}
function renderPredictEmpty() { $("#result").hidden = true; $("#result-empty").hidden = false; $("#mv-chip").hidden = true; }
function pickFile(file) {
  if (!file) return;
  state.file = file; state.result = null; state.saved = false;
  $("#file-name").textContent = file.name; $("#predict-error").hidden = true;
  const img = $("#preview"); img.src = URL.createObjectURL(file); img.hidden = false; $("#drop").classList.remove("is-empty");
  $("#run").disabled = false; renderPredictEmpty();
}
function fileToB64(file) {
  return new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res(String(r.result).split(",")[1]); r.onerror = rej; r.readAsDataURL(file); });
}
async function runPrediction() {
  if (!state.file) return;
  const btn = $("#run"); busy(btn, true); $("#predict-error").hidden = true;
  try {
    const r = await call(api.predict, state.file.name, await fileToB64(state.file));
    if (!r.ok) { const e = $("#predict-error"); e.textContent = errText(r.error); e.hidden = false; return; }
    state.result = r; state.saved = false; renderResult();
  } finally { busy(btn, false); btn.disabled = !state.file; }
}
function renderResult() {
  const r = state.result; if (!r) return renderPredictEmpty();
  $("#result-empty").hidden = true; $("#result").hidden = false;
  $("#mv-chip").hidden = false; $("#mv-chip").textContent = r.model_version;
  $("#r-dot").style.background = CLASS_COLOURS[r.label]; $("#r-label").textContent = cname(r.label);
  renderBars($("#r-bars"), r.probs);
  const top = sortedProbs(r.probs), note = $("#r-note");
  let html, cls = "note", ic = "warn";
  if (r.duplicate) { html = t("predict.note_dup"); cls = "note ok"; ic = "info"; }
  else if (r.low_margin) html = t("predict.note_low", { m: fmt(r.margin), a: cname(top[0][0]), b: cname(top[1][0]) });
  else { html = t("predict.note_ok", { m: fmt(r.margin), a: cname(top[0][0]), b: cname(top[1][0]) }); cls = "note ok"; ic = "info"; }
  note.className = cls; note.innerHTML = '<span class="ni">' + icon(ic) + '</span><div>' + html + '</div>';
  const locked = state.saved || r.duplicate;
  $("#r-review").disabled = locked; $("#r-save").disabled = locked;
  $("#r-review").classList.toggle("btn-primary", r.low_margin);
  $("#r-save").classList.toggle("btn-primary", !r.low_margin);
}
async function saveResult(toReview) {
  const r = state.result; if (!r) return;
  const res = await call(api.save, r.md5, toReview);
  if (!res.ok) { toast(errText(res.error), true); if (res.error === "duplicate") { r.duplicate = true; renderResult(); } return; }
  state.saved = true; renderResult(); toast(t(toReview ? "predict.sent" : "predict.saved")); refreshStatus();
}

// ---------- model support ----------
function renderProgress() {
  const s = state.status; if (!s) return;
  $("#progress-text").textContent = s.confirmed + " / " + s.retrain_min;
  $("#progress-fill").style.width = Math.min(100, 100 * s.confirmed / s.retrain_min) + "%";
}
function renderModelCard() {
  const s = state.status; if (!s) return;
  $("#m-version").textContent = s.model_version || "-";
  const left = Math.max(0, s.retrain_min - s.confirmed);
  $("#m-retrain").textContent = s.retraining ? t("support.retrain_running")
    : left > 0 ? t("support.retrain_starts", { min: s.retrain_min, left }) : t("support.retrain_ready", { n: s.confirmed });
  const btn = $("#retrain-btn");
  btn.disabled = s.retraining || left > 0; $(".spinner", btn).hidden = !s.retraining;
  btn.classList.toggle("btn-primary", !btn.disabled);
  const rounds = $("#rounds");
  rounds.innerHTML = s.rounds.length ? s.rounds.map(r => {
    const label = r.status === "done" ? t("support.round_done", { from: r.model_from, to: r.model_to })
      : r.status === "running" ? t("support.round_running", { from: r.model_from }) : t("support.round_failed", { from: r.model_from });
    return '<div class="round"><span>' + label + '<small>' + (r.finished_at || r.started_at).slice(0, 10) + '</small></span><span class="muted">' + t("support.round_n", { n: r.n_images }) + '</span></div>';
  }).join("") : '<p class="muted small">' + t("support.no_rounds") + '</p>';
}
async function loadQueue() {
  const r = await call(api.queue); if (!r || !r.ok) return;
  state.queue = r.items; state.qi = Math.min(state.qi, Math.max(0, state.queue.length - 1)); showQueueItem();
}
async function showQueueItem() {
  const items = state.queue, empty = items.length === 0;
  $("#sup-empty").hidden = !empty; $("#sup-body").hidden = empty;
  $("#sup-image-card").style.visibility = empty ? "hidden" : "visible";
  if (empty) return;
  const it = items[state.qi]; state.sel = null;
  $("#sup-count").textContent = t("support.pending_of", { i: state.qi + 1, n: items.length });
  $("#sup-margin").textContent = t("support.margin", { m: fmt(it.margin) });
  $("#sup-name").textContent = it.name;
  $("#sup-prev").disabled = state.qi === 0; $("#sup-next").disabled = state.qi >= items.length - 1;
  $("#s-dot").style.background = CLASS_COLOURS[it.predicted]; $("#s-label").textContent = cname(it.predicted);
  renderBars($("#s-bars"), it.probs); $("#s-notes").value = "";
  $("#s-classes").innerHTML = CLASSES.map(c => '<button type="button" class="class-btn" data-c="' + c + '" aria-pressed="false">' + dot(c) + cname(c) + '</button>').join("");
  const img = $("#sup-img"); img.hidden = true; $("#sup-img-load").hidden = false;
  const r = await call(api.image, it.id);
  if (items[state.qi] === it && r && r.ok) { img.src = r.image; img.hidden = false; $("#sup-img-load").hidden = true; }
}
async function decide(trueClass) {
  const it = state.queue[state.qi]; if (!it) return;
  const r = await call(api.review, it.id, trueClass, $("#s-notes").value);
  if (!r.ok) { toast(errText(r.error), true); return loadQueue(); }
  const wasSame = trueClass === it.predicted;
  state.queue.splice(state.qi, 1); state.qi = Math.min(state.qi, Math.max(0, state.queue.length - 1));
  toast(t(wasSame ? "support.confirmed_toast" : "support.corrected_toast"));
  const body = $("#sup-body"); body.classList.add("slide-out");
  setTimeout(() => { body.classList.remove("slide-out"); showQueueItem(); }, 200);
  refreshStatus();
}
async function startRetrain() {
  const r = await call(api.retrain); if (!r.ok) { toast(errText(r.error), true); return; }
  await refreshStatus(); startPoll();
}
function startPoll() {
  stopPoll();
  state.poll = setInterval(async () => {
    await refreshStatus();
    const s = state.status; if (s && !s.retraining) {
      stopPoll(); const l = s.last_round;
      if (l) toast(l.status === "promoted" ? t("support.retrain_ok", { v: l.version }) : l.status === "rejected" ? t("support.retrain_rejected") : t("support.retrain_failed"), l.status !== "promoted");
    }
  }, 2000);
}
function stopPoll() { if (state.poll) { clearInterval(state.poll); state.poll = null; } }

// ---------- history ----------
function fillFilters() {
  const c = $("#h-class"), s = $("#h-status"), cv = c.value, sv = s.value;
  c.innerHTML = '<option value="">' + t("history.all_classes") + '</option>' + CLASSES.map(x => '<option value="' + x + '">' + cname(x) + '</option>').join("");
  s.innerHTML = '<option value="">' + t("history.all_status") + '</option>' + ["saved", "pending", "confirmed", "corrected"].map(x => '<option value="' + x + '">' + t("status." + x) + '</option>').join("");
  c.value = cv; s.value = sv;
}
const STATUS_ICON = { saved: "save", pending: "clock", confirmed: "check", corrected: "pencil" };
function fmtDate(iso) { try { return new Date(iso).toLocaleDateString(state.lang === "pt" ? "pt-PT" : "en-GB", { day: "numeric", month: "short", year: "numeric" }); } catch (e) { return iso.slice(0, 10); } }
let hq;
async function loadHistory() {
  const r = await call(api.history, $("#h-q").value, $("#h-class").value || null, $("#h-status").value || null);
  if (!r || !r.ok) return;
  $("#h-empty").hidden = r.rows.length > 0; $("#h-count").textContent = t("history.count", { n: r.rows.length });
  $("#h-export").disabled = r.rows.length === 0; state.armed = null;
  $("#h-rows").innerHTML = r.rows.map((x, i) => {
    const pred = '<span class="cell-pred">' + dot(x.predicted) + cname(x.predicted) +
      (x.true_class && x.true_class !== x.predicted ? '<span class="muted">' + t("history.to") + '</span>' + dot(x.true_class) + cname(x.true_class) : "") + '</span>';
    const pill = '<span class="pill ' + x.status + '">' + icon(STATUS_ICON[x.status], 14) + t("status." + x.status) + '</span>';
    const send = x.status === "saved" ? '<button class="mini" data-act="send" data-id="' + x.id + '" title="' + t("history.send") + '" aria-label="' + t("history.send") + '">' + icon("send", 16) + '</button>' : "";
    const del = '<button class="mini" data-act="del" data-id="' + x.id + '" ' + (x.locked ? 'disabled title="' + t("history.locked") + '"' : 'title="' + t("history.delete") + '"') + ' aria-label="' + t("history.delete") + '">' + icon("trash", 16) + '</button>';
    return '<div class="trow" style="animation-delay:' + Math.min(i, 12) * 22 + 'ms"><span class="muted">' + fmtDate(x.date) + '</span><span class="cell-name" title="' + x.name.replace(/"/g, "&quot;") + '"></span>' + pred +
      '<span style="font-variant-numeric:tabular-nums">' + fmt(x.margin) + '</span><span class="muted">' + x.model + '</span>' + pill + '<span class="row-actions">' + send + del + '</span></div>';
  }).join("");
  // file names are set as text so that a name can never inject markup
  $$("#h-rows .trow").forEach((row, i) => { $(".cell-name", row).textContent = r.rows[i].name; });
}
async function historyClick(e) {
  const b = e.target.closest(".mini"); if (!b || b.disabled) return;
  const id = Number(b.dataset.id);
  if (b.dataset.act === "send") { const r = await call(api.send_to_review, id); if (r.ok) { toast(t("predict.sent")); await loadHistory(); refreshStatus(); } return; }
  if (state.armed !== id) { $$(".mini.armed").forEach(x => { x.classList.remove("armed"); x.title = t("history.delete"); }); state.armed = id; b.classList.add("armed"); b.title = t("history.delete_sure"); toast(t("history.delete_sure")); return; }
  const r = await call(api.delete, id); if (!r.ok) { toast(errText(r.error), true); return; }
  toast(t("history.deleted")); await loadHistory(); refreshStatus();
}
async function exportCsv() {
  const r = await call(api.export_csv); if (!r || !r.ok) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob(["﻿" + r.csv], { type: "text/csv;charset=utf-8" })); a.download = "history.csv"; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000); toast(t("history.exported"));
}

// ---------- login artwork: topographic brain contours that draw themselves ----------
function drawBrain() {
  const el = $("#login-art"); if (el.dataset.done) return; el.dataset.done = "1";
  const cx = 400, cy = 410, P = [];
  const contour = (sc, ph, a1, a2) => { const pts = [];
    for (let i = 0; i <= 360; i += 1) { const a = i * Math.PI / 180, r = 1 + (a1 * Math.sin(7 * a + ph) + a2 * Math.sin(19 * a + ph * 1.7)) / 100;
      pts.push((cx + Math.cos(a) * 205 * sc * r).toFixed(1) + "," + (cy + Math.sin(a) * 262 * sc * r).toFixed(1)); }
    return "M" + pts.join(" L") + "Z"; };
  const draw = (d, op, w, delay) => '<path class="draw" pathLength="1" d="' + d + '" fill="none" stroke-opacity="' + op + '" stroke-width="' + w + '" stroke-linejoin="round" style="animation-delay:' + delay + 's"/>';
  const ring = (rx, ry, op, w, delay) => '<ellipse class="draw" pathLength="1" cx="' + cx + '" cy="' + cy + '" rx="' + rx + '" ry="' + ry + '" fill="none" stroke-opacity="' + op + '" stroke-width="' + w + '" style="animation-delay:' + delay + 's"/>';
  P.push(ring(262, 318, .3, 2, 0), ring(246, 302, .2, 1.5, .1));
  [[1, 0, 3.2, 2.6], [.9, .9, 3.8, 3], [.8, 1.7, 4.4, 3.2], [.7, 2.6, 5, 3], [.6, 3.4, 5.4, 2.6], [.5, 4.2, 5.2, 2], [.4, 5, 4.4, 1.4]]
    .forEach((s, k) => P.push(draw(contour(...s), (.5 * (1 - k * .09)).toFixed(2), (2.2 - k * .15).toFixed(2), .2 + k * .18)));
  const fis = []; for (let y = cy - 262; y <= cy + 262; y += 4) fis.push((cx + 5 * Math.sin(y / 23)).toFixed(1) + "," + y);
  P.push(draw("M" + fis.join(" L"), .5, 2.4, .9));
  [-1, 1].forEach(sg => { const v = []; for (let a = 0; a <= 180; a += 6) v.push((cx + sg * (14 + 26 * Math.sin(a * Math.PI / 180) * .9 + 6)).toFixed(1) + "," + (cy - 40 + 78 * a / 180).toFixed(1)); P.push(draw("M" + v.join(" L"), .5, 2, 1.3)); });
  el.innerHTML = '<svg viewBox="0 0 800 820" preserveAspectRatio="xMidYMid slice"><g class="breathe">' + P.join("") + '</g></svg>';
}

// ---------- start ----------
function bind() {
  document.addEventListener("click", ripple);
  $$("[data-lang]").forEach(b => b.addEventListener("click", () => setLang(b.dataset.lang)));
  $$("[data-theme-toggle]").forEach(b => b.addEventListener("click", () => setTheme(state.theme === "dark" ? "light" : "dark")));
  $("#login-form").addEventListener("submit", submitLogin);
  $("#login-switch").addEventListener("click", () => { setLoginMode(state.mode === "register" ? "login" : "register"); $("#login-error").hidden = true; });
  $$(".tab").forEach(b => b.addEventListener("click", () => switchView(b.dataset.view)));
  $("#logout-btn").addEventListener("click", async () => { await api.logout(); showLogin(); });
  $("#about-btn").addEventListener("click", () => $("#about").showModal());
  $("#about-close").addEventListener("click", () => $("#about").close());
  $("#about").addEventListener("click", (e) => { if (e.target === $("#about")) $("#about").close(); });
  const drop = $("#drop");
  $("#choose").addEventListener("click", () => $("#file").click());
  drop.addEventListener("click", () => $("#file").click());
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#file").click(); } });
  $("#file").addEventListener("change", (e) => { pickFile(e.target.files[0]); e.target.value = ""; });
  ["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("is-over"); }));
  drop.addEventListener("drop", (e) => pickFile(e.dataTransfer.files[0]));
  window.addEventListener("dragover", (e) => e.preventDefault()); window.addEventListener("drop", (e) => e.preventDefault());
  $("#run").addEventListener("click", runPrediction);
  $("#r-save").addEventListener("click", () => saveResult(false));
  $("#r-review").addEventListener("click", () => saveResult(true));
  $("#sup-prev").addEventListener("click", () => { state.qi--; showQueueItem(); });
  $("#sup-next").addEventListener("click", () => { state.qi++; showQueueItem(); });
  $("#s-confirm").addEventListener("click", () => { const it = state.queue[state.qi]; if (it) decide(it.predicted); });
  $("#s-classes").addEventListener("click", (e) => { const b = e.target.closest(".class-btn"); if (b) decide(b.dataset.c); });
  $("#retrain-btn").addEventListener("click", startRetrain);
  $("#setup-retry").addEventListener("click", refreshAll);
  $("#h-q").addEventListener("input", () => { clearTimeout(hq); hq = setTimeout(loadHistory, 200); });
  $("#h-class").addEventListener("change", loadHistory); $("#h-status").addEventListener("change", loadHistory);
  $("#h-rows").addEventListener("click", historyClick); $("#h-export").addEventListener("click", exportCsv);
  $$("[data-logo]").forEach(el => { el.innerHTML = icon("brain", 17); });
  $$("[data-icon]").forEach(el => { el.innerHTML = icon(el.dataset.icon, el.classList.contains("empty-icon") || el.classList.contains("drop-icon") ? 32 : 18); });
  window.addEventListener("resize", placeInk);
}

function waitForApi() {
  return new Promise((resolve) => {
    if (window.pywebview && window.pywebview.api) return resolve(window.pywebview.api);
    let done = false;
    window.addEventListener("pywebviewready", () => { if (!done) { done = true; resolve(window.pywebview.api); } });
    // opened in a plain browser: use the demo data
    setTimeout(() => { if (!done) { done = true; const s = document.createElement("script"); s.src = "mock.js"; s.onload = () => { state.demo = true; resolve(window.MockApi); }; document.head.appendChild(s); } }, 700);
  });
}

(async function start() {
  try { state.lang = localStorage.getItem("lang") || ((navigator.language || "").startsWith("pt") ? "pt" : "en"); } catch (e) {}
  let th = null; try { th = localStorage.getItem("theme"); } catch (e) {}
  setTheme(th || (window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  bind(); applyI18n();
  api = await waitForApi();
  const s = await api.session();
  if (s.logged_in) enterApp(s.username); else showLogin();
})();
