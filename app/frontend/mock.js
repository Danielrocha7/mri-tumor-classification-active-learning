// Demo backend used when index.html is opened in a plain browser (no pywebview). Sample data only; nothing is classified.
(function () {
  const CLS = ["glioma", "meningioma", "notumor", "pituitary"];
  const delay = (ms) => new Promise(r => setTimeout(r, ms));
  const hash = (s) => { let h = 2166136261; for (const c of s) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); } return h >>> 0; };
  const margin = (p) => { const v = Object.values(p).sort((a, b) => b - a); return v[0] - v[1]; };
  const norm = (a) => { const s = a.reduce((x, y) => x + y, 0); return a.map(v => v / s); };
  function probsFor(seed) { let h = seed; const rnd = () => { h = Math.imul(h ^ (h >>> 15), 2246822507) >>> 0; return (h % 1000) / 1000; };
    const w = norm(CLS.map(() => Math.pow(rnd(), 3) + 0.01)); const o = {}; CLS.forEach((c, i) => o[c] = w[i]); return o; }
  function art(seed) { const c = document.createElement("canvas"); c.width = c.height = 320; const g = c.getContext("2d");
    g.fillStyle = "#111"; g.fillRect(0, 0, 320, 320);
    for (let i = 0; i < 7; i++) { g.strokeStyle = "rgba(255,255,255," + (0.35 - i * 0.04) + ")"; g.lineWidth = 3; g.beginPath(); g.ellipse(160, 160, 120 - i * 14, 140 - i * 16, 0, 0, 7); g.stroke(); }
    g.fillStyle = "rgba(255,255,255,.25)"; g.beginPath(); g.arc(120 + (seed % 80), 120 + (seed % 60), 22, 0, 7); g.fill(); return c.toDataURL("image/png"); }
  let id = 100, user = null, version = 1, running = false, last = null;
  const rows = [];
  const add = (name, seed, status, trueClass, daysAgo) => { const p = probsFor(seed), pr = Object.entries(p).sort((a, b) => b[1] - a[1])[0][0];
    rows.push({ id: ++id, name, probs: p, predicted: pr, true_class: trueClass || null, margin: margin(p), model: "v1", status, seed, locked: false,
      date: new Date(Date.now() - daysAgo * 864e5).toISOString() }); };
  [["sample_slice_001.png", 11, "pending", null, 0], ["sample_slice_002.png", 12, "confirmed", "glioma", 1], ["sample_slice_003.png", 13, "pending", null, 1],
   ["sample_slice_004.png", 14, "saved", null, 2], ["sample_slice_005.png", 15, "pending", null, 3], ["sample_slice_006.png", 16, "corrected", "pituitary", 3],
   ["sample_slice_007.png", 17, "pending", null, 4]].forEach(a => add(...a));
  const confirmed = () => rows.filter(r => (r.status === "confirmed" || r.status === "corrected") && !r.locked).length + 12;
  const rounds = [];
  window.MockApi = {
    async session() { return { logged_in: !!user, username: user }; },
    async register(u, p) { await delay(250); if (!/^[A-Za-z0-9._-]{3,32}$/.test(u)) return { ok: false, error: "bad_username" };
      if (p.length < 8) return { ok: false, error: "short_password" }; return { ok: true }; },
    async login(u, p) { await delay(300); if (!u || !p) return { ok: false, error: "bad_credentials" }; user = u; return { ok: true, username: u }; },
    async logout() { user = null; return { ok: true }; },
    async status() { const pend = rows.filter(r => r.status === "pending");
      return { ok: true, weights_installed: true, model_version: "v" + version, badge: pend.filter(r => r.margin < 0.8).length, pending: pend.length,
        confirmed: confirmed(), retrain_min: 20, margin_threshold: 0.8, rounds: rounds.slice().reverse(), retraining: running, last_round: null, weights_dir: "~/.mri-classifier/models" }; },
    async predict(name, b64) { await delay(700); if (!b64) return { ok: false, error: "not_image" };
      const seed = hash(name), p = probsFor(seed), top = Object.entries(p).sort((a, b) => b[1] - a[1]), m = margin(p);
      last = { name, probs: p, seed }; const dup = rows.some(r => r.name === name);
      return { ok: true, md5: "demo" + seed, name, label: top[0][0], confidence: top[0][1], margin: m, probs: p, model_version: "v" + version,
        low_margin: m < 0.8, duplicate: dup, image: "data:image/png;base64," + b64 }; },
    async save(md5, toReview) { if (!last) return { ok: false, error: "unknown_image" }; await delay(150);
      const pr = Object.entries(last.probs).sort((a, b) => b[1] - a[1])[0][0];
      rows.unshift({ id: ++id, name: last.name, probs: last.probs, predicted: pr, true_class: null, margin: margin(last.probs), model: "v" + version,
        status: toReview ? "pending" : "saved", seed: last.seed, locked: false, date: new Date().toISOString() }); return { ok: true, id }; },
    async queue() { return { ok: true, items: rows.filter(r => r.status === "pending").sort((a, b) => a.margin - b.margin)
      .map(r => ({ id: r.id, name: r.name, probs: r.probs, predicted: r.predicted, margin: r.margin })) }; },
    async image(i) { const r = rows.find(x => x.id === i); await delay(120); return r ? { ok: true, image: art(r.seed) } : { ok: false, error: "missing_image" }; },
    async review(i, c) { const r = rows.find(x => x.id === i && x.status === "pending"); if (!r) return { ok: false, error: "not_pending" };
      r.status = r.predicted === c ? "confirmed" : "corrected"; r.true_class = c; return { ok: true }; },
    async history(q, cls, st) { const s = (q || "").toLowerCase();
      return { ok: true, rows: rows.filter(r => (!s || r.name.toLowerCase().includes(s)) && (!cls || r.predicted === cls) && (!st || r.status === st))
        .map(r => ({ id: r.id, name: r.name, date: r.date, predicted: r.predicted, true_class: r.true_class, margin: r.margin, model: r.model, status: r.status, locked: r.locked })) }; },
    async send_to_review(i) { const r = rows.find(x => x.id === i && x.status === "saved"); if (r) r.status = "pending"; return { ok: !!r }; },
    async delete(i) { const k = rows.findIndex(x => x.id === i); if (k < 0 || rows[k].locked) return { ok: false, error: "in_training" }; rows.splice(k, 1); return { ok: true }; },
    async export_csv() { const h = await this.history(); return { ok: true, csv: "date,image,predicted,true_class,margin,model,status\n" +
      h.rows.map(r => [r.date, r.name, r.predicted, r.true_class || "", r.margin.toFixed(4), r.model, r.status].join(",")).join("\n") + "\n" }; },
    async retrain() { if (confirmed() < 20) return { ok: false, error: "not_enough" }; running = true;
      setTimeout(() => { running = false; version++; rounds.push({ model_from: "v" + (version - 1), model_to: "v" + version, n_images: 20, status: "done", started_at: new Date().toISOString(), finished_at: new Date().toISOString() }); }, 4000);
      return { ok: true }; },
  };
})();
