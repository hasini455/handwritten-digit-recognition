/* Shared shell: theme, mobile menu, model status, plus Home and History pages. */
const $ = (id) => document.getElementById(id);
const api = async (url, opt) => {            // fetch wrapper that returns {ok, data}
  try { const r = await fetch(url, opt); return { ok: r.ok, data: await r.json() }; }
  catch { return { ok: false, data: { error: "Cannot reach the server." } }; }
};
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (v) => (v * 100).toFixed(2) + "%";
if (window.lucide) lucide.createIcons();

// Theme (remembered in localStorage) and mobile navigation
document.documentElement.dataset.theme = localStorage.getItem("theme") || "dark";
$("themeBtn").onclick = () => {
  const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = t; localStorage.setItem("theme", t);
};
$("menuBtn").onclick = () => $("side").classList.toggle("open");

// Model status indicator + Home accuracy card (real value from artifacts/metrics.json)
api("/api/status").then(({ data }) => {
  const s = $("status");
  s.textContent = data.model_ready ? "Model ready" : "Model not trained";
  s.className = "pill " + (data.model_ready ? "ok" : "no");
  if ($("acc")) $("acc").textContent = data.model_ready ? pct(data.test_accuracy) : "Not trained yet";
});

// Home: latest prediction preview
if ($("latest")) api("/api/history").then(({ ok, data }) => {
  if (ok && data.length) { const h = data[0];
    $("latest").innerHTML = `<img class="thumb" src="data:image/png;base64,${h.thumb}" alt="processed digit"> Digit <b>${h.digit}</b> · ${h.conf.toFixed(1)}% · ${esc(h.method)} · ${esc(h.ts)}`; }
});

// History page: table, digit filter, confirmed clear
if ($("ht")) {
  let rows = [];
  const draw = () => {
    const f = $("flt").value, v = rows.filter((r) => f === "" || r.digit == f);
    $("he").hidden = v.length > 0;
    $("ht").innerHTML = v.length ? "<tr><th>Time</th><th>Digit</th><th>Confidence</th><th>Input</th><th>Image</th></tr>" +
      v.map((r) => `<tr><td>${esc(r.ts)}</td><td><b>${r.digit}</b></td><td>${r.conf.toFixed(2)}%</td><td>${esc(r.method)}</td><td><img class="thumb" alt="processed" src="data:image/png;base64,${r.thumb}"></td></tr>`).join("") : "";
  };
  api("/api/history").then(({ data }) => { rows = Array.isArray(data) ? data : []; draw(); });
  $("flt").onchange = draw;
  $("clr").onclick = async () => {
    if (!confirm("Delete all prediction history? This cannot be undone.")) return;
    await api("/api/history", { method: "DELETE" }); rows = []; draw();
  };
}
