/* Recognition Studio: drawing canvas, upload, and calls to the Flask CNN endpoints. */
const cv = $("cv"), ctx = cv.getContext("2d");
let drawing = false, strokes = [], cur = null, file = null, chart = null;
const MAX = 4 * 1024 * 1024;

function redraw() {                                   // repaint white canvas from stored strokes (enables undo)
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, 280, 280);
  ctx.strokeStyle = "#000"; ctx.lineCap = ctx.lineJoin = "round";
  for (const s of strokes) { ctx.lineWidth = s.w; ctx.beginPath();
    s.p.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    if (s.p.length === 1) ctx.lineTo(s.p[0][0] + .1, s.p[0][1]); ctx.stroke(); }
}
const pos = (e) => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) * 280 / r.width, (e.clientY - r.top) * 280 / r.height]; };
cv.onpointerdown = (e) => { drawing = true; cv.setPointerCapture(e.pointerId); cur = { w: +$("brush").value, p: [pos(e)] }; strokes.push(cur); redraw(); };
cv.onpointermove = (e) => { if (drawing) { cur.p.push(pos(e)); redraw(); } };
cv.onpointerup = cv.onpointercancel = () => (drawing = false);
$("undo").onclick = () => { strokes.pop(); redraw(); };
$("clear").onclick = () => { strokes = []; redraw(); $("err").textContent = ""; };
redraw();

document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => {   // switch Draw / Upload
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("on", x === t));
  $("drawPane").hidden = t.dataset.mode !== "draw"; $("uploadPane").hidden = t.dataset.mode !== "upload"; $("err").textContent = "";
}));

function setFile(f) {                                  // validate type + size, then preview
  $("err").textContent = "";
  if (f && !/\.(png|jpe?g)$/i.test(f.name)) { $("err").textContent = "Only PNG, JPG and JPEG files are allowed."; f = null; }
  else if (f && f.size > MAX) { $("err").textContent = "File too large (max 4 MB)."; f = null; }
  file = f; $("goUp").disabled = !f; $("rm").hidden = !f; $("prev").hidden = !f;
  if (f) $("prev").src = URL.createObjectURL(f); else $("file").value = "";
}
$("file").onchange = (e) => setFile(e.target.files[0]);
$("rm").onclick = () => setFile(null);

async function send(promise) {
  $("err").textContent = ""; $("idle").hidden = true; $("out").hidden = true; $("spin").hidden = false;
  $("spin").innerHTML = '<span class="spinner"></span>Processing…';
  const { ok, data } = await promise; $("spin").hidden = true;
  if (!ok) { $("err").textContent = data.error || "Prediction failed."; $("idle").hidden = false; return; }
  $("digit").textContent = data.digit; $("conf").textContent = data.confidence.toFixed(2) + "%"; $("out").hidden = false;
  if (chart) chart.destroy();
  chart = new Chart($("probChart"), { type: "bar", data: { labels: [0,1,2,3,4,5,6,7,8,9],
    datasets: [{ label: "Probability (%)", data: data.probabilities,
      backgroundColor: data.probabilities.map((_, i) => (i === data.digit ? "#22d3ee" : "#2f6bff88")) }] },
    options: { indexAxis: "y", plugins: { legend: { display: false } },
      scales: { x: { max: 100, title: { display: true, text: "Probability (%)" } }, y: { title: { display: true, text: "Digit class" } } } } });
}
$("goDraw").onclick = () => strokes.length
  ? send(api("/api/predict-drawing", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image: cv.toDataURL("image/png") }) }))
  : ($("err").textContent = "Please draw a digit first.");
$("goUp").onclick = () => { const fd = new FormData(); fd.append("image", file); send(api("/api/predict", { method: "POST", body: fd })); };
$("again").onclick = () => { strokes = []; redraw(); setFile(null); $("out").hidden = true; $("idle").hidden = false; };
