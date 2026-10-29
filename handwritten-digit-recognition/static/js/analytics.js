/* Dataset Explorer, Model Architecture and Model Analytics pages (all data comes from saved artifacts). */
const page = document.body.dataset.page;
const notTrained = "Model not trained yet. Run:  python train_model.py";
const imgTag = (b) => `<img alt="MNIST digit" src="data:image/png;base64,${b}">`;
const lineChart = (el, title, a, b, la, lb) => new Chart($(el), { type: "line",
  data: { labels: a.map((_, i) => i + 1), datasets: [{ label: la, data: a, borderColor: "#2f6bff" }, { label: lb, data: b, borderColor: "#7c4dff" }] },
  options: { plugins: { title: { display: true, text: title } }, scales: { x: { title: { display: true, text: "Epoch" } }, y: { title: { display: true, text: title } } } } });

if (page === "dataset") api("/api/dataset/samples").then(({ ok, data }) => {
  if (!ok) { $("gallery").innerHTML = `<p class="muted">${notTrained}</p>`; return; }
  $("gallery").innerHTML = data.samples.map((s) => `<figure>${imgTag(s.image)}<figcaption>Label: ${s.label}</figcaption></figure>`).join("");
  const c = data.class_counts, L = [0,1,2,3,4,5,6,7,8,9];
  new Chart($("distChart"), { type: "bar", data: { labels: L, datasets: [{ label: "Training images", data: c.train, backgroundColor: "#2f6bff" }, { label: "Test images", data: c.test, backgroundColor: "#22d3ee" }] },
    options: { scales: { x: { title: { display: true, text: "Digit class" } }, y: { title: { display: true, text: "Images" } } } } });
});

if (page === "architecture") api("/api/analytics").then(({ ok, data }) => {
  if (!ok) { $("layers").innerHTML = `<p class="muted">${notTrained}</p>`; return; }
  $("layers").innerHTML = data.layers.map((l) => `<span class="lay"><b>${esc(l.name)}</b> (${esc(l.type)})<span>Output: ${esc(l.output)}</span><span>Params: ${l.params.toLocaleString()}</span></span>`).join("");
  $("meta").textContent = `Total parameters: ${data.params.toLocaleString()} · Optimizer: ${data.optimizer} · Loss: ${data.loss_fn} · Metric: ${data.metric}`;
});

if (page === "analytics") api("/api/analytics").then(({ ok, data }) => {
  if (!ok) { $("empty").hidden = false; $("empty").innerHTML = `<h2>Model not trained yet</h2><p>${notTrained}</p>`; return; }
  $("data").hidden = false;
  $("ta").textContent = pct(data.test_accuracy); $("tl").textContent = data.test_loss.toFixed(4);
  $("ok").textContent = data.correct.toLocaleString(); $("bad").textContent = data.incorrect.toLocaleString();
  const h = data.history;
  lineChart("accC", "Accuracy", h.accuracy, h.val_accuracy, "Training accuracy", "Validation accuracy");
  lineChart("lossC", "Loss", h.loss, h.val_loss, "Training loss", "Validation loss");
  const cm = data.confusion_matrix, mx = Math.max(...cm.flat());
  $("cm").innerHTML = "<tr><th>True \\ Pred</th>" + cm.map((_, i) => `<th>${i}</th>`).join("") + "</tr>" +
    cm.map((r, i) => `<tr><th>${i}</th>` + r.map((v, j) => `<td style="background:rgba(${i === j ? "47,107,255" : "248,113,113"},${(v / mx * (i === j ? .6 : 3)).toFixed(2)})">${v}</td>`).join("") + "</tr>").join("");
  $("rep").innerHTML = "<tr><th>Digit</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Support</th></tr>" +
    Object.entries(data.report).map(([d, r]) => `<tr><td>${d}</td><td>${r.precision.toFixed(4)}</td><td>${r.recall.toFixed(4)}</td><td>${r["f1-score"].toFixed(4)}</td><td>${r.support}</td></tr>`).join("");
  $("mis").innerHTML = data.misclassified.map((m) => `<figure>${imgTag(m.image)}<figcaption>True ${m.true} → Pred ${m.pred}</figcaption></figure>`).join("");
});
