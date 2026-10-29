"""Flask backend: serves pages, preprocesses images, runs the trained CNN."""
import base64, io, json, os, re, sqlite3
from datetime import datetime
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model", "digit_cnn.keras")
METRICS = os.path.join(BASE, "artifacts", "metrics.json")
SAMPLES = os.path.join(BASE, "artifacts", "samples.json")
DB = os.path.join(BASE, "instance", "history.db")
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # 4 MB upload limit
ALLOWED = {"png", "jpg", "jpeg"}
_model = None

def get_model():
    """Load the saved model once (never retrains)."""
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

def load_json(path):
    if not os.path.exists(path): return None
    with open(path) as f: return json.load(f)

def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY, ts TEXT, digit INT, conf REAL, method TEXT, thumb TEXT)")
    return c

def preprocess(img):
    """Any PIL image -> (1,28,28,1) float array in MNIST format, plus 28x28 uint8 for thumbnail.
    Steps: flatten alpha onto white, grayscale, fix polarity, crop digit, resize to 20px, centre by centre of mass."""
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA"); bg = Image.new("RGBA", img.size, "white"); bg.alpha_composite(img); img = bg
    g = np.asarray(img.convert("L"), dtype=np.float32)
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    if np.median(border) > 127: g = 255 - g           # dark-on-light -> light-on-dark like MNIST
    g = g - np.percentile(border, 50) if np.median(border) < 127 else g
    g = np.clip(g, 0, None)
    if g.max() < 40: raise ValueError("No digit detected. Please use a clearer image.")
    g = g / g.max() * 255
    ys, xs = np.where(g > 0.25 * 255)
    if len(ys) < 15: raise ValueError("No digit detected. Please use a clearer image.")
    crop = Image.fromarray(g[ys.min():ys.max()+1, xs.min():xs.max()+1].astype("uint8"))
    w, h = crop.size; s = 20.0 / max(w, h)
    crop = crop.resize((max(1, round(w*s)), max(1, round(h*s))), Image.LANCZOS)
    a = np.asarray(crop, dtype=np.float32)
    m = a.sum(); com_y = (a * np.arange(a.shape[0])[:, None]).sum()/m; com_x = (a * np.arange(a.shape[1])[None, :]).sum()/m
    out = np.zeros((28, 28), np.float32)
    oy = int(np.clip(round(14 - com_y), 0, 28 - a.shape[0])); ox = int(np.clip(round(14 - com_x), 0, 28 - a.shape[1]))
    out[oy:oy+a.shape[0], ox:ox+a.shape[1]] = a
    return (out / 255.0)[None, ..., None], out.astype("uint8")

def predict_and_log(img, method):
    model = get_model()
    if model is None:
        return jsonify(error="Model not trained yet. Run: python train_model.py"), 503
    try:
        x, thumb = preprocess(img)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception:
        return jsonify(error="Could not process this image."), 400
    p = model.predict(x, verbose=0)[0]; d = int(p.argmax()); conf = float(p[d]) * 100
    buf = io.BytesIO(); Image.fromarray(thumb).save(buf, "PNG")
    with db() as c:
        c.execute("INSERT INTO history(ts,digit,conf,method,thumb) VALUES(?,?,?,?,?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), d, conf, method, base64.b64encode(buf.getvalue()).decode()))
    return jsonify(digit=d, confidence=round(conf, 2), probabilities=[round(float(v)*100, 3) for v in p],
                   processed=base64.b64encode(buf.getvalue()).decode())

@app.route("/api/predict", methods=["POST"])
def predict():
    f = request.files.get("image")
    if not f or not f.filename: return jsonify(error="No file uploaded."), 400
    if f.filename.rsplit(".", 1)[-1].lower() not in ALLOWED: return jsonify(error="Only PNG, JPG and JPEG files are allowed."), 400
    try:
        img = Image.open(f.stream); img.load()  # verify it really decodes as an image
    except Exception:
        return jsonify(error="The file is not a valid image."), 400
    return predict_and_log(ImageOps.exif_transpose(img), "upload")

@app.route("/api/predict-drawing", methods=["POST"])
def predict_drawing():
    data = (request.get_json(silent=True) or {}).get("image", "")
    if not re.match(r"^data:image/png;base64,", data): return jsonify(error="Invalid drawing data."), 400
    try: img = Image.open(io.BytesIO(base64.b64decode(data.split(",", 1)[1]))); img.load()
    except Exception: return jsonify(error="Invalid drawing data."), 400
    return predict_and_log(img, "drawing")

@app.route("/api/status")
def api_status():
    m = load_json(METRICS)
    return jsonify(model_ready=os.path.exists(MODEL_PATH) and m is not None,
                   test_accuracy=m["test_accuracy"] if m else None, params=m["params"] if m else None)

@app.route("/api/analytics")
def api_analytics():
    m = load_json(METRICS)
    return (jsonify(m) if m else (jsonify(error="Model not trained yet. Run: python train_model.py"), 404))

@app.route("/api/dataset/samples")
def api_samples():
    s = load_json(SAMPLES); m = load_json(METRICS)
    if not s: return jsonify(error="Model not trained yet. Run: python train_model.py"), 404
    return jsonify(samples=s, class_counts=m["class_counts"] if m else None)

@app.route("/api/history", methods=["GET", "DELETE"])
def api_history():
    with db() as c:
        if request.method == "DELETE":
            c.execute("DELETE FROM history"); return jsonify(ok=True)
        rows = c.execute("SELECT * FROM history ORDER BY id DESC LIMIT 200").fetchall()
    return jsonify([dict(r) for r in rows])

PAGES = [("index", "/", "Overview", "layout-dashboard"), ("recognition", "/recognition", "Recognition Studio", "pen-tool"),
         ("dataset", "/dataset", "Dataset Explorer", "images"), ("architecture", "/architecture", "Model Architecture", "network"),
         ("analytics", "/analytics", "Model Analytics", "bar-chart-3"), ("history", "/history", "Prediction History", "history"),
         ("about", "/about", "About", "info")]
for name, path, *_ in PAGES:
    app.add_url_rule(path, name, (lambda n: lambda: render_template(f"{n}.html", page=n, pages=PAGES))(name))

@app.errorhandler(404)
def nf(e): return render_template("404.html", page="", pages=PAGES), 404
@app.errorhandler(413)
def big(e): return jsonify(error="File too large (max 4 MB)."), 413

if __name__ == "__main__":
    app.run(debug=False, port=5000)
