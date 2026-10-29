# AI-Based Handwritten Digit Recognition System Using CNN

Flask web app + TensorFlow/Keras CNN trained on MNIST. Upload or draw a digit and get a real CNN prediction with all ten class probabilities.

## Prerequisites
Python 3.10 – 3.12 (TensorFlow does not yet support the newest Python versions), internet access for the first run
(MNIST download, and Chart.js / Lucide loaded from CDNs in the browser). CPU only is fine.

## Setup and run (exact commands)
```bash
cd handwritten-digit-recognition
python -m venv venv
# Windows:  venv\Scripts\activate        macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt
python train_model.py        # ONE-TIME: trains the CNN (a few minutes on CPU), saves model/ and artifacts/
python app.py                # starts the server (does NOT retrain)
```
Open **http://127.0.0.1:5000**. Until you run `train_model.py`, the site shows "Model not trained yet".

## How to use
- **Recognition Studio → Draw**: draw with mouse/touch, adjust stroke, Undo/Clear, press *Recognize Drawing*.
- **Recognition Studio → Upload**: choose PNG/JPG/JPEG (max 4 MB), preview, press *Recognize Digit*.
- **Analytics / Dataset / Architecture / History** pages read the files saved by training and the SQLite history.

## Project structure
```
app.py            Flask routes, image preprocessing, model loading, history (SQLite)
train_model.py    Trains + evaluates CNN, writes model/ and artifacts/
model/            digit_cnn.keras (created by training)
artifacts/        metrics.json, samples.json (created by training)
templates/        base, index, recognition, dataset, architecture, analytics, history, about, 404
static/           css/style.css, js/main.js, recognition.js, analytics.js
instance/         history.db (created on first prediction)
```

## CNN architecture (as implemented in train_model.py)
Input 28×28×1 → Conv2D(32, 3×3, ReLU) → MaxPool → Conv2D(64, 3×3, ReLU) → MaxPool → Flatten → Dense(128, ReLU) → Dropout(0.3) → Dense(10, softmax).
Optimizer Adam, loss sparse categorical cross-entropy, metric accuracy. The Architecture page reads the layers and parameter count from the saved results, so it always matches the trained model.

## Dataset and evaluation
MNIST via `tf.keras.datasets.mnist`: 60,000 training / 10,000 test images, 28×28 grayscale, 10 classes. Pixels are scaled to 0–1 and reshaped to (28, 28, 1).
10% of the training set is held out for validation; the separate 10,000-image test set is used only for final evaluation (accuracy, loss, confusion matrix, per-class precision/recall/F1, misclassified examples). Seed 42 is set, but results can still vary slightly between machines.

## Image preprocessing (viva note)
Transparency is flattened onto white → grayscale → polarity detected from the border (MNIST is light digit on dark background, so dark-on-light images are inverted) → digit cropped → scaled so its longest side is 20 px → centred by centre of mass in 28×28 → scaled to 0–1. Blank images are rejected. Uploads and drawings share this pipeline.

## Troubleshooting
- `No module named tensorflow`: use Python 3.10–3.12 and re-run `pip install -r requirements.txt`.
- "Model not trained yet": run `python train_model.py` first.
- Icons/charts missing: the browser needs internet for the CDN scripts.
- Port 5000 busy: change `port=5000` at the bottom of `app.py`.

## Known limitations
Trained only on clean, single, centred MNIST-style digits; photos with clutter, touching digits, letters or unusual styles may be misclassified. Confidence is a probability estimate, not a guarantee. This is an academic prototype, not production-ready.
