"""Train the CNN on MNIST and save the model + evaluation artifacts.
Run once:  python train_model.py
"""
import base64, io, json, os
import numpy as np, tensorflow as tf
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)
os.makedirs("model", exist_ok=True); os.makedirs("artifacts", exist_ok=True)

def png_b64(arr):  # 28x28 uint8 array -> base64 PNG string
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()

# 1. Load MNIST (60,000 train / 10,000 test)
(x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
# 2. Normalize 0-255 -> 0-1 and reshape to (N, 28, 28, 1)
xtr = (x_tr / 255.0).astype("float32")[..., None]
xte = (x_te / 255.0).astype("float32")[..., None]

# 3. Build the CNN
model = tf.keras.Sequential([
    tf.keras.Input((28, 28, 1)),
    tf.keras.layers.Conv2D(32, 3, activation="relu", name="conv1"),
    tf.keras.layers.MaxPooling2D(name="pool1"),
    tf.keras.layers.Conv2D(64, 3, activation="relu", name="conv2"),
    tf.keras.layers.MaxPooling2D(name="pool2"),
    tf.keras.layers.Flatten(name="flatten"),
    tf.keras.layers.Dense(128, activation="relu", name="dense"),
    tf.keras.layers.Dropout(0.3, name="dropout"),
    tf.keras.layers.Dense(10, activation="softmax", name="output"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 4. Train (10% of training data held out for validation); keep best weights
cbs = [tf.keras.callbacks.ModelCheckpoint("model/digit_cnn.keras", save_best_only=True, monitor="val_accuracy"),
       tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy")]
h = model.fit(xtr, y_tr, epochs=12, batch_size=128, validation_split=0.1, callbacks=cbs, verbose=2)
model.save("model/digit_cnn.keras")

# 5. Evaluate on the separate 10,000-image test set
loss, acc = model.evaluate(xte, y_te, verbose=0)
probs = model.predict(xte, verbose=0); pred = probs.argmax(1)
rep = classification_report(y_te, pred, output_dict=True)
wrong = np.where(pred != y_te)[0][:24]

metrics = {
    "history": {k: [float(v) for v in vs] for k, vs in h.history.items()},
    "test_accuracy": float(acc), "test_loss": float(loss),
    "correct": int((pred == y_te).sum()), "incorrect": int((pred != y_te).sum()),
    "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
    "report": {str(d): rep[str(d)] for d in range(10)},
    "misclassified": [{"image": png_b64(x_te[i]), "true": int(y_te[i]), "pred": int(pred[i])} for i in wrong],
    "class_counts": {"train": np.bincount(y_tr, minlength=10).tolist(), "test": np.bincount(y_te, minlength=10).tolist()},
    "params": int(model.count_params()),
    "optimizer": "Adam", "loss_fn": "Sparse categorical cross-entropy", "metric": "Accuracy",
    "layers": [{"name": l.name, "type": type(l).__name__, "output": str(l.output.shape[1:]), "params": int(l.count_params())} for l in model.layers],
}
json.dump(metrics, open("artifacts/metrics.json", "w"))
# 6. Save real sample images (4 per digit) for the Dataset Explorer
samples = [{"image": png_b64(x_tr[i]), "label": int(y_tr[i])} for d in range(10) for i in np.where(y_tr == d)[0][:4]]
json.dump(samples, open("artifacts/samples.json", "w"))
print(f"Done. Test accuracy = {acc:.4f}")
