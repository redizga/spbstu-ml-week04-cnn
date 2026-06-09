import os, glob, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from PIL import Image

gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
tf.keras.mixed_precision.set_global_policy("mixed_float16")
print("GPUs:", gpus)

IMG_SIZE = 128
BATCH_SIZE = 512
CLASSES = ["cat", "dog", "gungan"]
DATA_DIR = "/workspace/spbstu-ml-week04-cnn/data"
OUT_DIR  = "/workspace/spbstu-ml-week04-cnn"
BEST_W   = f"{OUT_DIR}/cat_dog_gungan_cnn.h5"

def load_split(split):
    images, labels = [], []
    for idx, cls in enumerate(CLASSES):
        for p in sorted(glob.glob(f"{DATA_DIR}/{split}/{cls}/*.jpg")):
            img = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            images.append(np.array(img, dtype=np.float32) / 255.0)
            labels.append(idx)
    return np.array(images), np.array(labels)

print("Loading train...", flush=True)
X_train, y_train = load_split("train")
print(f"Train: {X_train.shape}  {dict(zip(*np.unique(y_train, return_counts=True)))}")
print("Loading test...", flush=True)
X_test, y_test = load_split("test")
print(f"Test:  {X_test.shape}  {dict(zip(*np.unique(y_test, return_counts=True)))}")

y_train_cat = tf.keras.utils.to_categorical(y_train, len(CLASSES))
y_test_cat  = tf.keras.utils.to_categorical(y_test,  len(CLASSES))

def build_model():
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.RandomFlip("horizontal")(inp)
    x = layers.RandomRotation(0.12)(x)
    x = layers.RandomZoom(0.15)(x)
    x = layers.RandomTranslation(0.1, 0.1)(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(len(CLASSES), activation="softmax", dtype="float32")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model

model = build_model()
model.summary()

cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
cw_dict = dict(enumerate(cw))
print("Class weights:", {k: round(v, 2) for k, v in cw_dict.items()})

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    keras.callbacks.ModelCheckpoint(BEST_W, monitor="val_accuracy", save_best_only=True, verbose=1)
]

history = model.fit(
    X_train, y_train_cat,
    batch_size=BATCH_SIZE, epochs=50,
    validation_data=(X_test, y_test_cat),
    callbacks=callbacks,
    class_weight=cw_dict,
    verbose=1
)

# Load best weights saved by ModelCheckpoint
model.load_weights(BEST_W)
print("Loaded best weights from checkpoint.")

with open(f"{OUT_DIR}/training_history.json", "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest accuracy: {acc:.4f}  loss: {loss:.4f}")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print(classification_report(y_test, y_pred, target_names=CLASSES, labels=[0,1,2]))

meta = {"test_accuracy": float(acc), "test_loss": float(loss),
        "img_size": IMG_SIZE, "classes": CLASSES,
        "epochs_trained": len(history.history["loss"])}
with open(f"{OUT_DIR}/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Done.")
