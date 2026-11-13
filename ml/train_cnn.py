"""Train the image recognition CNN with transfer learning (MobileNetV2, ImageNet weights).

Expected layout (one folder per class; the folder name is the label shown to users):

    data/raw/images/
        Hyundai Creta/  img1.jpg ...
        Tata Nexon/     ...
    # or, if it already comes split:  data/raw/images/train/<class>/..  and  .../test/<class>/..

    python -m ml.train_cnn --inspect          # just count classes/images, no training
    python -m ml.train_cnn --epochs 10 --fine-tune-epochs 10

Requires:  pip install -r requirements-cnn.txt

This CNN recognises brand/model only. Because vehicle photos are usually NOT paired row-by-row with
prices, its output is not used as a price feature. Test metrics are computed on a held-out split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from ml import config as C

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def collect(root: Path):
    samples = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in cls_dir.rglob("*"):
            if f.suffix.lower() in EXTS:
                samples.append((str(f), cls_dir.name))
    return samples


def load_samples(data_dir: Path):
    """Returns (train_val_samples, test_samples_or_None)."""
    subs = {p.name.lower(): p for p in data_dir.iterdir() if p.is_dir()}
    if "train" in subs and ("test" in subs or "val" in subs or "valid" in subs):
        test_dir = subs.get("test") or subs.get("val") or subs.get("valid")
        return collect(subs["train"]), collect(test_dir)
    return collect(data_dir), None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default=str(C.IMAGES_DIR))
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--fine-tune-epochs", type=int, default=10)
    ap.add_argument("--fine-tune-layers", type=int, default=40)
    ap.add_argument("--min-images", type=int, default=10, help="drop classes with fewer images")
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists() or not any(data_dir.iterdir()):
        raise SystemExit(f"No images found in {data_dir}. Put one folder per class there.")
    trainval, test_given = load_samples(data_dir)

    counts: dict[str, int] = {}
    for _, lab in trainval + (test_given or []):
        counts[lab] = counts.get(lab, 0) + 1
    keep = {c for c, n in counts.items() if n >= args.min_images}
    print(f"Classes: {len(counts)} | images: {sum(counts.values()):,} | "
          f"classes kept (>= {args.min_images} images): {len(keep)}")
    if len(keep) < 2:
        raise SystemExit("Need at least 2 classes with enough images.")
    if args.inspect:
        for c, n in sorted(counts.items(), key=lambda kv: -kv[1])[:20]:
            print(f"  {c:40} {n}")
        return

    try:
        import numpy as np
        import tensorflow as tf
        from sklearn.metrics import f1_score
    except ImportError:
        raise SystemExit("TensorFlow is not installed. Run:  pip install -r requirements-cnn.txt")

    trainval = [s for s in trainval if s[1] in keep]
    classes = sorted(keep)
    cid = {c: i for i, c in enumerate(classes)}
    paths = [p for p, _ in trainval]
    labels = [cid[l] for _, l in trainval]
    if test_given is not None:
        tr_p, va_p, tr_l, va_l = train_test_split(paths, labels, test_size=args.val_size,
                                                  stratify=labels, random_state=args.seed)
        te_p = [p for p, l in test_given if l in keep]
        te_l = [cid[l] for _, l in test_given if l in keep]
    else:
        tr_p, te_p, tr_l, te_l = train_test_split(paths, labels, test_size=args.test_size,
                                                  stratify=labels, random_state=args.seed)
        tr_p, va_p, tr_l, va_l = train_test_split(tr_p, tr_l, test_size=args.val_size / (1 - args.test_size),
                                                  stratify=tr_l, random_state=args.seed)
    print(f"train {len(tr_p):,} | val {len(va_p):,} | test {len(te_p):,}")

    size = (args.img_size, args.img_size)

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, size)
        img.set_shape((*size, 3))
        return img, label

    def ds(p, l, shuffle):
        d = tf.data.Dataset.from_tensor_slices((p, l))
        if shuffle:
            d = d.shuffle(len(p), seed=args.seed)
        return d.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds, val_ds, test_ds = ds(tr_p, tr_l, True), ds(va_p, va_l, False), ds(te_p, te_l, False)

    base = tf.keras.applications.MobileNetV2(input_shape=(*size, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = tf.keras.Input(shape=(*size, 3))
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)          # augmentation: active only in training
    x = tf.keras.layers.RandomRotation(0.05)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(x)         # MobileNetV2 preprocessing, inside the model
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(classes), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    def compile_(lr):
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    compile_(1e-3)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[early])
    if args.fine_tune_epochs > 0:
        base.trainable = True
        for layer in base.layers[:-args.fine_tune_layers]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        compile_(1e-5)
        model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, callbacks=[early])

    probs = model.predict(test_ds, verbose=0)
    y_true = np.array(te_l)
    top1 = float(np.mean(probs.argmax(1) == y_true))
    top3 = float(np.mean([t in np.argsort(p)[::-1][:3] for t, p in zip(y_true, probs)]))
    macro_f1 = float(f1_score(y_true, probs.argmax(1), average="macro"))

    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(C.CNN_MODEL_PATH))
    C.CNN_CLASSES_PATH.write_text(json.dumps({"classes": classes, "img_size": args.img_size},
                                             indent=2, ensure_ascii=False), encoding="utf-8")
    metrics = {"architecture": "MobileNetV2 (ImageNet transfer learning)", "n_classes": len(classes),
               "n_train": len(tr_p), "n_val": len(va_p), "n_test": len(te_p),
               "test_top1_accuracy": round(top1, 4), "test_top3_accuracy": round(top3, 4),
               "test_macro_f1": round(macro_f1, 4)}
    C.CNN_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\n=== CNN TEST RESULTS (real) ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
