import os
import sys
import json
import numpy as np
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

weights_path = os.path.abspath("ml/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5")
model = tf.keras.applications.MobileNetV2(weights=weights_path)

with open("ml/models/imagenet_class_index.json", "r", encoding="utf-8") as f:
    class_index = json.loads(f.read())

print("Model & class index loaded successfully.")

# Find sample cars
car_dirs = [os.path.join("data/raw/images", d) for d in os.listdir("data/raw/images") if os.path.isdir(os.path.join("data/raw/images", d))]
sample_cars = []
for cd in car_dirs[:3]:
    for f in os.listdir(cd):
        if f.lower().endswith((".jpg", ".png", ".webp")):
            sample_cars.append(os.path.join(cd, f))
            break

for sc in sample_cars:
    print("\nTesting car image:", sc)
    img = Image.open(sc).convert("RGB").resize((224, 224))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img, dtype=np.float32)[None, ...])
    preds = model.predict(x, verbose=0)[0]
    top_indices = np.argsort(preds)[::-1][:5]
    for idx in top_indices:
        entry = class_index[str(idx)]
        print(f"  [{idx}] {entry[1]} ({entry[0]}): {preds[idx]:.4f}")
