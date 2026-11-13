import os
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

# Extended vehicle categories including road/passenger/utility vehicles & automotive parts
VEHICLE_CLASSES = {
    "sports_car", "convertible", "minivan", "pickup", "limousine",
    "cab", "jeep", "beach_wagon", "racer", "police_van", "ambulance",
    "moving_van", "tow_truck", "trailer_truck", "fire_engine",
    "garbage_truck", "recreational_vehicle", "minibus", "school_bus",
    "grille", "car_wheel", "snowplow", "golfcart"
}

vehicle_indices = [int(k) for k, (synset, name) in class_index.items() if name in VEHICLE_CLASSES]

# Test on 40 diverse car images across the 20 classes
car_scores = []
car_root = "data/raw/images"
for class_folder in sorted(os.listdir(car_root)):
    folder_path = os.path.join(car_root, class_folder)
    if os.path.isdir(folder_path):
        imgs = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png", ".webp"))]
        for img_name in imgs[:2]: # 2 images per class = 40 images
            p = os.path.join(folder_path, img_name)
            img = Image.open(p).convert("RGB").resize((224, 224))
            arr = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img, dtype=np.float32)[None, ...])
            preds = model.predict(arr, verbose=0)[0]
            veh_conf = float(np.sum(preds[vehicle_indices]))
            car_scores.append((class_folder, img_name, veh_conf))

print(f"Evaluated {len(car_scores)} car images across 20 vehicle classes.")
scores_only = [s[2] for s in car_scores]
print(f"Min car confidence: {min(scores_only)*100:.2f}%")
print(f"Max car confidence: {max(scores_only)*100:.2f}%")
print(f"Mean car confidence: {np.mean(scores_only)*100:.2f}%")
print(f"Median car confidence: {np.median(scores_only)*100:.2f}%")
print(f"Cars with confidence >= 25%: {sum(1 for s in scores_only if s >= 0.25)} / {len(scores_only)} ({sum(1 for s in scores_only if s >= 0.25)/len(scores_only)*100:.1f}%)")
print(f"Cars with confidence >= 30%: {sum(1 for s in scores_only if s >= 0.30)} / {len(scores_only)} ({sum(1 for s in scores_only if s >= 0.30)/len(scores_only)*100:.1f}%)")
print(f"Cars with confidence >= 35%: {sum(1 for s in scores_only if s >= 0.35)} / {len(scores_only)} ({sum(1 for s in scores_only if s >= 0.35)/len(scores_only)*100:.1f}%)")
print(f"Cars with confidence >= 40%: {sum(1 for s in scores_only if s >= 0.40)} / {len(scores_only)} ({sum(1 for s in scores_only if s >= 0.40)/len(scores_only)*100:.1f}%)")

# List any below 40%
low_cars = [s for s in car_scores if s[2] < 0.40]
if low_cars:
    print(f"\nCars with score < 40% ({len(low_cars)} items):")
    for c, n, sc in low_cars:
        print(f"  {c} ({n}): {sc*100:.2f}%")

