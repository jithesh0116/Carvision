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

VEHICLE_CLASSES = {
    "sports_car", "convertible", "minivan", "pickup", "limousine",
    "cab", "jeep", "beach_wagon", "racer", "police_van", "ambulance",
    "moving_van", "tow_truck", "trailer_truck", "fire_engine",
    "garbage_truck", "recreational_vehicle", "minibus", "school_bus",
    "grille", "car_wheel"
}

vehicle_indices = [int(k) for k, (synset, name) in class_index.items() if name in VEHICLE_CLASSES]
print(f"Tracking {len(vehicle_indices)} vehicle indices: {sorted(vehicle_indices)}")

def evaluate_image(img_path):
    try:
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img, dtype=np.float32)[None, ...])
        preds = model.predict(arr, verbose=0)[0]
        
        # Vehicle confidence: sum of probabilities across vehicle categories
        veh_conf = float(np.sum(preds[vehicle_indices]))
        
        # Top 3 overall classes
        top_idx = np.argsort(preds)[::-1][:3]
        top_classes = [(class_index[str(i)][1], float(preds[i])) for i in top_idx]
        
        # Top vehicle class
        veh_preds = [(class_index[str(i)][1], float(preds[i])) for i in vehicle_indices]
        veh_preds.sort(key=lambda x: x[1], reverse=True)
        top_veh = veh_preds[0]
        
        return {
            "success": True,
            "vehicle_confidence": veh_conf,
            "top_classes": top_classes,
            "top_vehicle_class": top_veh
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

print("\n--- POSITIVE SAMPLES (Cars) ---")
pos_dir = "tests/test_images/positives"
for f in sorted(os.listdir(pos_dir)):
    p = os.path.join(pos_dir, f)
    res = evaluate_image(p)
    if res["success"]:
        print(f"{f:35s} | Vehicle Conf: {res['vehicle_confidence']*100:6.2f}% | Top class: {res['top_classes'][0][0]} ({res['top_classes'][0][1]*100:.1f}%)")
    else:
        print(f"{f:35s} | ERROR: {res['error']}")

print("\n--- NEGATIVE SAMPLES (Non-Cars) ---")
neg_dir = "tests/test_images/negatives"
for f in sorted(os.listdir(neg_dir)):
    p = os.path.join(neg_dir, f)
    if f.endswith((".jpg", ".jpeg", ".png", ".webp")):
        res = evaluate_image(p)
        if res["success"]:
            print(f"{f:35s} | Vehicle Conf: {res['vehicle_confidence']*100:6.2f}% | Top class: {res['top_classes'][0][0]} ({res['top_classes'][0][1]*100:.1f}%)")
        else:
            print(f"{f:35s} | ERROR: {res['error']}")

