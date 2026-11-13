import json

with open("ml/models/imagenet_class_index.json", "r", encoding="utf-8") as f:
    class_index = json.loads(f.read())

vehicle_keywords = [
    "car", "automobile", "wagon", "cab", "convertible", "jeep", "limousine",
    "minivan", "pickup", "racer", "sports_car", "police_van", "ambulance",
    "fire_engine", "truck", "trailer", "tow_truck", "garbage_truck",
    "recreational_vehicle", "minibus", "grille", "car_wheel"
]

vehicle_classes = []
for k, (synset, name) in class_index.items():
    lower_name = name.lower()
    # Check if any vehicle keyword is part of name
    matched = False
    for kw in vehicle_keywords:
        if kw in lower_name:
            vehicle_classes.append((int(k), synset, name))
            matched = True
            break

print(f"Found {len(vehicle_classes)} vehicle candidate classes:")
for idx, synset, name in sorted(vehicle_classes):
    print(f"  [{idx:3d}] {synset}: {name}")

