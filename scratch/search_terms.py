import json

with open("ml/models/imagenet_class_index.json", "r", encoding="utf-8") as f:
    class_index = json.loads(f.read())

for k, (synset, name) in class_index.items():
    lower = name.lower()
    for term in ["bus", "truck", "van", "wagon", "wheel", "jeep", "suv", "sedan", "coupe", "motor", "taxi", "vehicle", "traffic"]:
        if term in lower:
            print(f"[{k}] {synset}: {name}")

