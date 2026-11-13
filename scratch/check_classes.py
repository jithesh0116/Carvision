import glob
import re

classes = set()
for p in glob.glob("templates/*.html"):
    with open(p, "r", encoding="utf-8") as f:
        content = f.read()
        for match in re.findall(r'class="([^"]+)"', content):
            for c in match.split():
                if not ("{{" in c or "{%" in c):
                    classes.add(c)

print("Found", len(classes), "unique classes across all templates:")
for c in sorted(classes):
    print(c)

