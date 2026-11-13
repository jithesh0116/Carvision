import urllib.request
import json
import os

os.makedirs("tests/test_images/positives", exist_ok=True)
os.makedirs("tests/test_images/negatives", exist_ok=True)

def search_commons(query, limit=3):
    url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch={urllib.request.quote(query)}&gsrnamespace=6&gsrlimit={limit}&prop=imageinfo&iiprop=url|size|mime&format=json"
    req = urllib.request.Request(url, headers={"User-Agent": "CarVisionTest/1.0 (test@carvision.org)"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    pages = data.get("query", {}).get("pages", {})
    results = []
    for pid, pinfo in pages.items():
        ii = pinfo.get("imageinfo", [{}])[0]
        img_url = ii.get("url")
        mime = ii.get("mime", "")
        if img_url and ("jpeg" in mime or "png" in mime or "jpg" in mime):
            results.append((pinfo.get("title"), img_url))
    return results

print("Ready to fetch negative samples if needed.")

