import os
import subprocess

edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

test_html = """<!doctype html>
<html>
<head><meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body>
<div id="dim"></div>
<script>
document.getElementById('dim').innerText = 'INNER_WIDTH:' + window.innerWidth + ' INNER_HEIGHT:' + window.innerHeight;
</script>
</body>
</html>
"""

with open("scratch/test_dim.html", "w", encoding="utf-8") as f:
    f.write(test_html)

for w in [1440, 768, 430, 320]:
    cmd = [
        edge_exe,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        f"--window-size={w},800",
        "--dump-dom",
        "file:///" + os.path.abspath("scratch/test_dim.html").replace("\\", "/")
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    for line in res.stdout.splitlines():
        if "INNER_WIDTH:" in line:
            print(f"Arg width {w} -> {line.strip()}")
