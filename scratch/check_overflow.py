import os
import subprocess
import json

edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
html_path = os.path.abspath("scratch/rendered_home.html")
file_url = "file:///" + html_path.replace("\\", "/")

# We can inject a small inline test script into rendered_home.html or a test copy that logs scrollWidth vs clientWidth
test_html_path = os.path.abspath("scratch/rendered_test_overflow.html")
with open(html_path, "r", encoding="utf-8") as f:
    content = f.read()

# Add script at bottom of body
script_inject = """
<script>
window.addEventListener('load', () => {
  const sw = document.documentElement.scrollWidth;
  const cw = document.documentElement.clientWidth;
  const iw = window.innerWidth;
  const overflow = sw > cw;
  const result = { sw, cw, iw, overflow };
  console.log('LAYOUT_RESULT:' + JSON.stringify(result));
  const el = document.createElement('div');
  el.id = 'layout-result';
  el.setAttribute('data-result', JSON.stringify(result));
  document.body.appendChild(el);
});
</script>
"""
content = content.replace("</body>", script_inject + "\n</body>")
with open(test_html_path, "w", encoding="utf-8") as f:
    f.write(content)

test_url = "file:///" + test_html_path.replace("\\", "/")

widths = [1440, 1366, 1024, 768, 430, 375, 320]

print("Checking viewport overflow across target widths:")
for w in widths:
    h = 800
    cmd = [
        edge_exe,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        f"--window-size={w},{h}",
        "--dump-dom",
        test_url
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    dom = res.stdout
    if 'data-result="' in dom:
        part = dom.split('data-result="')[1].split('"')[0].replace('&quot;', '"')
        data = json.loads(part)
        sw = data['sw']
        cw = data['cw']
        overflow = data['overflow']
        status = "FAIL (Overflow detected!)" if overflow else "PASS (No horizontal scroll)"
        print(f"Viewport width {w}px: clientWidth={cw}px, scrollWidth={sw}px -> {status}")
    else:
        print(f"Viewport width {w}px: Result not extracted from DOM dump")

