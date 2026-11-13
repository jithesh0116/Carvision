import asyncio
import json
import base64
import os
import subprocess
import shutil
import urllib.request
import websockets
from PIL import Image

edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
html_path = os.path.abspath("scratch/rendered_home.html")
file_url = "file:///" + html_path.replace("\\", "/")

viewports = [
    ("viewport_1440_desktop", 1440, 900, False),
    ("viewport_1366_laptop", 1366, 768, False),
    ("viewport_1024_small_laptop", 1024, 768, False),
    ("viewport_768_tablet", 768, 1024, True),
    ("viewport_430_mobile", 430, 932, True),
    ("viewport_375_small_mobile", 375, 667, True),
    ("viewport_320_narrow_mobile", 320, 568, True),
]

out_dir = os.path.abspath("scratch/screenshots")
os.makedirs(out_dir, exist_ok=True)
profile_dir = os.path.abspath("scratch/edge_cdp_profile")
shutil.rmtree(profile_dir, ignore_errors=True)
os.makedirs(profile_dir, exist_ok=True)

async def capture_all():
    port = 9228
    cmd = [
        edge_exe,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        f"--user-data-dir={profile_dir}",
        f"--remote-debugging-port={port}",
        "about:blank"
    ]
    proc = subprocess.Popen(cmd)
    
    # Wait for debug port to be ready
    for _ in range(10):
        await asyncio.sleep(0.5)
        try:
            req = urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version")
            data = json.loads(req.read().decode())
            ws_url = data.get("webSocketDebuggerUrl")
            if ws_url:
                break
        except Exception:
            continue
    else:
        proc.terminate()
        raise RuntimeError("Failed to connect to CDP debug port")

    print("Connected to CDP:", ws_url)
    
    try:
        # Get existing page target
        target_req = urllib.request.urlopen(f"http://127.0.0.1:{port}/json")
        target_data = json.loads(target_req.read().decode())
        page_ws = target_data[0].get("webSocketDebuggerUrl")
        
        async with websockets.connect(page_ws, max_size=20*1024*1024) as ws:
            msg_id = 1
            
            async def call(method, params=None):
                nonlocal msg_id
                mid = msg_id
                msg_id += 1
                payload = {"id": mid, "method": method, "params": params or {}}
                await ws.send(json.dumps(payload))
                while True:
                    resp = json.loads(await ws.recv())
                    if resp.get("id") == mid:
                        return resp.get("result", {})
            
            await call("Page.enable")
            
            for name, w, h, is_mobile in viewports:
                print(f"Configuring viewport {w}x{h} (mobile={is_mobile})...")
                await call("Emulation.setDeviceMetricsOverride", {
                    "width": w,
                    "height": h,
                    "deviceScaleFactor": 1,
                    "mobile": is_mobile,
                })
                
                await call("Page.navigate", {"url": file_url})
                await asyncio.sleep(0.6)
                
                # Verify exact viewport dimensions inside JS
                eval_res = await call("Runtime.evaluate", {
                    "expression": "JSON.stringify({innerWidth: window.innerWidth, scrollWidth: document.documentElement.scrollWidth, clientWidth: document.documentElement.clientWidth})"
                })
                val = eval_res.get("result", {}).get("value", "{}")
                print(f" -> Result: {val}")
                
                # Screenshot
                shot = await call("Page.captureScreenshot", {"format": "png"})
                img_data = base64.b64decode(shot["data"])
                out_path = os.path.join(out_dir, f"{name}.png")
                with open(out_path, "wb") as f:
                    f.write(img_data)
                
                im = Image.open(out_path)
                print(f" -> Saved {name}: {im.size} ({len(img_data)} bytes)")
                
    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    asyncio.run(capture_all())
