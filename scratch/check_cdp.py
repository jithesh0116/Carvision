import subprocess
import time
import json
import urllib.request
import base64
import os
import socket

# Check if port 9222 is free
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
port_open = s.connect_ex(('127.0.0.1', 9222)) == 0
s.close()

edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
proc = subprocess.Popen([
    edge_exe,
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--remote-debugging-port=9222",
    "about:blank"
])

time.sleep(1.5)

try:
    with urllib.request.urlopen("http://127.0.0.1:9222/json") as response:
        pages = json.loads(response.read().decode())
    print("CDP pages found:", len(pages))
    ws_url = pages[0]["webSocketDebuggerUrl"]
    print("WebSocket URL:", ws_url)
finally:
    proc.terminate()
    proc.wait()
