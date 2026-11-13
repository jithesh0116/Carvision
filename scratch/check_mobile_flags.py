import subprocess
import os

edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

flags_to_test = [
    ["--enable-viewport", "--window-size=320,600"],
    ["--enable-use-zoom-for-dsf", "--window-size=320,600"],
    ["--user-agent=Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1", "--window-size=320,600"],
]

for idx, flags in enumerate(flags_to_test):
    cmd = [
        edge_exe,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
    ] + flags + [
        "--dump-dom",
        "file:///" + os.path.abspath("scratch/test_dim.html").replace("\\", "/")
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    for line in res.stdout.splitlines():
        if "INNER_WIDTH:" in line and "document" not in line:
            print(f"Test {idx}: {flags[0][:30]} -> {line.strip()}")
