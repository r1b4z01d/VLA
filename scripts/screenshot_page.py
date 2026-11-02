#!/usr/bin/env python3
"""Screenshot a web page with headless Chrome over the DevTools protocol, after optional JS steps.

The docs' GUI screenshots (docs/img/*.png) are made with this, 1920x1080, so they stay comparable:

  ./run.sh scripts/screenshot_page.py OUT.png URL [--size 1920x1080] [--wait 3] [--gl] [--js "expr" --wait 2 ...]

  # the manager's tabs deep-link with #<view>; three.js pages (the 3D view) need --gl (software WebGL):
  ./run.sh scripts/screenshot_page.py docs/img/manager-services.png http://192.168.11.119:8081/#services --wait 5
  ./run.sh scripts/screenshot_page.py docs/img/route-editor-3d.png http://192.168.11.80:8092/3d --gl --wait 20
  # a click before the shot (here: select a dataset so its episode table shows):
  ./run.sh scripts/screenshot_page.py docs/img/manager-datasets.png http://127.0.0.1:8081/#datasets       --js "[...document.querySelectorAll('#dataset-list *')].find(e=>!e.childElementCount&&e.textContent.trim()=='pickup_tigger').click()" --wait 4

Needs the websockets package (in the venv via uvicorn[standard]) and Chrome: CHROME=/path/to/chrome overrides the
macOS default. Only one instance at a time (fixed DevTools port).

Each --js runs Runtime.evaluate (awaited if it returns a promise), each --wait sleeps that many
seconds; they run in the order given, then the screenshot is taken.
"""
import asyncio, base64, json, subprocess, sys, time, urllib.request

import websockets

import os
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
GL = []   # --gl: software WebGL (SwiftShader) for three.js pages; headless + --disable-gpu has no GL at all
PORT = 9333


async def main(argv):
    out, url = argv[0], argv[1]
    steps, size, scale = [], (1600, 1000), 1
    i = 2
    while i < len(argv):
        a = argv[i]
        if a == "--size":
            size = tuple(int(x) for x in argv[i + 1].split("x")); i += 2
        elif a == "--scale":
            scale = float(argv[i + 1]); i += 2
        elif a == "--gl":
            GL.append(True); i += 1
        elif a in ("--js", "--wait"):
            steps.append((a, argv[i + 1])); i += 2
        else:
            raise SystemExit(f"unknown arg {a}")
    gl = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] if GL else ["--disable-gpu"]
    proc = subprocess.Popen([CHROME, "--headless=new", *gl, "--hide-scrollbars", "--no-first-run",
                             f"--remote-debugging-port={PORT}", f"--window-size={size[0]},{size[1]}", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                page = next(t for t in targets if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        else:
            raise SystemExit("chrome did not come up")
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=2**27) as ws:
            n = 0

            async def call(method, **params):
                nonlocal n
                n += 1
                await ws.send(json.dumps({"id": n, "method": method, "params": params}))
                while True:
                    m = json.loads(await ws.recv())
                    if m.get("id") == n:
                        if "error" in m:
                            raise RuntimeError(m["error"])
                        return m.get("result", {})

            await call("Page.enable"); await call("Runtime.enable")
            await call("Emulation.setDeviceMetricsOverride", width=size[0], height=size[1], deviceScaleFactor=scale, mobile=False)
            await call("Page.navigate", url=url)
            await asyncio.sleep(2.5)
            for kind, val in steps:
                if kind == "--wait":
                    await asyncio.sleep(float(val))
                else:
                    r = await call("Runtime.evaluate", expression=val, awaitPromise=True, returnByValue=True)
                    if r.get("exceptionDetails"):
                        print("js error:", r["exceptionDetails"].get("text"), file=sys.stderr)
            shot = await call("Page.captureScreenshot", format="png")
            open(out, "wb").write(base64.b64decode(shot["data"]))
            print(out)
    finally:
        proc.terminate()
        try:
            proc.wait(5)
        except Exception:
            proc.kill()


asyncio.run(main(sys.argv[1:]))
