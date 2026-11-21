"""In-process demo for the thumbnail endpoint (no Docker required).

    uvicorn app.main:app --port 8000 &
    python scripts/thumbnail_client.py --base-url http://localhost:8000

Posts the same 256x256 drawing the demo generator writes, prints the decision
(maximum factor, threshold, region counts), and saves the returned PNG next to
the request so it can be inspected.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_thumbnail_demo import demo_drawing, png_bytes  # noqa: E402


def post_png(base_url: str, data: bytes) -> dict:
    boundary = "----thumbnaildemo"
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
            'filename="thumbnail.png"\r\nContent-Type: image/png\r\n\r\n'
        ).encode()
        + data
        + b"\r\n"
        + f"--{boundary}--\r\n".encode()
    )
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/thumbnail",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--outdir", default="demo")
    args = parser.parse_args()

    image = demo_drawing()
    source_png = png_bytes(image)
    try:
        payload = post_png(args.base_url, source_png)
    except Exception as exc:  # noqa: BLE001 - demo tool
        print(f"request failed: {exc}", file=sys.stderr)
        return 2

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "thumbnail.request.png").write_bytes(source_png)
    if payload.get("image_base64"):
        (outdir / "thumbnail.result.png").write_bytes(
            base64.b64decode(payload["image_base64"])
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("status") in ("reduced", "keep_original") else 1


if __name__ == "__main__":
    raise SystemExit(main())
