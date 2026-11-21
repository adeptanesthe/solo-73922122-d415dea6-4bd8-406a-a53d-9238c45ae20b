"""Generate a demo bi-level drawing and a ready-to-send thumbnail request.

Writes ``demo/thumbnail.png``: a 256x256 bi-level (0/255) PNG with separate
strokes, parallel lines, a closed ring and a solid block -- the features the
thumbnail service must keep apart.  Also writes ``demo/thumbnail.sh`` and
``demo/expected_fields.json``.

Usage:  python scripts/make_thumbnail_demo.py [outdir]
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def demo_drawing() -> np.ndarray:
    """A deterministic 256x256 mechanical drawing in pure black and white."""
    img = np.full((256, 256), 255, dtype=np.uint8)

    # Border frame, drawn as four separate strokes with a gap at the top-right
    # so the sheet itself is not a closed region.
    img[8:11, 8:248] = 0
    img[245:248, 8:248] = 0
    img[8:248, 8:11] = 0
    img[8:248, 245:248] = 0

    # Parallel thin strokes that must stay separate.
    for y in (40, 52, 64):
        img[y : y + 3, 24:120] = 0

    # Closed contour with white inside (a hole that must survive).
    img[100:180, 30:110] = 0
    img[112:168, 42:98] = 255
    img[130:150, 60:80] = 0  # boss inside the hole

    # Solid block.
    img[190:230, 150:235] = 0

    # Thin outline circle (closed contour), stamped by distance from a centre.
    yy, xx = np.mgrid[0:256, 0:256]
    ring = (np.abs(np.hypot(xx - 180, yy - 90) - 40) <= 1.2)
    img[ring] = 0

    return img


def png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def main(argv: list[str]) -> int:
    outdir = Path(argv[1] if len(argv) > 1 else "demo")
    outdir.mkdir(parents=True, exist_ok=True)

    image = demo_drawing()
    (outdir / "thumbnail.png").write_bytes(png_bytes(image))
    cmd = (
        "curl -sS -X POST http://localhost:8000/thumbnail \\\n"
        f"  -F 'image=@{outdir / 'thumbnail.png'};type=image/png'\n"
    )
    (outdir / "thumbnail.sh").write_text(cmd, encoding="utf-8")
    (outdir / "expected_fields.json").write_text(
        json.dumps(
            {
                "status": "reduced",
                "note": "factor/threshold 由服务计算；运行 thumbnail.sh 可查看实际值",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"wrote {outdir}/thumbnail.png ({image.shape[1]}x{image.shape[0]}, bi-level)")
    print(f"wrote {outdir}/thumbnail.sh")
    print("start the service with:  docker compose up --build")
    print(f"then run:                bash {outdir}/thumbnail.sh")
    print("interactive docs:        http://localhost:8000/docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
