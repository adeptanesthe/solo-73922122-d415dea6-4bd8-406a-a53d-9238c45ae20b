"""Tests for the topology-preserving thumbnail endpoint.

The assertions are mostly *independent* of the implementation: a naive
pixel-by-pixel reference checker (``reference_ok``) re-derives the block
association rule straight from the specification, and the API's answer is
verified against it, including the choice of factor and threshold.
"""

from __future__ import annotations

import base64
import io
from collections import deque

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.thumbnail import BiLevelError, block_grid, validate_input
from app.topology import Topology
from tests.fixtures import png_bytes, png_bytes_16bit, png_bytes_palette, png_bytes_rgb

client = TestClient(app)


# ---------------------------------------------------------------------------
# Reference implementation, deliberately naive and independent
# ---------------------------------------------------------------------------


def _components(mask: np.ndarray, eight: bool) -> tuple[np.ndarray, int]:
    height, width = mask.shape
    labels = np.full((height, width), -1, dtype=np.int32)
    if eight:
        neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    count = 0
    for y in range(height):
        for x in range(width):
            if mask[y, x] and labels[y, x] < 0:
                queue = deque([(y, x)])
                labels[y, x] = count
                while queue:
                    cy, cx = queue.popleft()
                    for dy, dx in neighbours:
                        ny, nx = cy + dy, cx + dx
                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and mask[ny, nx]
                            and labels[ny, nx] < 0
                        ):
                            labels[ny, nx] = count
                            queue.append((ny, nx))
                count += 1
    return labels, count


def _white_regions(black: np.ndarray) -> tuple[np.ndarray, int, bool]:
    """White labels: -2 outside, -1 black, >=0 enclosed. Returns has_background."""
    labels, _ = _components(~black, eight=False)
    border = set(labels[0].tolist()) | set(labels[-1].tolist())
    border |= set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
    border.discard(-1)
    out = np.full(black.shape, -2, dtype=np.int32)
    mapping: dict[int, int] = {}
    for y in range(black.shape[0]):
        for x in range(black.shape[1]):
            label = int(labels[y, x])
            if label < 0:
                out[y, x] = -1
            elif label not in border:
                if label not in mapping:
                    mapping[label] = len(mapping)
                out[y, x] = mapping[label]
    return out, len(mapping), bool(np.any(out == -2))


def _reference_render(image: np.ndarray, factor: int, threshold: int) -> np.ndarray:
    height, width = image.shape
    rows = -(-height // factor)
    columns = -(-width // factor)
    out = np.full((rows, columns), 255, dtype=np.uint8)
    for by in range(rows):
        for bx in range(columns):
            count = 0
            for y in range(by * factor, min((by + 1) * factor, height)):
                for x in range(bx * factor, min((bx + 1) * factor, width)):
                    if image[y, x] == 0:
                        count += 1
            if count >= threshold:
                out[by, bx] = 0
    return out


def _reference_ok(image: np.ndarray, factor: int, threshold: int) -> bool:
    """Literal transcription of the spec's association rule."""
    black = image == 0
    source_black, n_source_black = _components(black, eight=True)
    source_white, n_source_white, source_bg = _white_regions(black)

    thumb = _reference_render(image, factor, threshold)
    target_black = thumb == 0
    target_black_labels, n_target_black = _components(target_black, eight=True)
    target_white, n_target_white, target_bg = _white_regions(target_black)

    if source_bg != target_bg:
        return False

    height, width = image.shape
    black_pairs = set()
    white_pairs = set()
    for y in range(height):
        for x in range(width):
            by, bx = y // factor, x // factor
            a = int(source_black[y, x])
            b = int(target_black_labels[by, bx])
            if a >= 0 and b >= 0:
                black_pairs.add((a, b))
            a = int(source_white[y, x])
            b = int(target_white[by, bx])
            if a == -1 or b == -1:
                continue
            if (a == -2) != (b == -2):
                return False
            white_pairs.add((a, b))

    def bijective(pairs: set, n_source: int, n_target: int, source_background: bool):
        source_ids = set(range(n_source))
        target_ids = set(range(n_target))
        if source_background:
            source_ids.add(-2)
            target_ids.add(-2)
        return (
            {a for a, _ in pairs} == source_ids
            and {b for _, b in pairs} == target_ids
            and len(pairs) == len(source_ids) == len(target_ids)
        )

    return bijective(
        black_pairs, n_source_black, n_target_black, False
    ) and bijective(white_pairs, n_source_white, n_target_white, source_bg)


def _reference_choice(image: np.ndarray) -> tuple[int, int] | None:
    """Largest factor first, smallest passing threshold within it."""
    for factor in (6, 5, 4, 3, 2):
        for threshold in range(1, factor * factor + 1):
            if _reference_ok(image, factor, threshold):
                return factor, threshold
    return None


def post_thumbnail(array: np.ndarray, content_type: str = "image/png", fmt=png_bytes):
    return client.post(
        "/thumbnail",
        files={"image": ("drawing.png", fmt(array), content_type)},
    )


def decode_base64_png(payload: str) -> np.ndarray:
    raw = base64.b64decode(payload)
    with Image.open(io.BytesIO(raw)) as im:
        assert im.format == "PNG"
        return np.asarray(im, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Fixtures: small drawings composed of strokes and closed outlines
# ---------------------------------------------------------------------------


def blank(size=(48, 48)) -> np.ndarray:
    return np.full((size[1], size[0]), 255, dtype=np.uint8)


def drawing() -> np.ndarray:
    """Three separate strokes, one closed ring and one solid square."""
    img = blank()
    img[4:40, 4:7] = 0  # vertical stroke
    img[4:40, 12:15] = 0  # parallel vertical stroke (kept separate)
    img[30:33, 20:40] = 0  # horizontal stroke
    img[6:24, 20:36] = 0  # ring outer
    img[10:20, 24:32] = 255  # ring inner white (closed contour)
    img[28:38, 40:46] = 0  # solid blob
    return img


def post_check(array: np.ndarray):
    r = post_thumbnail(array)
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# Endpoint contract
# ---------------------------------------------------------------------------


def test_docs_lists_thumbnail_route():
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "/thumbnail" in schema["paths"]


def test_reduced_png_is_strictly_bi_level_and_geometry_matches():
    image = drawing()
    body = post_check(image)
    assert body["status"] == "reduced"
    expected = _reference_choice(image)
    assert expected is not None
    assert (body["factor"], body["threshold"]) == expected

    thumb = decode_base64_png(body["image_base64"])
    factor = body["factor"]
    rows = -(-image.shape[0] // factor)
    columns = -(-image.shape[1] // factor)
    assert thumb.shape == (rows, columns)
    assert set(np.unique(thumb).tolist()) <= {0, 255}
    # Each thumbnail pixel is exactly the thresholded block count.
    reference = _reference_render(image, factor, body["threshold"])
    assert np.array_equal(thumb, reference)
    info = body["thumbnail"]
    assert info["columns"] == columns and info["rows"] == rows
    assert info["output_width"] == columns and info["output_height"] == rows
    assert info["padded_columns"] == columns * factor - image.shape[1]
    assert info["padded_rows"] == rows * factor - image.shape[0]


def test_reported_regions_match_independent_labelling():
    image = drawing()
    body = post_check(image)
    source = body["source"]
    black = image == 0
    _black_labels, n_black = _components(black, eight=True)
    _white_labels, n_white, _bg = _white_regions(black)
    assert source["black_regions"] == n_black
    assert source["enclosed_white_regions"] == n_white
    assert source["black_pixels"] == int(black.sum())
    assert source["white_pixels"] == int(image.size - black.sum())

    thumb = decode_base64_png(body["image_base64"])
    _target_black, target_n_black = _components(thumb == 0, eight=True)
    _target_white, target_n_white, _target_bg = _white_regions(thumb == 0)
    assert body["thumbnail"]["black_regions"] == target_n_black
    assert body["thumbnail"]["enclosed_white_regions"] == target_n_white
    assert body["thumbnail"]["black_blocks"] == int((thumb == 0).sum())


def test_threshold_within_block_size():
    image = drawing()
    body = post_check(image)
    assert 1 <= body["threshold"] <= body["factor"] ** 2
    assert body["thumbnail"]["block_pixels"] == body["factor"] ** 2


def test_threshold_is_the_smallest_qualifying_one():
    image = drawing()
    body = post_check(image)
    factor, threshold = body["factor"], body["threshold"]
    assert _reference_ok(image, factor, threshold)
    assert not any(_reference_ok(image, factor, t) for t in range(1, threshold))


def test_result_agrees_with_reference_on_random_drawings():
    rng = np.random.default_rng(2024)
    for _ in range(8):
        height = int(rng.integers(8, 40))
        width = int(rng.integers(8, 40))
        image = np.where(rng.random((height, width)) < 0.5, 0, 255).astype(np.uint8)
        body = post_check(image)
        expected = _reference_choice(image)
        if expected is None:
            assert body["status"] == "keep_original", (height, width)
        else:
            assert body["status"] == "reduced"
            assert (body["factor"], body["threshold"]) == expected


def test_largest_qualifying_factor_wins():
    # A single black pixel must stay a single black pixel; a 2x2 block keeps it
    # but 6x6 would drop it at almost any threshold, so the answer is not 6.
    image = blank()
    image[10, 10] = 0
    body = post_check(image)
    assert body["status"] == "reduced"
    assert body["factor"] == 6  # threshold 1 keeps the lone black pixel
    assert body["threshold"] == 1

    # A drawing whose block grid cannot keep two neighbouring thin strokes apart
    # at factor 6 but can at a smaller factor.
    image = blank()
    image[2:46, 2:4] = 0
    image[2:46, 5:7] = 0  # separated by one white column
    body = post_check(image)
    if body["status"] == "reduced":
        assert _reference_ok(image, body["factor"], body["threshold"])
        assert not any(
            _reference_ok(image, bigger, t)
            for bigger in range(body["factor"] + 1, 7)
            for t in range(1, bigger * bigger + 1)
        )


def test_keep_original_when_nothing_qualifies():
    # Dense random noise has thousands of tiny regions; every block would merge
    # them, so no factor in 2..6 can preserve the region count.
    rng = np.random.default_rng(7)
    image = np.where(rng.random((64, 64)) < 0.5, 0, 255).astype(np.uint8)
    body = post_check(image)
    assert body["status"] == "keep_original"
    assert body["factor"] is None and body["threshold"] is None
    assert body["image_base64"] is None
    assert body["thumbnail"] is None
    assert "保留原图" in body["message"]
    assert body["evaluated_factors"] == [6, 5, 4, 3, 2]
    assert [rej["factor"] for rej in body["rejected"]] == [6, 5, 4, 3, 2]


def test_keep_original_message_is_the_only_hint():
    image = np.tile(np.array([[0, 255]], dtype=np.uint8), (3, 3))
    body = post_check(image)
    assert body["status"] in ("reduced", "keep_original")
    if body["status"] == "keep_original":
        assert set(body) >= {"status", "message", "source"}
        assert body["image_base64"] is None


def test_closed_contour_stays_closed():
    image = blank()
    image[10:40, 10:40] = 0
    image[16:34, 16:34] = 255
    body = post_check(image)
    assert body["status"] == "reduced"
    thumb = decode_base64_png(body["image_base64"])
    _labels, n_black = _components(thumb == 0, eight=True)
    _labels, n_white, _bg = _white_regions(thumb == 0)
    assert n_black == 1 and n_white == 1


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rejects_non_bi_level_pixels():
    image = drawing()
    image[0, 0] = 128
    r = post_thumbnail(image)
    assert r.status_code == 400
    assert "0 也非 255" in r.json()["detail"]


def test_rejects_more_than_256_pixels_per_side():
    image = np.full((257, 10), 255, dtype=np.uint8)
    r = post_thumbnail(image)
    assert r.status_code == 400
    assert "256" in r.json()["detail"]

    assert post_thumbnail(np.full((256, 256), 255, dtype=np.uint8)).status_code == 200


def test_rejects_non_png_and_bad_formats():
    image = drawing()
    assert post_thumbnail(image, content_type="image/jpeg").status_code == 400
    r = client.post(
        "/thumbnail", files={"image": ("x.png", b"not a png", "image/png")}
    )
    assert r.status_code == 400
    r = post_thumbnail(image, fmt=png_bytes_rgb)
    assert r.status_code == 400
    r = post_thumbnail(image, fmt=png_bytes_16bit)
    assert r.status_code == 400
    r = post_thumbnail(image, fmt=png_bytes_palette)
    assert r.status_code == 400


def test_requires_the_image_field():
    assert client.post("/thumbnail").status_code == 422


# ---------------------------------------------------------------------------
# Module-level units
# ---------------------------------------------------------------------------


def test_validate_input_contract():
    with pytest.raises(BiLevelError):
        validate_input(np.zeros((4, 4), dtype=np.float64))
    with pytest.raises(BiLevelError):
        validate_input(np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(BiLevelError):
        validate_input(np.full((1, 1), 7, dtype=np.uint8))
    with pytest.raises(BiLevelError):
        validate_input(np.zeros((0, 4), dtype=np.uint8))
    assert validate_input(np.array([[0, 255]], dtype=np.uint8)).tolist() == [[True, False]]


def test_block_grid_pads_remainder_blocks():
    grid = block_grid(10, 7, 4)
    assert (grid.columns, grid.rows) == (3, 2)
    assert (grid.extra_columns, grid.extra_rows) == (2, 1)
    assert grid.source_block_index().shape == (7, 10)
    assert grid.source_block_index()[0, 10 - 1] == 2


def test_topology_counts_use_different_connectivities():
    # A checkerboard corner: 8-connected black is one region, 4-connected white
    # background is still one region, and there is no enclosed hole.
    image = np.array(
        [[0, 255, 0], [255, 0, 255], [0, 255, 0]], dtype=np.uint8
    )
    topology = Topology(image == 0)
    assert topology.n_black == 1
    assert topology.n_white == 0
