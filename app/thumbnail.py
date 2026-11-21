"""Topology-preserving reduction of a bi-level drawing to a blocky thumbnail.

The input is a 2-D uint8 array whose pixels are only 0 (black/ink) or 255
(white/paper).  For an integer factor ``s`` in 2..6 the image is cut into
``s x s`` blocks starting at the top-left corner; a partial block on the right
or bottom edge is padded with white.  Each block becomes one output pixel:
black when the block holds at least ``threshold`` black pixels, white otherwise.
``threshold`` is shared by the whole image and may be anything from 1 up to the
number of pixels in a block (``s*s``).

A candidate ``(s, threshold)`` is accepted only when the block association is a
one-to-one correspondence between the connected regions of the original and of
the thumbnail, for both colours:

* black regions are 8-connected; each must associate with a distinct thumbnail
  black region and vice versa (separate strokes stay separate, touching strokes
  stay merged);
* white regions are 4-connected; enclosed white areas (holes, the inside of
  closed contours) must associate one-to-one, and the external white background
  must correspond to the thumbnail's external background -- no ink may bleed into
  the margin and no thumbnail black region may touch the frame.

"Associate through the block" is taken literally: a source pixel and its block
pair up their regions only when their colours agree, and a black source pixel in
a white block (or a white source pixel in a black block) contributes nothing.
The check therefore works purely on per-block associations; it never needs to
align individual pixels.

Thresholds are only tried where the rendering can change (``count + 1`` for every
block count), and the smallest accepted threshold of the largest accepted factor
is returned.  Nothing larger is possible, nothing smaller is as compact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .topology import BACKGROUND, Topology

BLACK = 0
WHITE = 255
MIN_FACTOR = 2
MAX_FACTOR = 6
MAX_SIDE = 256


class BiLevelError(ValueError):
    """Raised when the input is not a supported 0/255 image."""


@dataclass(frozen=True)
class BlockGrid:
    """Geometry of an ``s``-px block grid over an image."""

    factor: int
    source_width: int
    source_height: int
    columns: int
    rows: int

    @property
    def extra_columns(self) -> int:
        return self.columns * self.factor - self.source_width

    @property
    def extra_rows(self) -> int:
        return self.rows * self.factor - self.source_height

    @property
    def block_pixels(self) -> int:
        return self.factor * self.factor

    def source_block_index(self) -> np.ndarray:
        """Flat block id of every *source* pixel, shape (height, width)."""
        column = np.arange(self.source_width, dtype=np.int64) // self.factor
        row = np.arange(self.source_height, dtype=np.int64) // self.factor
        return (row[:, None] * self.columns + column[None, :]).astype(np.int64)


@dataclass(frozen=True)
class ThumbnailResult:
    """One accepted factor/threshold combination and its rendering."""

    grid: BlockGrid
    threshold: int
    counts: np.ndarray  # black pixel count per block, shape (rows, columns)
    pixels: np.ndarray  # uint8 thumbnail, 0/255

    @property
    def black_blocks(self) -> int:
        return int((self.pixels == BLACK).sum())


def validate_input(array: np.ndarray) -> np.ndarray:
    """Validate shape/dtype/size/pixel values and return the black mask."""
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise BiLevelError("缩略图输入必须是二维灰度图")
    if array.dtype != np.uint8:
        raise BiLevelError("缩略图输入必须是 8 位灰度")
    if array.size == 0:
        raise BiLevelError("缩略图输入不能为空图")
    if array.shape[0] > MAX_SIDE or array.shape[1] > MAX_SIDE:
        raise BiLevelError(f"缩略图输入每边至多 {MAX_SIDE} 像素")
    allowed = (array == BLACK) | (array == WHITE)
    if not bool(allowed.all()):
        bad = int(array.size - allowed.sum())
        raise BiLevelError(f"图像含 {bad} 个既非 0 也非 255 的像素；只接受二值黑白像素")
    return array == BLACK


def block_grid(width: int, height: int, factor: int) -> BlockGrid:
    if not MIN_FACTOR <= factor <= MAX_FACTOR:
        raise ValueError(f"缩小倍数必须在 {MIN_FACTOR}～{MAX_FACTOR} 之间")
    return BlockGrid(
        factor=factor,
        source_width=width,
        source_height=height,
        columns=(width + factor - 1) // factor,
        rows=(height + factor - 1) // factor,
    )


def black_counts(black: np.ndarray, grid: BlockGrid) -> np.ndarray:
    """Black pixel count per block; partial edge blocks are padded with white."""
    s = grid.factor
    padded = np.zeros((grid.rows * s, grid.columns * s), dtype=bool)
    padded[: grid.source_height, : grid.source_width] = black
    return padded.reshape(grid.rows, s, grid.columns, s).sum(axis=(1, 3), dtype=np.int32)


def render(counts: np.ndarray, threshold: int) -> np.ndarray:
    """Paint blocks with at least ``threshold`` black pixels black."""
    pixels = np.full(counts.shape, WHITE, dtype=np.uint8)
    pixels[counts >= threshold] = BLACK
    return pixels


def candidate_thresholds(counts: np.ndarray, block_pixels: int) -> list[int]:
    """Thresholds at which the rendering can change, in ascending order.

    A block flips from black to white exactly when ``threshold`` exceeds its
    pixel count, so only ``1`` and ``count + 1`` (still within a block) need to
    be tried.  Scanning them in ascending order means the first accepted value
    is also the smallest accepted one.
    """
    values = [1]
    for count in sorted({int(v) for v in counts.reshape(-1)}):
        if 1 <= count < block_pixels:
            values.append(count + 1)
    return values


def _pairs_are_bijective(
    source_ids: np.ndarray, target_ids: np.ndarray, n_source: int, n_target: int
) -> bool:
    """Every source id and every target id must occur in exactly one pair."""
    if source_ids.size == 0:
        return n_source == 0 and n_target == 0
    if n_target == 0:
        return False
    pairs = np.unique(source_ids.astype(np.int64) * n_target + target_ids.astype(np.int64))
    return (
        int(np.unique(pairs // n_target).size) == n_source
        and int(np.unique(pairs % n_target).size) == n_target
        and int(pairs.size) == n_source
        and n_source == n_target
    )


def _black_association_ok(
    source: Topology, target: Topology, block_index: np.ndarray
) -> bool:
    source_flat = source.black.reshape(-1)
    target_per_pixel = target.black.reshape(-1)[block_index].reshape(-1)
    keep = (source_flat >= 0) & (target_per_pixel >= 0)
    return _pairs_are_bijective(
        source_flat[keep], target_per_pixel[keep], source.n_black, target.n_black
    )


def _white_association_ok(
    source: Topology, target: Topology, block_index: np.ndarray
) -> bool:
    source_flat = source.white.reshape(-1)
    target_per_pixel = target.white.reshape(-1)[block_index].reshape(-1)

    source_has_background = bool((source.white == BACKGROUND).any())
    target_has_background = bool((target.white == BACKGROUND).any())
    if source_has_background != target_has_background:
        return False

    keep = (source_flat != -1) & (target_per_pixel != -1)
    # The external background is one more region on each side; give it the id
    # ``n`` so enclosed regions keep 0..n-1 and the arithmetic below stays simple.
    source_ids = np.where(source_flat == BACKGROUND, source.n_white, source_flat)[keep]
    target_ids = np.where(target_per_pixel == BACKGROUND, target.n_white, target_per_pixel)[keep]
    if not np.array_equal(source_ids == source.n_white, target_ids == target.n_white):
        return False  # a background pixel is inside an enclosed thumbnail lake or vice versa
    n_source = source.n_white + int(source_has_background)
    n_target = target.n_white + int(target_has_background)
    return _pairs_are_bijective(source_ids, target_ids, n_source, n_target)


def topology_preserved(
    source: Topology, target: Topology, block_index: np.ndarray
) -> bool:
    """True when both colours associate one-to-one between the two images."""
    return _black_association_ok(
        source, target, block_index
    ) and _white_association_ok(source, target, block_index)


def evaluate_rendering(
    counts: np.ndarray,
    threshold: int,
    grid: BlockGrid,
    block_index: np.ndarray,
    source: Topology,
) -> tuple[bool, np.ndarray]:
    """Render one threshold and test its topology against ``source``."""
    pixels = render(counts, threshold)
    target = Topology(pixels == BLACK)
    return topology_preserved(source, target, block_index), pixels


def reduce_image(
    array: np.ndarray,
) -> tuple[ThumbnailResult | None, list[tuple[int, str]]]:
    """Find the largest factor (and its smallest threshold) that preserves topology.

    Returns ``(result, rejected)``.  ``result`` is None when no factor in 2..6
    qualifies; the caller should then keep the original image.  ``rejected``
    lists every factor that failed and a short reason, largest first.
    """
    black = validate_input(array)
    height, width = black.shape
    source = Topology(black)
    rejected: list[tuple[int, str]] = []

    for factor in range(MAX_FACTOR, MIN_FACTOR - 1, -1):
        grid = block_grid(width, height, factor)
        counts = black_counts(black, grid)
        block_index = grid.source_block_index()
        for threshold in candidate_thresholds(counts, grid.block_pixels):
            ok, pixels = evaluate_rendering(counts, threshold, grid, block_index, source)
            if ok:
                return (
                    ThumbnailResult(
                        grid=grid,
                        threshold=threshold,
                        counts=counts,
                        pixels=pixels,
                    ),
                    rejected,
                )
        rejected.append(
            (
                factor,
                f"阈值 1～{grid.block_pixels} 全部未通过：缩略图与原图的黑色区域"
                "或白色区域（含外部白底）无法一一对应",
            )
        )

    return None, rejected
