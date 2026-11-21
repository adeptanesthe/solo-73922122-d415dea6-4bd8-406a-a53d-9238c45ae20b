"""Thumbnail orchestration: validate the upload, reduce it, describe the result."""

from __future__ import annotations

import base64

import numpy as np

from .imaging import encode_grayscale_png
from .models import (
    ThumbnailGridInfo,
    ThumbnailRejection,
    ThumbnailResponse,
    ThumbnailSourceInfo,
)
from .thumbnail import BLACK, MAX_FACTOR, ThumbnailResult, reduce_image, validate_input
from .topology import Topology

KEEP_ORIGINAL_MESSAGE = (
    "倍数 2～6 均无法在保持黑色区域、闭合白色区域与外部白底一一对应的情况下缩小；"
    "请保留原图，不做缩略。"
)


def _source_info(array: np.ndarray) -> ThumbnailSourceInfo:
    black = array == BLACK
    topology = Topology(black)
    return ThumbnailSourceInfo(
        width=int(array.shape[1]),
        height=int(array.shape[0]),
        black_pixels=int(black.sum()),
        white_pixels=int(array.size - black.sum()),
        black_regions=topology.n_black,
        enclosed_white_regions=topology.n_white,
    )


def _grid_info(result: ThumbnailResult) -> ThumbnailGridInfo:
    grid = result.grid
    target = Topology(result.pixels == BLACK)
    return ThumbnailGridInfo(
        factor=grid.factor,
        threshold=result.threshold,
        block_pixels=grid.block_pixels,
        columns=grid.columns,
        rows=grid.rows,
        padded_columns=grid.extra_columns,
        padded_rows=grid.extra_rows,
        output_width=grid.columns,
        output_height=grid.rows,
        black_blocks=result.black_blocks,
        black_regions=target.n_black,
        enclosed_white_regions=target.n_white,
        black_one_to_one=True,
        white_one_to_one=True,
    )


def make_thumbnail(array: np.ndarray) -> ThumbnailResponse:
    """Reduce a strictly bi-level image and build the API response.

    Raises :class:`~app.thumbnail.BiLevelError` when the array is not a
    supported 0/255 image; the route maps that to HTTP 400.
    """
    validate_input(array)  # reject bad pixels before any labelling work
    source_info = _source_info(array)
    result, rejected_factors = reduce_image(array)
    rejected = [
        ThumbnailRejection(factor=factor, detail=detail)
        for factor, detail in rejected_factors
    ]
    evaluated = list(range(MAX_FACTOR, 1, -1))

    if result is None:
        return ThumbnailResponse(
            status="keep_original",
            message=KEEP_ORIGINAL_MESSAGE,
            source=source_info,
            evaluated_factors=evaluated,
            rejected=rejected,
        )

    png = encode_grayscale_png(result.pixels)
    return ThumbnailResponse(
        status="reduced",
        message=(
            f"采用最大合格缩小倍数 {result.grid.factor}，共用阈值 {result.threshold}；"
            "黑色区域与白色区域（含外部白底）均与原图一一对应。"
        ),
        factor=result.grid.factor,
        threshold=result.threshold,
        source=source_info,
        thumbnail=_grid_info(result),
        evaluated_factors=evaluated,
        rejected=rejected,
        image_base64=base64.b64encode(png).decode("ascii"),
    )
