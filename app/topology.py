"""Connected-component labelling for the bi-level thumbnail check.

Black pixels use 8-connectivity, white pixels use 4-connectivity, and every
white pixel connected (through white pixels) to the image frame belongs to the
single external background region.  The two conventions differ on purpose:
8-connected foreground joined with 4-connected background keeps the digital
Jordan property, which is what lets "separate strokes" and "closed contours"
survive the comparison meaningfully.

The labeler is run-based: each row is split into runs of set pixels, runs of
neighbouring rows are merged with a union-find, and the labels are written back
into a dense array.  A pure Python loop over pixels would be too slow for a
256x256 input when dozens of candidate thresholds are evaluated; the run-based
version keeps per-row work proportional to the number of runs.

Label conventions used by :class:`Topology`:

* ``black``: ``>= 0`` component id on black pixels, ``-1`` on white pixels.
* ``white``: ``>= 0`` id for white regions *enclosed* by black, ``-1`` on black
  pixels, and :data:`BACKGROUND` (``-2``) for white connected to the frame.
"""

from __future__ import annotations

import numpy as np

BACKGROUND = -2


class _DisjointSet:
    """Tiny union-find with path halving and union by rank."""

    __slots__ = ("_parent", "_rank")

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, item: int) -> int:
        parent = self._parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        rank = self._rank
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1


def _row_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (row, start, end) for every horizontal run of True pixels."""
    height, width = mask.shape
    padded = np.zeros((height, width + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    delta = np.diff(padded, axis=1)
    rows, cols = np.nonzero(delta)
    values = delta[rows, cols]
    run_rows = rows[values == 1].astype(np.int32)
    starts = cols[values == 1].astype(np.int32)
    ends = cols[values == -1].astype(np.int32)
    return run_rows, starts, ends


def label_components(
    mask: np.ndarray, *, eight_connected: bool
) -> tuple[np.ndarray, int]:
    """Label the True pixels of ``mask`` and return ``(labels, count)``.

    ``labels`` is an ``int32`` array shaped like the input, with ``-1`` on False
    pixels.  Component ids are consecutive integers starting at 0.
    """
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    rows, starts, ends = _row_runs(mask)
    n_runs = int(starts.size)
    if n_runs == 0:
        return np.full((height, width), -1, dtype=np.int32), 0

    dsu = _DisjointSet(n_runs)
    # First run id of every row, plus a sentinel at the end.
    row_offsets = np.searchsorted(rows, np.arange(height + 1)).astype(np.int32)
    starts_list = starts.tolist()
    ends_list = ends.tolist()

    for y in range(1, height):
        prev_begin, prev_end = int(row_offsets[y - 1]), int(row_offsets[y])
        cur_begin, cur_end = int(row_offsets[y]), int(row_offsets[y + 1])
        if prev_begin == prev_end or cur_begin == cur_end:
            continue
        j = prev_begin
        for i in range(cur_begin, cur_end):
            start, end = starts_list[i], ends_list[i]
            if eight_connected:
                # Runs touching even at a single point are 8-adjacent.
                while j < prev_end and ends_list[j] < start:
                    j += 1
                k = j
                while k < prev_end and starts_list[k] <= end:
                    dsu.union(i, k)
                    k += 1
            else:
                # Runs must share at least one column to be 4-adjacent.
                while j < prev_end and ends_list[j] <= start:
                    j += 1
                k = j
                while k < prev_end and starts_list[k] < end:
                    dsu.union(i, k)
                    k += 1

    roots = np.fromiter(
        (dsu.find(run) for run in range(n_runs)), dtype=np.int32, count=n_runs
    )
    _unique, inverse = np.unique(roots, return_inverse=True)
    component = inverse.astype(np.int32)
    count = int(component.max()) + 1

    labels = np.full((height, width), -1, dtype=np.int32)
    for i in range(n_runs):
        labels[rows[i], starts_list[i] : ends_list[i]] = component[i]
    return labels, count


def _white_regions(black_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected white regions, with frame-touching white merged to BACKGROUND."""
    labels, count = label_components(~black_mask, eight_connected=False)
    if count == 0:
        return np.full(black_mask.shape, -1, dtype=np.int32), 0

    border = np.concatenate(
        (labels[0], labels[-1], labels[:, 0], labels[:, -1])
    )
    border_ids = np.unique(border[border >= 0])
    enclosed = np.ones(count, dtype=bool)
    enclosed[border_ids] = False
    n_enclosed = int(enclosed.sum())

    lookup = np.full(count, BACKGROUND, dtype=np.int32)
    lookup[enclosed] = np.arange(n_enclosed, dtype=np.int32)
    # ``labels`` is -1 exactly on black pixels; clamp only to keep indexing safe.
    out = np.where(labels >= 0, lookup[np.maximum(labels, 0)], -1).astype(np.int32)
    return out, n_enclosed


class Topology:
    """Connected regions of one bi-level image, ready for block association."""

    __slots__ = ("black", "n_black", "white", "n_white")

    def __init__(self, black_mask: np.ndarray) -> None:
        black_mask = np.asarray(black_mask, dtype=bool)
        self.black, self.n_black = label_components(black_mask, eight_connected=True)
        self.white, self.n_white = _white_regions(black_mask)

    def as_tuple(self) -> tuple[np.ndarray, int, np.ndarray, int]:
        return self.black, self.n_black, self.white, self.n_white
