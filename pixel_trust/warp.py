"""Forward-warp the left (source) image into the right (target) camera with a z-buffer.

For a rectified stereo pair a left pixel (y, x) with disparity d lands at (y, x - d) in the right image.
Pixels of the target that receive no source pixel are disocclusions or outside the source view:
the generator has to invent them.
"""
from __future__ import annotations

import cv2
import numpy as np


def forward_warp(img: np.ndarray, disp: np.ndarray, extras: dict[str, np.ndarray] | None = None):
    h, w = disp.shape
    ys, xs = np.mgrid[0:h, 0:w]
    xt = xs - disp
    out_img = np.zeros_like(img)
    out_disp = np.zeros((h, w), np.float32)
    filled = np.zeros((h, w), bool)
    out_extra = {k: np.zeros((h, w), np.float32) for k in (extras or {})}
    hits = np.zeros(h * w, np.int32)

    # nearest splat for the conflict count (how many source pixels compete for a target pixel)
    xn = np.round(xt).astype(int)
    ok = (xn >= 0) & (xn < w) & (disp > 0)
    np.add.at(hits, (ys[ok] * w + xn[ok]), 1)

    # splat to floor and ceil to avoid 1-px cracks; z-buffer: larger disparity (closer) wins
    for off in (np.floor(xt), np.ceil(xt)):
        xi = off.astype(int)
        valid = (xi >= 0) & (xi < w) & (disp > 0)
        lin = ys[valid] * w + xi[valid]
        d = disp[valid]
        order = np.argsort(d, kind="stable")
        lin_o = lin[order]
        # keep, for every target index, the source with maximal disparity
        rev = lin_o[::-1]
        _, first = np.unique(rev, return_index=True)
        pick = (len(rev) - 1 - first)
        tgt = lin_o[pick]
        src_valid_idx = np.flatnonzero(valid.ravel())[order][pick]
        cur = out_disp.ravel()
        better = d[order][pick] > cur[tgt]
        tgt, src = tgt[better], src_valid_idx[better]
        out_img.reshape(-1, img.shape[2])[tgt] = img.reshape(-1, img.shape[2])[src]
        cur[tgt] = disp.ravel()[src]
        filled.ravel()[tgt] = True
        for k, v in (extras or {}).items():
            out_extra[k].ravel()[tgt] = v.ravel()[src]

    holes = ~filled
    return {"img": out_img, "disp": out_disp, "holes": holes,
            "hits": hits.reshape(h, w).astype(np.float32), **{f"w_{k}": v for k, v in out_extra.items()}}


def inpaint_mask(holes: np.ndarray, dilate: int = 2) -> np.ndarray:
    k = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
    return cv2.dilate(holes.astype(np.uint8), k) > 0


def fill_holes_nearest(a: np.ndarray, holes: np.ndarray, background_min: bool = True) -> np.ndarray:
    """Fill hole values from the nearest filled pixel on the same row (background side)."""
    out = a.copy()
    h, w = a.shape
    for y in range(h):
        row, hr = out[y], holes[y]
        if hr.all():
            continue
        idx = np.where(~hr, np.arange(w), -1)
        left = np.maximum.accumulate(idx)
        idx2 = np.where(~hr, np.arange(w), w)
        right = np.minimum.accumulate(idx2[::-1])[::-1]
        lv = np.where(left >= 0, row[np.clip(left, 0, w - 1)], np.inf)
        rv = np.where(right < w, row[np.clip(right, 0, w - 1)], np.inf)
        fill = np.minimum(lv, rv) if background_min else np.maximum(np.where(np.isinf(lv), -np.inf, lv),
                                                                     np.where(np.isinf(rv), -np.inf, rv))
        row[hr] = fill[hr]
    out[~np.isfinite(out)] = 0
    return out
