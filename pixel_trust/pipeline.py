"""Per-image pipeline: depth prior -> warp -> K generated samples -> trust features + true error.

Everything a deployed system could compute is a *feature*. The true error against the real right
image is only used as the label for evaluation.
"""
from __future__ import annotations

import cv2
import numpy as np

from .depth import build_prior, corrupt_prior
from .warp import fill_holes_nearest, forward_warp, inpaint_mask

FEATURES = ["inv", "dist", "hits", "edge", "disp", "tex", "var", "ens_std", "lidar_rel"]
SOTA = ["var", "traj", "sem", "vggt"]   # published signals (see baselines.py)
GROUPS = {
    "warp support": ["inv", "dist", "hits", "edge", "disp", "tex"],
    "+ sample variance": ["inv", "dist", "hits", "edge", "disp", "tex", "var"],
    "+ depth reliability": FEATURES,
}


def _gray_blur(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img.astype(np.float32) / 255.0, (0, 0), 1.0)


def per_pixel_error(gen: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.abs(_gray_blur(gen) - _gray_blur(gt)).mean(-1)


def make_features(warp: dict, samples: list[np.ndarray], prior_t: np.ndarray, mask: np.ndarray) -> dict:
    holes = warp["holes"]
    disp_t = fill_holes_nearest(warp["disp"], holes)
    inv = mask.astype(np.float32)
    dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    gx = cv2.Sobel(disp_t, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(disp_t, cv2.CV_32F, 0, 1, ksize=3)
    S = np.stack([_gray_blur(s) for s in samples])           # K x H x W x 3
    mean_img = S.mean(0)
    g = cv2.cvtColor((mean_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    tex = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    feats = {
        "inv": inv,
        "dist": np.log1p(dist),
        "hits": np.log1p(warp["hits"]),
        "edge": np.log1p(np.hypot(gx, gy)),
        "disp": disp_t,
        "tex": tex,
        "ens_std": fill_holes_nearest(warp["w_ens_std"], holes) / (disp_t + 1.0),
        "lidar_rel": fill_holes_nearest(warp["w_lidar_res"], holes) / (disp_t + 1.0),
    }
    K = len(samples)
    for k in range(2, K + 1):  # variance with the first k samples (for the compute experiment)
        feats[f"var_k{k}"] = S[:k].std(0).mean(-1)
    feats["var"] = feats[f"var_k{K}"]
    return feats


def process(pair, preds_rel: list[np.ndarray], lidar: np.ndarray, generator, K: int, condition: str,
            regen_quantiles=(), seed_base: int = 1000, baselines: dict | None = None, log=print):
    """Run one image under one depth condition ('clean' or 'corrupt')."""
    P = build_prior(preds_rel, lidar)
    corrupt_src = np.zeros(pair.disp.shape, bool)
    prior = P["prior"]
    if condition == "corrupt":
        prior, corrupt_src = corrupt_prior(prior, seed=pair.idx)
        # the residual against LiDAR is recomputed from the corrupted prior (LiDAR still disagrees there)
        from .depth import dense_residual
        P = {**P, "lidar_res": dense_residual(prior, lidar)}
    warp = forward_warp(pair.left, prior, extras={"ens_std": P["ens_std"], "lidar_res": P["lidar_res"],
                                                   "corrupt": corrupt_src.astype(np.float32)})
    mask = inpaint_mask(warp["holes"])
    seeds = [seed_base + pair.idx * 10 + k for k in range(K)]
    samples, dt = generator(warp["img"], mask, seeds)
    feats = make_features(warp, samples, prior, mask)
    # v2: published baselines, each a per-pixel distrust map on the delivered view (sample 0)
    if getattr(generator, "last_traj", None) is not None:
        feats["traj"] = generator.last_traj
    for key, fn in (baselines or {}).items():
        try:
            feats[key] = fn(samples[0], pair.left)
        except Exception as e:
            log(f"baseline {key} failed on idx {pair.idx}: {type(e).__name__}: {e}")
    err = per_pixel_error(samples[0], pair.right)
    out = {"feats": feats, "err": err, "mask": mask, "gen_seconds": dt,
           "corrupt_t": warp["w_corrupt"] > 0.5, "warped": warp["img"], "sample0": samples[0]}

    # Thrust 2: regenerate the least reliable warped pixels instead of pasting them
    regen = {}
    for q in regen_quantiles:
        if q == "oracle":
            extra = out["corrupt_t"] & ~mask
        else:
            r = feats["lidar_rel"]
            extra = (r > np.quantile(r[~mask], float(q))) & ~mask
        m2 = inpaint_mask(mask | extra, dilate=1)
        s2, _ = generator(warp["img"] * (~m2[..., None]), m2, seeds[:1])
        regen[str(q)] = {"err": per_pixel_error(s2[0], pair.right), "mask": m2}
    out["regen"] = regen
    return out


def downsample(a: np.ndarray, f: int = 2) -> np.ndarray:
    return a[::f, ::f]
