"""Depth prior for the target view, aligned to sparse LiDAR, plus depth-reliability signals.

The vehicle setting of View Translation gives the generator a depth prior. Here the prior is a
monocular estimate (Depth Anything V2) aligned in disparity space to a sparse LiDAR subset,
which is how a real vehicle would combine a camera network with a sparse range sensor.

Two reliability signals are produced per source pixel:
  * ens_std    - disagreement between two depth models and a flipped pass (epistemic proxy)
  * lidar_res  - |prior - LiDAR| at LiDAR returns, interpolated to a dense map
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.interpolate import NearestNDInterpolator


def fit_scale_shift(rel: np.ndarray, lidar: np.ndarray, iters: int = 3) -> tuple[float, float]:
    """Robust least squares  lidar ≈ a*rel + b  on LiDAR pixels (trimmed refits)."""
    m = lidar > 0
    x, y = rel[m].astype(np.float64), lidar[m].astype(np.float64)
    keep = np.ones_like(x, dtype=bool)
    a, b = 1.0, 0.0
    for _ in range(iters):
        A = np.stack([x[keep], np.ones(keep.sum())], 1)
        a, b = np.linalg.lstsq(A, y[keep], rcond=None)[0]
        r = np.abs(a * x + b - y)
        keep = r <= np.quantile(r, 0.9)
    return float(a), float(b)


def dense_residual(prior: np.ndarray, lidar: np.ndarray) -> np.ndarray:
    """Interpolate |prior - lidar| from LiDAR returns to every pixel, then smooth."""
    m = lidar > 0
    h, w = prior.shape
    ys, xs = np.nonzero(m)
    if len(ys) < 10:
        return np.zeros_like(prior)
    res = np.abs(prior[m] - lidar[m])
    f = NearestNDInterpolator(np.stack([ys, xs], 1), res)
    gy, gx = np.mgrid[0:h, 0:w]
    dense = f(gy, gx).astype(np.float32)
    return cv2.GaussianBlur(dense, (0, 0), 5)


class DepthAnything:
    """Two Depth Anything V2 sizes + horizontal-flip test-time augmentation (relative disparity)."""

    def __init__(self, device: str = "cuda", models=("depth-anything/Depth-Anything-V2-Small-hf",
                                                      "depth-anything/Depth-Anything-V2-Base-hf")):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.torch = torch
        self.device = device
        self.nets = []
        for mid in models:
            proc = AutoImageProcessor.from_pretrained(mid)
            net = AutoModelForDepthEstimation.from_pretrained(mid).to(device).eval()
            self.nets.append((proc, net))

    def _run(self, proc, net, img: np.ndarray) -> np.ndarray:
        torch = self.torch
        inp = proc(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = net(**inp).predicted_depth  # relative inverse depth
        out = torch.nn.functional.interpolate(out[:, None], size=img.shape[:2], mode="bicubic",
                                              align_corners=False)[0, 0]
        return out.float().cpu().numpy()

    def predict(self, img: np.ndarray) -> list[np.ndarray]:
        preds = []
        for proc, net in self.nets:
            preds.append(self._run(proc, net, img))
            preds.append(self._run(proc, net, img[:, ::-1].copy())[:, ::-1])
        return preds


class OracleNoisyDepth:
    """CPU stand-in for tests: densified ground truth with smooth noise, as several 'models'."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def predict_from_gt(self, disp: np.ndarray) -> list[np.ndarray]:
        m = disp > 0
        ys, xs = np.nonzero(m)
        f = NearestNDInterpolator(np.stack([ys, xs], 1), disp[m])
        gy, gx = np.mgrid[0:disp.shape[0], 0:disp.shape[1]]
        dense = f(gy, gx).astype(np.float32)
        outs = []
        for _ in range(4):
            noise = cv2.GaussianBlur(self.rng.normal(0, 1, disp.shape).astype(np.float32), (0, 0), 15)
            outs.append(0.5 * dense * (1 + 0.6 * noise) + 3.0)   # unknown scale/shift, like a mono model
        return outs


def build_prior(preds_rel: list[np.ndarray], lidar: np.ndarray) -> dict[str, np.ndarray]:
    """Align each relative prediction to LiDAR; prior = mean, ens_std = spread, lidar_res = residual."""
    aligned = []
    for p in preds_rel:
        a, b = fit_scale_shift(p, lidar)
        aligned.append(np.clip(a * p + b, 0.0, 400.0))
    aligned = np.stack(aligned)
    prior = aligned.mean(0).astype(np.float32)
    ens_std = aligned.std(0).astype(np.float32)
    return {"prior": prior, "ens_std": ens_std, "lidar_res": dense_residual(prior, lidar)}


def corrupt_prior(prior: np.ndarray, seed: int, n_blobs=(2, 4), area=(0.04, 0.09)) -> tuple[np.ndarray, np.ndarray]:
    """Simulate off-road depth failures (vegetation, dust, thin structures): a few smooth blobs
    whose disparity is scaled by 0.55-0.8 or 1.25-1.6. Returns corrupted prior and blob mask."""
    rng = np.random.default_rng(seed)
    h, w = prior.shape
    mask = np.zeros((h, w), np.float32)
    for _ in range(rng.integers(n_blobs[0], n_blobs[1] + 1)):
        cy, cx = rng.integers(h // 4, h), rng.integers(0, w)
        ar = rng.uniform(*area) * h * w
        ry = np.sqrt(ar / np.pi / 2.5)
        cv2.ellipse(mask, (int(cx), int(cy)), (int(ry * 2.5), int(ry)), float(rng.uniform(-20, 20)), 0, 360, 1.0, -1)
    soft = cv2.GaussianBlur(mask, (0, 0), 6)
    factor = rng.choice([rng.uniform(0.55, 0.8), rng.uniform(1.25, 1.6)])
    out = prior * (1 + (factor - 1) * soft)
    return out.astype(np.float32), (soft > 0.5)
