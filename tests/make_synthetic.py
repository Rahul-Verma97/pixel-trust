"""Write a tiny KITTI-2015-shaped dataset of layered synthetic scenes for CPU tests."""
import os
import sys

import cv2
import numpy as np


def scene(seed, H=375, W=1242, density=0.3):
    rng = np.random.default_rng(seed)
    tex = lambda: cv2.resize(rng.integers(0, 255, (H // 8, W // 8, 3)).astype(np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)
    bg = tex()
    yy = np.arange(H)[:, None].repeat(W, 1).astype(np.float32)
    disp = 5 + 40 * np.clip((yy - H * 0.35) / (H * 0.65), 0, 1)  # ground plane, nearer at the bottom
    L = bg.copy()
    layers = []
    for _ in range(rng.integers(3, 6)):
        x0, y0 = rng.integers(0, W - 200), rng.integers(H // 4, H - 80)
        w, h = rng.integers(60, 200), rng.integers(40, 120)
        d = float(rng.uniform(20, 70))
        layers.append((x0, y0, w, h, d, tex()))
    layers.sort(key=lambda t: t[4])
    R = np.zeros_like(L)
    xs = np.arange(W)
    for y in range(H):  # ground plane shift
        src = np.clip(np.round(xs + disp[y]).astype(int), 0, W - 1)
        R[y] = bg[y, src]
    for x0, y0, w, h, d, t in layers:
        L[y0:y0 + h, x0:x0 + w] = t[y0:y0 + h, x0:x0 + w]
        disp[y0:y0 + h, x0:x0 + w] = d
        xr0 = int(round(x0 - d))
        a, b = max(0, xr0), min(W, xr0 + w)
        if b > a:
            R[y0:y0 + h, a:b] = t[y0:y0 + h, x0 + (a - xr0):x0 + (b - xr0)]
    gt = np.where(rng.random((H, W)) < density, disp, 0)
    return L, R, gt


if __name__ == "__main__":
    root, n = sys.argv[1], int(sys.argv[2])
    for sub in ("image_2", "image_3", "disp_occ_0"):
        os.makedirs(os.path.join(root, "training", sub), exist_ok=True)
    for i in range(n):
        L, R, D = scene(i)
        cv2.imwrite(os.path.join(root, "training", "image_2", f"{i:06d}_10.png"), L)
        cv2.imwrite(os.path.join(root, "training", "image_3", f"{i:06d}_10.png"), R)
        cv2.imwrite(os.path.join(root, "training", "disp_occ_0", f"{i:06d}_10.png"), (D * 256).astype(np.uint16))
    print("wrote", n)
