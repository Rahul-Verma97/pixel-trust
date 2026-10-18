"""Write a tiny Middlebury-shaped raw dataset (im0/im1/disp0.pfm/disp1.pfm per scene) for CPU tests."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from make_synthetic import scene  # noqa: E402


def write_pfm(path, a):
    with open(path, "wb") as f:
        f.write(b"Pf\n%d %d\n-1.0\n" % (a.shape[1], a.shape[0]))
        np.flipud(a).astype("<f4").tofile(f)


def right_disp(d):
    h, w = d.shape
    out = np.full((h, w), np.inf, np.float32)
    ys, xs = np.mgrid[0:h, 0:w]
    xt = np.round(xs - d).astype(int)
    ok = (d > 0) & (xt >= 0)
    for y, x, v in sorted(zip(ys[ok], xt[ok], d[ok]), key=lambda t: t[2]):
        out[y, x] = v
    return out


if __name__ == "__main__":
    raw, names = sys.argv[1], sys.argv[2].split(",")
    for i, n in enumerate(names):
        L, R, D = scene(100 + i, H=736, W=1180, density=1.0)
        dense = D.copy()
        sdir = os.path.join(raw, n, f"{n}-perfect")
        os.makedirs(sdir, exist_ok=True)
        cv2.imwrite(os.path.join(sdir, "im0.png"), L)
        cv2.imwrite(os.path.join(sdir, "im1.png"), R)
        write_pfm(os.path.join(sdir, "disp0.pfm"), np.where(dense > 0, dense, np.inf).astype(np.float32))
        write_pfm(os.path.join(sdir, "disp1.pfm"), right_disp(dense))
    print("wrote", len(names))
