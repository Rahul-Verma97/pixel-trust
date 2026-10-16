"""KITTI Stereo 2015 loading.

Layout expected (official `data_scene_flow.zip`):
    <root>/training/image_2/000000_10.png   left camera
    <root>/training/image_3/000000_10.png   right camera
    <root>/training/disp_occ_0/000000_10.png  left disparity (uint16 / 256, 0 = invalid)
"""
from __future__ import annotations

import glob
import os
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np

KITTI_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip"
CROP_H, CROP_W = 368, 1216  # multiples of 8 that fit every KITTI 2015 frame


@dataclass
class Pair:
    idx: int
    left: np.ndarray    # HxWx3 uint8 RGB
    right: np.ndarray   # HxWx3 uint8 RGB
    disp: np.ndarray    # HxW float32, 0 = invalid (LiDAR-based ground truth)


def find_kitti_root(candidates: list[str]) -> str | None:
    """Return a directory containing training/image_2, searching a few levels deep."""
    for base in candidates:
        if not base or not os.path.isdir(base):
            continue
        for d in [base] + glob.glob(os.path.join(base, "*")) + glob.glob(os.path.join(base, "*", "*")) \
                + glob.glob(os.path.join(base, "*", "*", "*")):
            if os.path.isdir(os.path.join(d, "training", "image_2")) and \
               os.path.isdir(os.path.join(d, "training", "disp_occ_0")):
                return d
    return None


def download_kitti(dest: str) -> str:
    """Download and extract only the files we need (~1.7 GB zip)."""
    os.makedirs(dest, exist_ok=True)
    zpath = os.path.join(dest, "data_scene_flow.zip")
    if not os.path.exists(zpath):
        subprocess.run(["wget", "-q", "-O", zpath, KITTI_URL], check=True)
    subprocess.run(["unzip", "-q", "-o", zpath,
                    "training/image_2/*_10.png", "training/image_3/*_10.png",
                    "training/disp_occ_0/*_10.png", "-d", dest], check=True)
    return dest


def _crop(a: np.ndarray) -> np.ndarray:
    h, w = a.shape[:2]
    y0 = h - CROP_H              # drop sky rows at the top
    x0 = (w - CROP_W) // 2
    return a[y0:y0 + CROP_H, x0:x0 + CROP_W]


def load_pair(root: str, idx: int) -> Pair:
    name = f"{idx:06d}_10.png"
    L = cv2.cvtColor(cv2.imread(os.path.join(root, "training", "image_2", name)), cv2.COLOR_BGR2RGB)
    R = cv2.cvtColor(cv2.imread(os.path.join(root, "training", "image_3", name)), cv2.COLOR_BGR2RGB)
    D = cv2.imread(os.path.join(root, "training", "disp_occ_0", name), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
    return Pair(idx, _crop(L), _crop(R), _crop(D))


def list_indices(root: str) -> list[int]:
    files = sorted(glob.glob(os.path.join(root, "training", "disp_occ_0", "*_10.png")))
    return [int(os.path.basename(f)[:6]) for f in files]


def lidar_subset(disp: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Keep a random fraction of valid ground-truth points as the vehicle's sparse LiDAR."""
    rng = np.random.default_rng(seed)
    keep = (disp > 0) & (rng.random(disp.shape) < frac)
    return np.where(keep, disp, 0).astype(np.float32)
