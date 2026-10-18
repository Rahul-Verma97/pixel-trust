"""Middlebury stereo (2014 + 2021) as a second, out-of-domain dataset.

Indoor, high-resolution scenes with dense, accurate ground-truth disparity, so it differs from
KITTI in content (indoor objects vs streets), camera, baseline and depth range.

* 2014: the 23 scenes with public ground truth ("perfect" rectification), ~3000 x 2000 px.
* 2021: the 24 scenes captured with a mobile rig, 1920 x 1080 px.

Every scene gives two pairs: left -> right as captured, and right -> left by mirroring both images
(the mirrored right image becomes the source; its ground truth is disp1). Frames are resized to
height 368 (disparity scaled with the image) and centre-cropped to a width divisible by 8.

Splits are made by *scene group* (Sword1/Sword2, chess1/2/3, ... share a group), so no scene
appears in two splits; groups are assigned deterministically for a 40 / 20 / 40 train / calib / test share.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .data import Pair

BASE = "https://vision.middlebury.edu/stereo/data"
SCENES_2014 = ["Adirondack", "Backpack", "Bicycle1", "Cable", "Classroom1", "Couch", "Flowers", "Jadeplant",
               "Mask", "Motorcycle", "Piano", "Pipes", "Playroom", "Playtable", "Recycle", "Shelves",
               "Shopvac", "Sticks", "Storage", "Sword1", "Sword2", "Umbrella", "Vintage"]
SCENES_2021 = ["artroom1", "artroom2", "bandsaw1", "bandsaw2", "chess1", "chess2", "chess3", "curule1",
               "curule2", "curule3", "ladder1", "ladder2", "octogons1", "octogons2", "pendulum1", "pendulum2",
               "podium1", "skates1", "skates2", "skiboots1", "skiboots2", "skiboots3", "traproom1", "traproom2"]
OUT_H = 368
MAX_W = 1216
LIDAR_FRAC = 0.05   # dense GT -> keep 5% of pixels, a density comparable to 10% of KITTI's LiDAR ground truth


FILES = ("im0.png", "im1.png", "disp0.pfm", "disp1.pfm", "calib.txt")


def scene_url(name: str) -> str:
    """2014: one zip per scene. 2021: no per-scene zip; files are served from data/<scene>/."""
    if name in SCENES_2014:
        return f"{BASE}/scenes2014/zip/{name}-perfect.zip"
    return f"{BASE}/scenes2021/data/{name}"


def read_pfm(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.readline().decode().strip()
        assert header in ("Pf", "PF"), header
        w, h = map(int, f.readline().decode().split())
        scale = float(f.readline().decode().strip())
        data = np.fromfile(f, "<f4" if scale < 0 else ">f4")
    ch = 3 if header == "PF" else 1
    a = data.reshape(h, w, ch)[..., 0] if ch == 3 else data.reshape(h, w)
    return np.flipud(a).astype(np.float32)


def group_of(name: str) -> str:
    return re.sub(r"\d+$", "", name).lower()


def scene_splits(names: list[str], target=(("train", 0.4), ("calib", 0.2), ("test", 0.4))) -> dict[str, str]:
    """Deterministic group-wise split: largest groups first, each to the split furthest below its target share."""
    size: dict[str, int] = {}
    for n in names:
        size[group_of(n)] = size.get(group_of(n), 0) + 1
    have = {s: 0 for s, _ in target}
    gsplit = {}
    for g in sorted(size, key=lambda g: (-size[g], g)):
        s = max(target, key=lambda t: (t[1] * len(names) - have[t[0]], -[x for x, _ in target].index(t[0])))[0]
        gsplit[g] = s
        have[s] += size[g]
    return {n: gsplit[group_of(n)] for n in names}


def _resize_pair(L, R, D):
    h, w = L.shape[:2]
    s = OUT_H / h
    W = int(round(w * s))
    size = (W, OUT_H)
    L = cv2.resize(L, size, interpolation=cv2.INTER_AREA)
    R = cv2.resize(R, size, interpolation=cv2.INTER_AREA)
    D = np.where(np.isfinite(D) & (D > 0), D, 0).astype(np.float32)
    D = cv2.resize(D, size, interpolation=cv2.INTER_NEAREST) * s
    cw = min(MAX_W, (W // 8) * 8)
    x0 = (W - cw) // 2
    return L[:, x0:x0 + cw], R[:, x0:x0 + cw], D[:, x0:x0 + cw]


def _fetch(name: str, raw: str, log=print) -> str | None:
    zpath = os.path.join(raw, os.path.basename(scene_url(name)))
    sdir = os.path.join(raw, name)
    if glob.glob(os.path.join(sdir, "**", "im0.png"), recursive=True):
        return sdir
    try:
        if name not in SCENES_2014:   # 2021: fetch the individual files
            os.makedirs(sdir, exist_ok=True)
            for fn in FILES:
                dst = os.path.join(sdir, fn)
                if not os.path.exists(dst):
                    r = subprocess.run(["wget", "-q", "-O", dst + ".part", f"{scene_url(name)}/{fn}"])
                    if r.returncode != 0:
                        if os.path.exists(dst + ".part"):
                            os.remove(dst + ".part")
                        if fn in ("disp1.pfm", "calib.txt"):
                            continue
                        raise RuntimeError(f"download failed: {fn} (wget exit {r.returncode})")
                    os.replace(dst + ".part", dst)
            return sdir
        if not os.path.exists(zpath):
            subprocess.run(["wget", "-q", "-O", zpath + ".part", scene_url(name)], check=True)
            os.replace(zpath + ".part", zpath)
        with zipfile.ZipFile(zpath) as z:
            keep = [m for m in z.namelist() if os.path.basename(m) in ("im0.png", "im1.png", "disp0.pfm", "disp1.pfm", "calib.txt")]
            z.extractall(sdir, members=keep)
        os.remove(zpath)
        return sdir
    except Exception as e:
        log(f"middlebury: {name} FAILED ({type(e).__name__}: {e})")
        return None


def prepare(dest: str, scenes: list[str] | None = None, log=print, workers: int = 6) -> str:
    """Download, resize and cache every pair as <dest>/pairs/<idx>.npz; returns dest."""
    scenes = scenes or (SCENES_2014 + SCENES_2021)
    index_path = os.path.join(dest, "index.json")
    if os.path.exists(index_path):
        return dest
    raw = os.path.join(dest, "raw")
    os.makedirs(raw, exist_ok=True)
    os.makedirs(os.path.join(dest, "pairs"), exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:
        dirs = dict(zip(scenes, ex.map(lambda n: _fetch(n, raw, log), scenes)))
    ok = sorted([n for n, d in dirs.items() if d], key=str.lower)
    splits = scene_splits(ok)
    index = {}
    for k, name in enumerate(ok):
        files = {os.path.basename(p): p for p in glob.glob(os.path.join(dirs[name], "**", "*"), recursive=True)}
        im0 = cv2.cvtColor(cv2.imread(files["im0.png"]), cv2.COLOR_BGR2RGB)
        im1 = cv2.cvtColor(cv2.imread(files["im1.png"]), cv2.COLOR_BGR2RGB)
        directions = [(0, im0, im1, read_pfm(files["disp0.pfm"]))]
        if "disp1.pfm" in files:   # mirrored: right image becomes the source
            directions.append((1, im1[:, ::-1].copy(), im0[:, ::-1].copy(), read_pfm(files["disp1.pfm"])[:, ::-1].copy()))
        for dr, L, R, D in directions:
            L, R, D = _resize_pair(L, R, D)
            idx = 2 * k + dr
            np.savez_compressed(os.path.join(dest, "pairs", f"{idx:06d}.npz"), left=L, right=R, disp=D)
            index[idx] = {"scene": name, "mirrored": bool(dr), "split": splits[name], "shape": list(D.shape),
                          "max_disp": float(D.max()), "gt_density": float((D > 0).mean())}
        log(f"middlebury: {name:12s} {splits[name]:5s} pairs={len(directions)} -> {index[2 * k]['shape']}")
    json.dump(index, open(index_path, "w"), indent=1)
    return dest


class MiddleburyDataset:
    name = "middlebury"
    lidar_frac = LIDAR_FRAC

    def __init__(self, root: str):
        self.root = root
        self.index = {int(k): v for k, v in json.load(open(os.path.join(root, "index.json"))).items()}

    def indices(self) -> list[int]:
        return sorted(self.index)

    def split_of(self, idx: int) -> str:
        return self.index[idx]["split"]

    def load_pair(self, idx: int) -> Pair:
        z = np.load(os.path.join(self.root, "pairs", f"{idx:06d}.npz"))
        return Pair(idx, z["left"], z["right"], z["disp"].astype(np.float32))

    def describe(self, idx: int) -> str:
        e = self.index[idx]
        return f"{e['scene']}{' (mirrored)' if e['mirrored'] else ''}"
