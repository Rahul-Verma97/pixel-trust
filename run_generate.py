"""Stage 1 (GPU): generate views and trust features for every KITTI 2015 pair.

    python run_generate.py --root /kaggle/working/kitti --out /kaggle/working/out --shard 0 --nshards 2
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np

from pixel_trust.data import KittiDataset, lidar_subset
from pixel_trust.pipeline import downsample, process


def make_dataset(name: str, root: str):
    if name == "middlebury":
        from pixel_trust.middlebury import MiddleburyDataset
        return MiddleburyDataset(root)
    return KittiDataset(root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--gen", default="sd", choices=["sd", "genwarp", "telea"])
    ap.add_argument("--depth", default="dav2", choices=["dav2", "oracle"])
    ap.add_argument("--dataset", default="kitti", choices=["kitti", "middlebury"])
    ap.add_argument("--lidar-frac", type=float, default=None, help="default: 0.1 KITTI, 0.05 Middlebury")
    ap.add_argument("--seg-model", default=None, help="semantic-entropy segmenter (default: Cityscapes for KITTI, ADE20K otherwise)")
    ap.add_argument("--regen", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-png", type=int, default=8)
    ap.add_argument("--baselines", default="real", choices=["real", "stub", "none"])
    a = ap.parse_args()

    ds = make_dataset(a.dataset, a.root)
    lidar_frac = a.lidar_frac if a.lidar_frac is not None else ds.lidar_frac
    seg_model = a.seg_model or ("nvidia/segformer-b2-finetuned-cityscapes-1024-1024" if a.dataset == "kitti"
                                else "nvidia/segformer-b2-finetuned-ade-512-512")
    idxs = ds.indices()
    if a.limit:
        idxs = idxs[:a.limit]
    idxs = idxs[a.shard::a.nshards]
    os.makedirs(os.path.join(a.out, "feats"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "png"), exist_ok=True)

    if a.gen == "genwarp":
        from pixel_trust.genwarp_gen import GenWarpGen
        gen = GenWarpGen(steps=a.steps)
    elif a.gen == "sd":
        from pixel_trust.generate import SDInpaint
        gen = SDInpaint(steps=a.steps)
    else:
        from pixel_trust.generate import TeleaInpaint
        gen = TeleaInpaint()
    if a.depth == "dav2":
        from pixel_trust.depth import DepthAnything
        dmodel = DepthAnything()
    else:
        from pixel_trust.depth import OracleNoisyDepth
        dmodel = OracleNoisyDepth()

    from pixel_trust.baselines import load_baselines
    baselines = load_baselines(a.baselines, seg_model=seg_model, log=lambda m: print(f"[shard {a.shard}] {m}", flush=True))
    log = open(os.path.join(a.out, f"log_shard{a.shard}.jsonl"), "a")
    regen_q = [q for q in a.regen.split(",") if q]
    png_left = a.save_png
    for n, idx in enumerate(idxs):
        dst_clean = os.path.join(a.out, "feats", f"{idx:06d}_clean.npz")
        dst_corr = os.path.join(a.out, "feats", f"{idx:06d}_corrupt.npz")
        if os.path.exists(dst_clean) and os.path.exists(dst_corr):
            continue
        t0 = time.time()
        pair = ds.load_pair(idx)
        lidar = lidar_subset(pair.disp, lidar_frac, seed=idx)
        preds = dmodel.predict(pair.left) if a.depth == "dav2" else dmodel.predict_from_gt(pair.disp)
        split = ds.split_of(idx)
        for cond, dst in (("clean", dst_clean), ("corrupt", dst_corr)):
            rq = regen_q if (cond == "corrupt" and split == "test") else []
            r = process(pair, preds, lidar, gen, a.k, cond, regen_quantiles=rq, baselines=baselines,
                        log=lambda m: print(f"[shard {a.shard}] {m}", flush=True))
            gt_valid = downsample(pair.disp > 0)
            payload = {f"f_{k}": downsample(v).astype(np.float16) for k, v in r["feats"].items()}
            payload.update(err=downsample(r["err"]).astype(np.float16), mask=downsample(r["mask"]),
                           corrupt=downsample(r["corrupt_t"]), gt_valid=gt_valid,
                           gen_seconds=np.float32(r["gen_seconds"]), split=split,
                           lidar_density=np.float32((lidar > 0).mean()), dataset=ds.name)
            for q, v in r["regen"].items():
                payload[f"regen_err_{q}"] = downsample(v["err"]).astype(np.float16)
                payload[f"regen_mask_{q}"] = downsample(v["mask"])
            np.savez_compressed(dst, **payload)
            if split == "test" and png_left > 0:
                base = os.path.join(a.out, "png", f"{idx:06d}_{cond}")
                cv2.imwrite(base + "_warped.jpg", cv2.cvtColor(r["warped"], cv2.COLOR_RGB2BGR))
                cv2.imwrite(base + "_gen.jpg", cv2.cvtColor(r["sample0"], cv2.COLOR_RGB2BGR))
                if cond == "clean":
                    cv2.imwrite(base.replace("_clean", "") + "_left.jpg", cv2.cvtColor(pair.left, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(base.replace("_clean", "") + "_right.jpg", cv2.cvtColor(pair.right, cv2.COLOR_RGB2BGR))
        if split == "test":
            png_left -= 1
        log.write(json.dumps({"idx": idx, "scene": ds.describe(idx), "dataset": ds.name, "split": split, "seconds": round(time.time() - t0, 2),
                              "model": getattr(gen, "model_id", "?")}) + "\n")
        log.flush()
        print(f"[shard {a.shard}] {n + 1}/{len(idxs)} idx={idx} {split} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
