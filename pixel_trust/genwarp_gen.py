"""GenWarp (Seo et al., NeurIPS 2024) as a second, real geometry-guided view-synthesis generator.

Unlike the inpainting stand-in, GenWarp regenerates the *whole* target view from the source image,
its depth and the relative camera pose, so warped pixels can also be altered or hallucinated.

GenWarp works on 512x512 squares. A KITTI frame (368 x 1216) is covered by four overlapping
368x368 crops (x0 = 0, 283, 565, 848). Each target column is taken from the right-most crop that
contains it, which keeps it away from that crop's right edge (where content leaves the source crop).
Stereo geometry: the target camera is the source camera translated by the baseline b along +x, so a
point with disparity d moves d pixels left; depth = f * b / d. Any (f, b) gives the same pixel shift.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import types

import cv2
import numpy as np

CROP = 368
X0S = (0, 283, 565, 848)
F_PX, BASELINE = 721.5, 0.54
GW_FILES = {
    ("Sony/genwarp", "multi1"): ["config.json", "denoising_unet.pth", "pose_guider.pth", "reference_unet.pth"],
    ("stabilityai/sd-vae-ft-mse", ""): ["config.json", "diffusion_pytorch_model.safetensors"],
    ("lambdalabs/sd-image-variations-diffusers", "image_encoder"): ["config.json", "pytorch_model.bin"],
}


def softmax_splat(frame, flow, metric, eps=1e-6):
    """Pure-torch forward softmax splatting with bilinear weights (same contract as pesser/splatting),
    so no CUDA extension has to be compiled."""
    import torch
    B, C, H, W = frame.shape
    dev, dt = frame.device, torch.float32
    frame, flow, metric = frame.to(dt), flow.to(dt), metric.to(dt)
    w = torch.exp(metric.clamp(max=60))
    gy, gx = torch.meshgrid(torch.arange(H, device=dev, dtype=dt), torch.arange(W, device=dev, dtype=dt), indexing="ij")
    x = gx[None] + flow[:, 0]
    y = gy[None] + flow[:, 1]
    x0, y0 = torch.floor(x), torch.floor(y)
    out = torch.zeros(B, C + 1, H * W, device=dev, dtype=dt)
    src = torch.cat([frame * w, w], 1).reshape(B, C + 1, H * W)
    for dx in (0, 1):
        for dy in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            wb = (1 - (x - xi).abs()) * (1 - (y - yi).abs())
            ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H) & (wb > 0)
            idx = (yi.clamp(0, H - 1) * W + xi.clamp(0, W - 1)).long().reshape(B, 1, -1).expand(B, C + 1, -1)
            out.scatter_add_(2, idx, src * (wb * ok).reshape(B, 1, -1))
    out = out.reshape(B, C + 1, H, W)
    return out[:, :C] / (out[:, C:] + eps)


def _splatting_function(kind, frame, flow, metric, eps=1e-6):
    assert kind == "softmax"
    return softmax_splat(frame, flow, metric, eps).to(frame.dtype)


def column_owner(width: int = 1216) -> np.ndarray:
    """Index of the crop that supplies each output column (right-most crop containing it)."""
    own = np.zeros(width, int)
    for k, x0 in enumerate(X0S):
        own[x0:x0 + CROP] = k
    return own


class GenWarpGen:
    model_id = "Sony/genwarp (multi1)"
    needs_source = True

    def __init__(self, device="cuda", steps=20, repo_dir="/kaggle/working/genwarp", ckpt_dir="/kaggle/working/gw_ckpt",
                 traj_steps=10):
        import torch
        from huggingface_hub import hf_hub_download
        self.torch, self.device, self.steps, self.traj_steps = torch, device, steps, traj_steps
        if not os.path.isdir(repo_dir):
            subprocess.run(["git", "clone", "-q", "https://github.com/sony/genwarp", repo_dir], check=True)
        sys.modules.setdefault("splatting", types.SimpleNamespace(splatting_function=_splatting_function))
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        for (repo, sub), files in GW_FILES.items():
            dst = {"Sony/genwarp": "multi1", "stabilityai/sd-vae-ft-mse": "sd-vae-ft-mse"}.get(repo, "image_encoder")
            os.makedirs(os.path.join(ckpt_dir, dst), exist_ok=True)
            for fn in files:
                p = hf_hub_download(repo, f"{sub}/{fn}" if sub else fn)
                link = os.path.join(ckpt_dir, dst, fn)
                if not os.path.exists(link):
                    os.symlink(p, link)
        from genwarp import GenWarp
        from genwarp import ops
        ops.splatting_function = _splatting_function
        self.ops = ops
        self.gw = GenWarp(cfg=dict(pretrained_model_path=ckpt_dir, checkpoint_name="multi1",
                                   half_precision_weights=True, num_inference_steps=steps), device=device)
        # record the predicted clean latent at every DDIM step (for trajectory variance)
        sched = self.gw.scheduler
        orig = sched.step
        self._x0, self._rec = [], False

        def step(*args, **kwargs):
            rd = kwargs.pop("return_dict", True)
            out = orig(*args, return_dict=True, **kwargs)
            if self._rec:
                self._x0.append(out.pred_original_sample.detach().float())
            return out if rd else (out.prev_sample, out.pred_original_sample)

        sched.step = step
        fovy = torch.tensor([2 * np.arctan((CROP / 2) / F_PX)])
        self.proj = ops.get_projection_matrix(fovy, 1.0, 0.01, 5000.).to(device, torch.float16)
        rel = torch.eye(4)[None]
        rel[0, 0, 3] = -BASELINE
        self.rel = rel.to(device, torch.float16)
        self.last_traj = None

    def _crop_call(self, src: np.ndarray, disp: np.ndarray, seeds: list[int]):
        torch = self.torch
        K = len(seeds)
        img = torch.from_numpy(src.astype(np.float32) / 255.).permute(2, 0, 1)[None].repeat(K, 1, 1, 1)
        depth = F_PX * BASELINE / np.maximum(disp.astype(np.float32), 0.25)
        dep = torch.from_numpy(depth)[None, None].repeat(K, 1, 1, 1)
        img, dep = img.to(self.device, torch.float16), dep.to(self.device, torch.float16)
        torch.manual_seed(int(seeds[0]))
        self._x0, self._rec = [], True
        out = self.gw(src_image=img, src_depth=dep, rel_view_mtx=self.rel.repeat(K, 1, 1),
                      src_proj_mtx=self.proj.repeat(K, 1, 1), tar_proj_mtx=self.proj.repeat(K, 1, 1))
        self._rec = False
        syn = out["synthesized"].float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        syn = [cv2.resize((s * 255).round().astype(np.uint8), (CROP, CROP), interpolation=cv2.INTER_AREA) for s in syn]
        traj = None
        if len(self._x0) >= 2:
            X = torch.stack(self._x0[-self.traj_steps:])      # T x K x C x (1) x h x w
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[-2], X.shape[-1])
            v = X[:, 0].var(0).mean(0).cpu().numpy().astype(np.float32)
            traj = cv2.resize(v, (CROP, CROP), interpolation=cv2.INTER_LINEAR)
        return syn, traj

    def __call__(self, warped, mask, seeds, steps=None, src=None, disp=None):
        assert src is not None and disp is not None, "GenWarp needs the source image and its disparity"
        H, W = mask.shape
        assert H == CROP and W == X0S[-1] + CROP, (H, W)
        own = column_owner(W)
        outs = [np.zeros((H, W, 3), np.uint8) for _ in seeds]
        traj = np.zeros((H, W), np.float32)
        t0 = time.time()
        for k, x0 in enumerate(X0S):
            syn, tr = self._crop_call(src[:, x0:x0 + CROP], disp[:, x0:x0 + CROP], seeds)
            cols = np.flatnonzero(own == k)
            for s_out, s in zip(outs, syn):
                s_out[:, cols] = s[:, cols - x0]
            if tr is not None:
                traj[:, cols] = tr[:, cols - x0]
        self.last_traj = traj
        return outs, time.time() - t0
