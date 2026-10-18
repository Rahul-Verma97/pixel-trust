"""Published uncertainty signals used as baselines (v2).

Each baseline maps a generated view to a per-pixel *distrust* map (higher = less trusted), at the
full image resolution. Every baseline is optional: if its model cannot be loaded the run continues
and the analysis skips it.

* Trajectory variance  - Aithal et al., "Understanding Hallucinations in Diffusion Models through
  Mode Interpolation", NeurIPS 2024: variance of the predicted clean sample x0 over the final
  denoising steps. Captured inside the generator (generate.py), not here.
* Semantic entropy     - Bose et al., "Uncertainty-Aware Diffusion Guided Refinement of 3D Scenes",
  ICCV 2025: per-pixel Shannon entropy of a segmentation network's class distribution on the
  generated view. The paper uses LSeg with BLIP-2 tags; for street scenes we use SegFormer-B2
  trained on Cityscapes (19 classes), entropy normalised by log(19).
* Cross-view geometric confidence - the verification idea of OracleGS (2025) and ConFixGS (2026),
  implemented with VGGT (Wang et al., CVPR 2025): run the 3D foundation model jointly on the real
  source image and the generated view; low point-map confidence on the generated view marks
  content that is not consistent with the source in 3D. Distrust = 1 / confidence.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np


class SemanticEntropy:
    name = "semantic entropy (Bose et al., ICCV 2025)"

    def __init__(self, device="cuda", model="nvidia/segformer-b2-finetuned-cityscapes-1024-1024"):
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        self.torch, self.device = torch, device
        self.proc = SegformerImageProcessor.from_pretrained(model)
        self.net = SegformerForSemanticSegmentation.from_pretrained(model).to(device).eval().half()

    def __call__(self, img: np.ndarray, source: np.ndarray | None = None) -> np.ndarray:
        torch = self.torch
        inp = self.proc(images=img, return_tensors="pt", do_resize=False)
        px = inp["pixel_values"].to(self.device).half()
        with torch.no_grad():
            logits = self.net(pixel_values=px).logits.float()
        logits = torch.nn.functional.interpolate(logits, size=img.shape[:2], mode="bilinear", align_corners=False)
        p = torch.softmax(logits, 1)
        ent = -(p * torch.log(p.clamp_min(1e-8))).sum(1)[0] / np.log(p.shape[1])
        return ent.cpu().numpy().astype(np.float32)


class VGGTConfidence:
    name = "cross-view confidence (VGGT; OracleGS/ConFixGS-style)"

    def __init__(self, device="cuda", repo_dir="/kaggle/working/vggt"):
        import torch
        if not os.path.isdir(repo_dir):
            subprocess.run(["git", "clone", "-q", "https://github.com/facebookresearch/vggt", repo_dir], check=True)
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        self.torch, self.device, self.load = torch, device, load_and_preprocess_images
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.net = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
        self.tmp = tempfile.mkdtemp()

    def __call__(self, img: np.ndarray, source: np.ndarray | None = None) -> np.ndarray:
        torch = self.torch
        pa, pb = os.path.join(self.tmp, "src.png"), os.path.join(self.tmp, "gen.png")
        cv2.imwrite(pa, cv2.cvtColor(source, cv2.COLOR_RGB2BGR))
        cv2.imwrite(pb, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        images = self.load([pa, pb], mode="crop").to(self.device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                pred = self.net(images)
        conf = pred["world_points_conf"][0, 1].float().cpu().numpy()   # confidence on the generated view
        conf = cv2.resize(conf, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        return (1.0 / np.maximum(conf, 1e-3)).astype(np.float32)


class RandomStub:
    """CPU test stand-in for any baseline."""

    def __init__(self, name="stub", seed=0):
        self.name, self.rng = name, np.random.default_rng(seed)

    def __call__(self, img: np.ndarray, source: np.ndarray | None = None) -> np.ndarray:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        edges = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
        return (edges + 0.05 * self.rng.random(g.shape)).astype(np.float32)


def load_baselines(kind: str, seg_model: str | None = None, log=print) -> dict:
    """kind = 'real' (GPU models) or 'stub' (tests). Failures are logged and skipped."""
    out = {}
    if kind == "none":
        return out
    sem = (lambda: SemanticEntropy(model=seg_model)) if seg_model else SemanticEntropy
    specs = {"sem": sem, "vggt": VGGTConfidence} if kind == "real" else \
            {"sem": lambda: RandomStub("sem", 1), "vggt": lambda: RandomStub("vggt", 2)}
    for key, ctor in specs.items():
        try:
            out[key] = ctor()
            log(f"baseline loaded: {key}")
        except Exception as e:  # keep the run alive; analysis skips missing baselines
            log(f"baseline FAILED to load: {key}: {type(e).__name__}: {e}")
    return out
