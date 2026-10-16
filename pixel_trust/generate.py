"""Generators that fill the pixels the warp could not supply.

`SDInpaint` is the real generator (latent diffusion inpainting, run K times with different seeds).
`TeleaInpaint` is a CPU stand-in used only by the tests.
In both cases warped pixels are kept exactly and only masked pixels are generated, so a generated
view = warped evidence + invented content, as in geometry-guided view synthesis.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

SD_CANDIDATES = (
    "stabilityai/stable-diffusion-2-inpainting",
    "Lykon/dreamshaper-8-inpainting",
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
)
PROMPT = "a realistic dashcam photo of a road scene, natural lighting, high detail"
NEGATIVE = "blurry, cartoon, painting, text, watermark, distorted"


class SDInpaint:
    def __init__(self, device: str = "cuda", steps: int = 25, guidance: float = 5.0, model: str | None = None):
        import torch
        from diffusers import AutoPipelineForInpainting
        self.torch, self.device, self.steps, self.guidance = torch, device, steps, guidance
        last = None
        for mid in ([model] if model else SD_CANDIDATES):
            try:
                self.pipe = AutoPipelineForInpainting.from_pretrained(mid, torch_dtype=torch.float16).to(device)
                self.model_id = mid
                break
            except Exception as e:  # model removed or gated: try the next one
                last = e
        else:
            raise RuntimeError(f"no inpainting model could be loaded: {last}")
        self.pipe.set_progress_bar_config(disable=True)

    def __call__(self, warped: np.ndarray, mask: np.ndarray, seeds: list[int], steps: int | None = None):
        from PIL import Image
        torch = self.torch
        h, w = mask.shape
        gens = [torch.Generator(self.device).manual_seed(int(s)) for s in seeds]
        t0 = time.time()
        out = self.pipe(prompt=[PROMPT] * len(seeds), negative_prompt=[NEGATIVE] * len(seeds),
                        image=Image.fromarray(warped), mask_image=Image.fromarray((mask * 255).astype(np.uint8)),
                        height=h, width=w, num_inference_steps=steps or self.steps,
                        guidance_scale=self.guidance, generator=gens).images
        dt = time.time() - t0
        samples = [np.asarray(im.resize((w, h))) for im in out]
        return [np.where(mask[..., None], s, warped) for s in samples], dt


class TeleaInpaint:
    """Classical inpainting plus seed-dependent smooth noise: gives sample-to-sample variability."""
    model_id = "opencv-telea (test stand-in)"

    def __call__(self, warped: np.ndarray, mask: np.ndarray, seeds: list[int], steps: int | None = None):
        base = cv2.inpaint(warped, mask.astype(np.uint8), 5, cv2.INPAINT_TELEA).astype(np.float32)
        outs = []
        for s in seeds:
            rng = np.random.default_rng(s)
            noise = cv2.GaussianBlur(rng.normal(0, 25, warped.shape).astype(np.float32), (0, 0), 3)
            samp = np.clip(base + noise, 0, 255).astype(np.uint8)
            outs.append(np.where(mask[..., None], samp, warped))
        return outs, 0.0
