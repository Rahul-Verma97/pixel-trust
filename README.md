# pixel-trust

Preliminary experiments for the research proposal **"Knowing Which Generated Pixels to Trust: Trustworthy, Compute-Efficient View Synthesis for Autonomous Ground Vehicles"** (Rahul Verma).

A geometry-guided generator warps one camera into another through a depth prior and uses diffusion to fill what the warp cannot supply. Some output pixels are supported by geometry; others are invented. These experiments ask whether a system can tell them apart, with a statistical guarantee.

## Setup
- **Data:** KITTI Stereo 2015, 200 real stereo pairs. Left image = source camera, right image = target camera and ground truth. Split by index: 80 train / 40 calibration / 80 test.
- **Depth prior:** Depth Anything V2 (Small + Base, with flip augmentation) aligned in disparity space to a sparse LiDAR subset (10% of the KITTI ground-truth points).
- **Generator:** forward warp with a z-buffer, then Stable Diffusion inpainting for the disoccluded and out-of-view pixels, K = 4 seeds.
- **Wrong pixel:** blurred colour error above 0.08 against the real right image.

## Experiments
1. **Thrust 1, trust:** per-pixel signals (warp support, sample variance, depth reliability) and a learned combination; AUROC and sparsification (AUSE); conformal risk control so that at most α of the wrong pixels are accepted as trusted on held-out images.
2. **Thrust 2, evidence:** the depth prior is corrupted in a few smooth regions (as vegetation, dust or thin structures would); we compare trust with and without the depth-reliability signal, and regenerating the least reliable warped pixels.
3. **Thrust 3, compute:** trust quality as a function of the number of diffusion samples.

## v2: comparison with published uncertainty signals
Every signal below is turned into a per-pixel distrust map on the same generated views and calibrated with the same conformal procedure:
- **Multi-sample variance**, the sampling/ensemble principle behind Conffusion and BayesDiff.
- **Trajectory variance** of the predicted clean image over the last 10 DDIM steps (Aithal et al., NeurIPS 2024).
- **Semantic entropy** of a segmentation network on the generated view (Bose et al., ICCV 2025; SegFormer-B2 Cityscapes instead of LSeg + BLIP-2).
- **Cross-view geometric confidence**: VGGT (CVPR 2025) run jointly on the real source image and the generated view; low confidence = inconsistent in 3D (the verification idea of OracleGS and ConFixGS).
Generation uses the DDIM scheduler in v2 so the predicted x0 is available at every step.

## Run
On Kaggle (GPU T4 x2, Internet on): import `pixel_trust_kaggle.ipynb` and run all. Locally:
```
python run_generate.py --root <kitti> --out out --shard 0 --nshards 1
python run_analysis.py --out out
```
CPU smoke test with synthetic scenes and stand-in models:
```
python tests/make_synthetic.py /tmp/syn 20
python run_generate.py --root /tmp/syn --out /tmp/synout --gen telea --depth oracle
python run_analysis.py --out /tmp/synout
```

## v3: a second, real generator (GenWarp)
`--gen genwarp` replaces the inpainting stand-in with GenWarp (Seo et al., NeurIPS 2024), which regenerates the whole target view from the source image, the depth prior and the stereo pose (`pixel_trust/genwarp_gen.py`). GenWarp works on 512x512 squares, so each KITTI frame is covered by four overlapping 368x368 crops stitched column-wise. Softmax splatting is reimplemented in plain PyTorch so no CUDA extension has to be compiled. Kaggle notebook: `pixel_trust_kaggle_genwarp.ipynb`.

## v4: a second dataset (Middlebury) and cross-domain transfer
`--dataset middlebury` runs the same pipeline on Middlebury stereo 2014 (23 scenes with ground truth) and 2021 (24 scenes): indoor, dense ground-truth disparity, a different camera, baseline and depth range (`pixel_trust/middlebury.py`). Each scene is used in both directions (mirroring the pair), frames are resized to height 368, and splits are made by scene group (38 / 18 / 38 pairs). The sparse "LiDAR" keeps 5% of the dense ground truth, a density comparable to the KITTI setting. Semantic entropy uses SegFormer-ADE20K on this indoor data.

`run_transfer.py` then asks whether trust transfers: a trust model trained and calibrated on KITTI is applied to Middlebury as is (zero-shot), and with its threshold re-fitted on n Middlebury images (n >= 10, the smallest number that can certify α = 0.1), against an in-domain reference. Kaggle notebook: `pixel_trust_kaggle_v4_middlebury.ipynb` (attach the v2 and v3 notebook outputs as inputs to reuse the KITTI features).

CPU smoke test:
```
python tests/make_synthetic_mb.py /tmp/mb/raw Adirondack,Sword1,chess1,...
python -c "from pixel_trust.middlebury import prepare; prepare('/tmp/mb', scenes=[...])"
python run_generate.py --dataset middlebury --root /tmp/mb --out /tmp/mbout --gen telea --depth oracle --baselines stub
python run_transfer.py --pair "Telea=/tmp/synout,/tmp/mbout" --out /tmp/transfer
```
