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
