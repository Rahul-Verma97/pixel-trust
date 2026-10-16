"""Stage 2 (CPU): all numbers, figures and REPORT.md from the features written by run_generate.py.

    python run_analysis.py --out /kaggle/working/out
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pixel_trust import analysis as A
from pixel_trust.pipeline import GROUPS

INK, BLUE, ORANGE, GREY, GREEN = "#1B2A3A", "#2F6690", "#C2571A", "#8C96A0", "#3B7D4F"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fig_dir = os.path.join(a.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    data = A.load_all(a.out)
    splits = {s: sum(1 for (i, c), d in data.items() if c == "clean" and str(d["split"]) == s)
              for s in ("train", "calib", "test")}
    R = {"n_images": splits, "eps": A.EPS}
    print("images per split:", splits)

    # ---------- learned combiners (trained on train images, clean + corrupted depth) ----------
    clfs = {g: A.train_combiner(data, names) for g, names in GROUPS.items()}
    methods = {
        "warp mask only": A.scorer("inv"),
        "sample variance only": A.scorer("var"),
        "depth reliability only": A.scorer("lidar_rel"),
        **{f"learned: {g}": A.scorer("model", clfs[g], GROUPS[g]) for g in GROUPS},
    }
    full = f"learned: + depth reliability"

    # ---------- Thrust 1: ranking quality (AUSE / AUROC) on held-out test images ----------
    R["t1"] = {}
    curves = {}
    for cond in ("clean", "corrupt"):
        R["t1"][cond] = {}
        for name, fn in {**methods, "random": A.scorer("random")}.items():
            ev = A.evaluate_scores(data, "test", cond, fn)
            R["t1"][cond][name] = {"ause": ev["ause"], "auroc": ev["auroc"]}
            curves[(cond, name)] = ev
        R["t1"][cond]["error_rate"] = curves[(cond, "random")]["err_rate"]
        print(cond, json.dumps({k: v for k, v in R["t1"][cond].items()}, indent=0)[:600])

    # ---------- Thrust 1: conformal guarantee ----------
    R["crc"] = {}
    for cond in ("clean", "corrupt"):
        R["crc"][cond] = {name: {str(k): v for k, v in A.crc_evaluate(data, cond, fn).items()}
                          for name, fn in methods.items() if name != "warp mask only"}
        R["crc"][cond]["warp mask only (uncalibrated)"] = A.warp_mask_policy(data, cond)

    # ---------- Thrust 2: regeneration of unreliable warped pixels ----------
    R["t2_regen"] = A.regen_table(data)

    # ---------- Thrust 3: trust quality vs number of diffusion samples ----------
    Ks = sorted(int(k.split("_k")[1]) for k in next(iter(data.values())) if k.startswith("f_var_k"))
    R["t3"] = {}
    for K in Ks:
        vk = f"var_k{K}"
        clf = A.train_combiner(data, GROUPS["+ depth reliability"], var_key=vk)
        e_full = A.evaluate_scores(data, "test", "clean", A.scorer("model", clf, GROUPS["+ depth reliability"], vk))
        e_var = A.evaluate_scores(data, "test", "clean", A.scorer(vk))
        R["t3"][K] = {"ause_full": e_full["ause"], "auroc_full": e_full["auroc"],
                      "ause_var": e_var["ause"], "auroc_var": e_var["auroc"]}
    secs = [float(d["gen_seconds"]) for d in data.values()]
    R["gen_seconds_per_image_K"] = float(np.mean(secs))

    json.dump(R, open(os.path.join(a.out, "results.json"), "w"), indent=2, default=float)

    # =================== figures ===================
    # Fig A: sparsification curves (clean depth)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, cond, title in zip(axes, ("clean", "corrupt"), ("Depth prior as estimated", "Depth prior with off-road-style corruption")):
        ax.plot(A.FRACS, curves[(cond, "random")]["oracle"], color=INK, lw=2, ls="--", label="oracle (true error)")
        for name, col in (("random", GREY), ("warp mask only", "#B9A27A"), ("learned: warp support", BLUE),
                          ("learned: + sample variance", "#6C9BC0"), (full, ORANGE)):
            ax.plot(A.FRACS, curves[(cond, name)]["curve"], color=col, lw=2,
                    label=f"{name} (AUSE {curves[(cond, name)]['ause']:.4f})")
        ax.set_title(title); ax.set_xlabel("fraction of least-trusted pixels removed"); ax.set_ylabel("mean error of kept pixels")
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig_sparsification.png"), dpi=170); plt.close(fig)

    # Fig B: conformal guarantee: achieved miss rate vs target, and coverage
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = ["sample variance only", "depth reliability only", "learned: warp support", "learned: + sample variance", full]
    cols = [GREY, "#B9A27A", BLUE, "#6C9BC0", ORANGE]
    for j, cond in enumerate(("clean", "corrupt")):
        ax = axes[j]
        x = np.arange(len(names))
        cov = [R["crc"][cond][n]["0.1"]["coverage"] for n in names]
        miss = [R["crc"][cond][n]["0.1"]["miss"] for n in names]
        ax.bar(x, cov, color=cols)
        for xi, c, m in zip(x, cov, miss):
            ax.text(xi, c + 0.01, f"{c:.0%}\nmiss {m:.0%}", ha="center", fontsize=8)
        wm = R["crc"][cond]["warp mask only (uncalibrated)"]
        ax.axhline(wm["coverage"], color=INK, ls=":", lw=1)
        ax.text(len(names) - 0.5, wm["coverage"] + 0.01, f"warp mask: keeps {wm['coverage']:.0%}, lets {wm['miss']:.0%} of wrong pixels through",
                ha="right", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([n.replace("learned: ", "") for n in names], rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.15); ax.set_ylabel("fraction of view kept as trusted")
        ax.set_title(f"{'Estimated' if cond == 'clean' else 'Corrupted'} depth: guarantee α = 0.10")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig_conformal.png"), dpi=170); plt.close(fig)

    # Fig C: T3 samples
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(Ks, [R["t3"][k]["auroc_full"] for k in Ks], "o-", color=ORANGE, label="learned (all signals)")
    ax.plot(Ks, [R["t3"][k]["auroc_var"] for k in Ks], "o-", color=GREY, label="sample variance only")
    ax.set_xlabel("diffusion samples per view (K)"); ax.set_ylabel("AUROC for wrong pixels"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig_samples.png"), dpi=170); plt.close(fig)

    # Fig D: qualitative examples
    png_dir = os.path.join(a.out, "png")
    examples = sorted({int(f[:6]) for f in os.listdir(png_dir) if f.endswith("_left.jpg")})[:3] if os.path.isdir(png_dir) else []
    if examples:
        fig, axes = plt.subplots(len(examples), 5, figsize=(20, 2.2 * len(examples)))
        axes = np.atleast_2d(axes)
        fn = methods[full]
        lam = R["crc"]["clean"][full]["0.1"]["lambda"]
        for r, idx in enumerate(examples):
            d = data[(idx, "clean")]
            h, w = d["err"].shape
            rd = lambda n: cv2.cvtColor(cv2.imread(os.path.join(png_dir, n)), cv2.COLOR_BGR2RGB)
            gen = rd(f"{idx:06d}_clean_gen.jpg"); gt = rd(f"{idx:06d}_right.jpg")
            s = fn(d).reshape(h, w)
            panels = [(gt, "real right image (ground truth)"), (gen, "generated right view (left + depth + diffusion)"),
                      (d["err"].astype(np.float32), "true error"), (s, "predicted distrust"),
                      (None, "accepted as trusted at α = 0.10")]
            for c, (im, t) in enumerate(panels):
                ax = axes[r, c]
                if c == 4:
                    small = cv2.resize(gen, (w, h))
                    acc = (s <= lam)
                    over = small.copy(); over[~acc] = (0.35 * over[~acc] + 0.65 * np.array([194, 87, 26])).astype(np.uint8)
                    ax.imshow(over)
                elif c in (2, 3):
                    ax.imshow(im, cmap="magma", vmin=0, vmax=np.quantile(im, 0.99))
                else:
                    ax.imshow(im)
                ax.set_axis_off()
                if r == 0:
                    ax.set_title(t, fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig_examples.png"), dpi=150); plt.close(fig)

    write_report(R, a.out)
    print("done:", os.path.join(a.out, "REPORT.md"))


def write_report(R, out):
    L = ["# pixel-trust: preliminary results", "",
         f"KITTI Stereo 2015, {sum(R['n_images'].values())} real stereo pairs "
         f"(train {R['n_images']['train']} / calibration {R['n_images']['calib']} / test {R['n_images']['test']}). "
         f"A pixel is *wrong* if its colour error exceeds {R['eps']:.2f}.", "",
         "## Thrust 1: ranking wrong pixels (test images)", "",
         "| Signal | AUROC (estimated depth) | AUSE (estimated depth) | AUROC (corrupted depth) | AUSE (corrupted depth) |",
         "|---|---|---|---|---|"]
    for name in R["t1"]["clean"]:
        if name == "error_rate":
            continue
        c, k = R["t1"]["clean"][name], R["t1"]["corrupt"][name]
        L.append(f"| {name} | {c['auroc']:.3f} | {c['ause']:.4f} | {k['auroc']:.3f} | {k['ause']:.4f} |")
    L += ["", f"Share of wrong pixels: {R['t1']['clean']['error_rate']:.1%} (estimated depth), "
              f"{R['t1']['corrupt']['error_rate']:.1%} (corrupted depth).", "",
          "## Thrust 1: conformal guarantee on held-out images (target: at most α of wrong pixels accepted)", "",
          "| Depth | Method | α | achieved miss rate | view kept | wrong among kept | wrong pixels in corrupted region accepted |",
          "|---|---|---|---|---|---|---|"]
    for cond in ("clean", "corrupt"):
        for name, v in R["crc"][cond].items():
            if "uncalibrated" in name:
                L.append(f"| {cond} | {name} | – | {v['miss']:.3f} | {v['coverage']:.1%} | {v['wrong_among_accepted']:.1%} | – |")
                continue
            for al, r in v.items():
                mc = "–" if r["miss_in_corrupt"] is None else f"{r['miss_in_corrupt']:.1%}"
                L.append(f"| {cond} | {name} | {al} | {r['miss']:.3f} | {r['coverage']:.1%} | {r['wrong_among_accepted']:.1%} | {mc} |")
    L += ["", "## Thrust 2: regenerate unreliable warped pixels (corrupted depth, test images)", "",
          "| Policy | mean error (all) | wrong (all) | mean error (corrupted region) | wrong (corrupted region) |",
          "|---|---|---|---|---|"]
    for k, v in R["t2_regen"].items():
        L.append(f"| {k} | {v['mae_all']:.4f} | {v['wrong_all']:.1%} | {v['mae_corrupt']:.4f} | {v['wrong_corrupt']:.1%} |")
    L += ["", "## Thrust 3: trust quality vs number of diffusion samples", "",
          "| K | AUROC all signals | AUROC variance only | AUSE all signals |", "|---|---|---|---|"]
    for K, v in R["t3"].items():
        L.append(f"| {K} | {v['auroc_full']:.3f} | {v['auroc_var']:.3f} | {v['ause_full']:.4f} |")
    L += ["", f"Mean generation time per image for all K samples: {R['gen_seconds_per_image_K']:.1f} s.", ""]
    open(os.path.join(out, "REPORT.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    main()
