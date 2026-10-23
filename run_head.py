"""v5: learned spatial trust head vs the per-pixel tree combiner, on KITTI and Middlebury.

Reads the per-pixel feature files written by run_generate.py (no generation needed):

    python run_head.py --kitti-sd K_SD --kitti-gw K_GW --mb-sd M_SD --mb-gw M_GW --out OUT

For each generator it compares, with identical calibration (conformal risk control, alpha = 0.1):
  trees (9 signals)      gradient-boosted trees on per-pixel signals (v1-v4 "ours")
  published (trees)      the four published signals combined by the same trees
  head (9 signals)       small U-Net over the nine signal maps            <- new
  head (9 + published)   the same head with the published maps as extra input channels
  head (joint)           one head trained on both generators, generator id as input
on KITTI (fixed split), on Middlebury (in-domain, 200 random scene-level splits) and for
KITTI -> Middlebury transfer (zero-shot ranking, threshold recalibrated on Middlebury scenes).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from pixel_trust import analysis as A
from pixel_trust.pipeline import FEATURES, SOTA
from pixel_trust.trust_head import TrustHead
from run_transfer import scene_units, split_eval, ALPHA

GENS = ("sd", "genwarp")
GEN_LABEL = {"sd": "SD inpainting", "genwarp": "GenWarp"}
ITERS = int(os.environ.get("PT_HEAD_ITERS", 3000))
BOOT = int(os.environ.get("PT_BOOT", 500))


def train_records(data):
    return [d for (i, c), d in sorted(data.items()) if str(d["split"]) == "train"]


def cache(data, key, fn):
    for d in data.values():
        d[f"f_{key}"] = fn(d).astype(np.float32).reshape(d["err"].shape)


def raw(key):
    return lambda d: d[f"f_{key}"].astype(np.float32).ravel()


def group_of(i, d):
    return i // 2 if str(d.get("dataset", "")) == "middlebury" else i


def paired_gap(data, cond, key_a, key_b, rng, sub=4000):
    """95% bootstrap interval (over test scenes) of AUROC(a) - AUROC(b)."""
    recs = [(group_of(i, d), d) for (i, c), d in sorted(data.items()) if c == cond and str(d["split"]) == "test"]
    groups = np.array([g for g, _ in recs])
    sel = [rng.choice(d["err"].size, min(sub, d["err"].size), replace=False) for _, d in recs]
    y = [(d["err"].astype(np.float32).ravel()[s] > A.EPS) for (_, d), s in zip(recs, sel)]
    sa = [d[f"f_{key_a}"].ravel()[s] for (_, d), s in zip(recs, sel)]
    sb = [d[f"f_{key_b}"].ravel()[s] for (_, d), s in zip(recs, sel)]
    ug, gaps = np.unique(groups), []
    for _ in range(BOOT):
        idx = np.concatenate([np.flatnonzero(groups == g) for g in rng.choice(ug, len(ug))])
        yy = np.concatenate([y[j] for j in idx])
        if yy.all() or not yy.any():
            continue
        gaps.append(roc_auc_score(yy, np.concatenate([sa[j] for j in idx])) -
                    roc_auc_score(yy, np.concatenate([sb[j] for j in idx])))
    g = np.array(gaps)
    return {"ci": [float(np.quantile(g, 0.025)), float(np.quantile(g, 0.975))], "p_le_0": float((g <= 0).mean())}


def kitti_eval(data, keys):
    out = {}
    for cond in ("clean", "corrupt"):
        out[cond] = {}
        for k in keys:
            ev = A.evaluate_scores(data, "test", cond, raw(k))
            cr = A.crc_evaluate(data, cond, raw(k), alphas=(ALPHA,))[ALPHA]
            out[cond][k] = {"auroc": ev["auroc"], "ause": ev["ause"], "miss": cr["miss"], "kept": cr["coverage"],
                            "miss_in_corrupt": cr["miss_in_corrupt"]}
    return out


def mb_eval(data, keys, rng):
    out = {}
    for cond in ("clean", "corrupt"):
        out[cond] = {}
        for k in keys:
            ev = A.evaluate_scores(data, "test", cond, raw(k))
            u = scene_units(data, cond, raw(k))
            out[cond][k] = {"auroc": ev["auroc"], "ause": ev["ause"],
                            "splits": split_eval(u, max(10, len(u) // 2), rng)}
    return out


def main():
    ap = argparse.ArgumentParser()
    for g in ("kitti-sd", "kitti-gw", "mb-sd", "mb-gw"):
        ap.add_argument(f"--{g}", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    K = {"sd": A.load_all(a.kitti_sd), "genwarp": A.load_all(a.kitti_gw)}
    M = {"sd": A.load_all(a.mb_sd), "genwarp": A.load_all(a.mb_gw)}
    sota = A.available({**K["sd"], **K["genwarp"], **M["sd"], **M["genwarp"]}, SOTA)
    allf = FEATURES + [k for k in sota if k not in FEATURES]
    print("published signals available:", sota, flush=True)
    R = {"iters": ITERS, "sota": sota}
    t0 = time.time()

    # joint, generator-aware heads (one on KITTI, one on Middlebury)
    joint = {}
    for name, D in (("kitti", K), ("mb", M)):
        recs, gids = [], []
        for gi, g in enumerate(GENS):
            tr = train_records(D[g]); recs += tr; gids += [gi] * len(tr)
        print(f"[joint {name}] training on {len(recs)} records", flush=True)
        joint[name] = TrustHead(FEATURES, n_gen=len(GENS), iters=ITERS).fit(recs, gids)

    for gi, g in enumerate(GENS):
        print(f"===== {GEN_LABEL[g]} ({time.time() - t0:.0f}s)", flush=True)
        kd, md = K[g], M[g]
        # --- KITTI-trained models ---
        trees_k = A.train_combiner(kd, FEATURES)
        pub_k = A.train_combiner(kd, sota) if sota else None
        head_k = TrustHead(FEATURES, iters=ITERS).fit(train_records(kd))
        headp_k = TrustHead(allf, iters=ITERS).fit(train_records(kd))
        print(f"  head params: {head_k.n_params():,}", flush=True)
        # --- Middlebury-trained models ---
        trees_m = A.train_combiner(md, FEATURES)
        pub_m = A.train_combiner(md, sota) if sota else None
        head_m = TrustHead(FEATURES, iters=ITERS).fit(train_records(md))
        headp_m = TrustHead(allf, iters=ITERS).fit(train_records(md))

        sc = lambda clf, names: (lambda d: clf.predict_proba(A.feat_matrix(d, names))[:, 1])
        for D, tag, trees, pub, head, headp, jt in ((kd, "", trees_k, pub_k, head_k, headp_k, joint["kitti"]),
                                                    (md, "", trees_m, pub_m, head_m, headp_m, joint["mb"])):
            cache(D, "trees", sc(trees, FEATURES))
            if pub is not None:
                cache(D, "pub", sc(pub, sota))
            cache(D, "head", head.scorer())
            cache(D, "headp", headp.scorer())
            cache(D, "joint", jt.scorer(gi))
        # KITTI-trained models applied to Middlebury (transfer)
        cache(md, "trees_k", sc(trees_k, FEATURES))
        cache(md, "head_k", head_k.scorer())
        if pub_k is not None:
            cache(md, "pub_k", sc(pub_k, sota))

        keys = ["trees", "head", "headp", "joint"] + (["pub"] if pub_k is not None else [])
        r = {"kitti": kitti_eval(kd, keys), "mb": mb_eval(md, keys, rng),
             "transfer": mb_eval(md, ["trees_k", "head_k"] + (["pub_k"] if pub_k is not None else []), rng)}
        r["gaps"] = {
            "kitti head-trees": {c: paired_gap(kd, c, "head", "trees", rng) for c in ("clean", "corrupt")},
            "mb head-trees": {c: paired_gap(md, c, "head", "trees", rng) for c in ("clean", "corrupt")},
            "transfer head_k-trees_k": {c: paired_gap(md, c, "head_k", "trees_k", rng) for c in ("clean", "corrupt")},
        }
        if pub_k is not None:
            r["gaps"]["kitti head-pub"] = {c: paired_gap(kd, c, "head", "pub", rng) for c in ("clean", "corrupt")}
            r["gaps"]["mb head-pub"] = {c: paired_gap(md, c, "head", "pub", rng) for c in ("clean", "corrupt")}
        R[g] = r
        for c in ("clean", "corrupt"):
            print(f"  [{g} {c}] KITTI AUROC " + " ".join(f"{k}={r['kitti'][c][k]['auroc']:.3f}" for k in keys)
                  + " | kept " + " ".join(f"{k}={r['kitti'][c][k]['kept']:.3f}" for k in keys), flush=True)
            print(f"  [{g} {c}] MB AUROC " + " ".join(f"{k}={r['mb'][c][k]['auroc']:.3f}" for k in keys)
                  + " | transfer " + " ".join(f"{k}={v['auroc']:.3f}" for k, v in r['transfer'][c].items()), flush=True)
        json.dump(R, open(os.path.join(a.out, "head.json"), "w"), indent=2, default=float)

    report(R, a.out)
    figure(R, a.out)
    print(open(os.path.join(a.out, "REPORT_head.md")).read())


NAMES = {"trees": "Trees, 9 signals (v1-v4 ours)", "pub": "Published signals (trees)", "head": "Trust head, 9 signals",
         "headp": "Trust head, 9 + published", "joint": "Trust head, joint over generators",
         "trees_k": "Trees (KITTI-trained)", "head_k": "Trust head (KITTI-trained)", "pub_k": "Published (KITTI-trained)"}


def report(R, out):
    L = ["# v5: learned spatial trust head", "",
         f"U-Net over the signal maps, {R['iters']} iterations; calibration and metrics identical to v1-v4 "
         f"(conformal risk control, alpha = {ALPHA:.0%}). Gap intervals: 95% bootstrap over test scenes.", ""]
    for g in GENS:
        if g not in R:
            continue
        r = R[g]
        L += [f"## {GEN_LABEL[g]}", ""]
        for c in ("clean", "corrupt"):
            L += [f"### depth prior: {c}", "",
                  "| method | KITTI AUROC | KITTI AUSE | KITTI kept | KITTI miss | Middlebury AUROC | Middlebury kept (200 splits) | Middlebury mean miss |",
                  "|---|---|---|---|---|---|---|---|"]
            for k, v in r["kitti"][c].items():
                m = r["mb"][c][k]
                L.append(f"| {NAMES[k]} | {v['auroc']:.3f} | {v['ause']:.4f} | {v['kept']:.1%} | {v['miss']:.1%} | "
                         f"{m['auroc']:.3f} | {m['splits']['kept_mean']:.1%} | {m['splits']['miss_mean']:.1%} |")
            L += ["", "Transfer KITTI -> Middlebury (zero-shot ranking; threshold recalibrated on Middlebury scenes):", "",
                  "| method | AUROC | kept | mean miss | P(miss > alpha) |", "|---|---|---|---|---|"]
            for k, v in r["transfer"][c].items():
                s = v["splits"]
                L.append(f"| {NAMES[k]} | {v['auroc']:.3f} | {s['kept_mean']:.1%} | {s['miss_mean']:.1%} | {s['violation_rate']:.0%} |")
            L += ["", "AUROC gaps (95% CI):", ""]
            for name, gg in r["gaps"].items():
                x = gg[c]
                L.append(f"- {name}: {x['ci'][0]:+.3f} to {x['ci'][1]:+.3f} (share <= 0: {x['p_le_0']:.0%})")
            L.append("")
    open(os.path.join(out, "REPORT_head.md"), "w").write("\n".join(L))


def figure(R, out):
    keys = ["pub", "trees", "head", "headp", "joint"]
    cols = {"pub": "#2F6690", "trees": "#8C96A0", "head": "#C2571A", "headp": "#E08A4F", "joint": "#3B7D4F"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, ds, title in ((axes[0], "kitti", "KITTI (test)"), (axes[1], "mb", "Middlebury (test)")):
        x = np.arange(len(GENS))
        ks = [k for k in keys if all(k in R[g][ds]["clean"] for g in GENS if g in R)]
        w = 0.8 / len(ks)
        for j, k in enumerate(ks):
            vals = [R[g][ds]["clean"][k]["auroc"] for g in GENS]
            ax.bar(x + (j - (len(ks) - 1) / 2) * w, vals, w, color=cols[k], label=NAMES[k])
            for xi, v in zip(x, vals):
                ax.text(xi + (j - (len(ks) - 1) / 2) * w, v + 0.003, f"{v:.3f}", ha="center", fontsize=7, rotation=90)
        ax.set_xticks(x); ax.set_xticklabels([GEN_LABEL[g] for g in GENS])
        lo = min(R[g][ds]["clean"][k]["auroc"] for g in GENS for k in ks)
        ax.set_ylim(max(0.5, lo - 0.05), 1.0); ax.set_ylabel("AUROC (wrong pixels)"); ax.set_title(title)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_head.png"), dpi=170); plt.close(fig)


if __name__ == "__main__":
    main()
