"""v4: does trust transfer to a new domain, and does the guarantee survive?

For each generator, a trust model is trained and calibrated on the *source* dataset (KITTI) and
applied to the held-out test images of the *target* dataset (Middlebury):

  in-domain     trained on target train, calibrated on target calib        (upper reference)
  zero-shot     trained on source train, calibrated on source calib        (no target data at all)
  recalibrated  trained on source train, threshold re-fitted on n target
                calibration images (n = 10, 12, ..., all), 200 random draws per n
                (n >= 10 is the smallest that can certify alpha = 0.1)

Ranking quality (AUROC, AUSE) depends only on the trained model; the guarantee (miss rate <= alpha)
depends on the threshold. So the question splits cleanly: do the signals transfer, and does the
threshold?

    python run_transfer.py --pair "SD inpainting=/path/kitti_sd_out,/path/mb_sd_out" \
                           --pair "GenWarp=/path/kitti_gw_out,/path/mb_gw_out" --out /kaggle/working/transfer
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pixel_trust import analysis as A
from pixel_trust.pipeline import FEATURES, SOTA

ALPHA = 0.10
DRAWS = 200
INK, BLUE, ORANGE, GREY, GREEN = "#1B2A3A", "#2F6690", "#C2571A", "#8C96A0", "#3B7D4F"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def items(data, split, cond, fn):
    return [(d["err"].astype(np.float32).ravel(), fn(d)) for (i, c), d in sorted(data.items())
            if c == cond and str(d["split"]) == split]


def grid_of(cal):
    allS = np.concatenate([s for _, s in cal])
    return np.unique(np.quantile(allS, np.linspace(0, 1, 1000)))


def apply(lam, test):
    miss, cov = [], []
    for e, s in test:
        y, acc = e > A.EPS, s <= lam
        miss.append(float((acc & y).sum() / max(1, y.sum())) if y.any() else 0.0)
        cov.append(float(acc.mean()))
    return float(np.mean(miss)), float(np.mean(cov))


def ranking(test):
    e = np.concatenate([x for x, _ in test]); s = np.concatenate([x for _, x in test])
    from sklearn.metrics import roc_auc_score
    y = e > A.EPS
    return float(roc_auc_score(y, s)) if y.any() and (~y).any() else float("nan")


def recal_curve(cal_t, test, ns, rng):
    out = {}
    for n in ns:
        n = min(n, len(cal_t))
        runs = []
        for _ in range(DRAWS if n < len(cal_t) else 1):
            sub = [cal_t[j] for j in rng.choice(len(cal_t), n, replace=False)]
            lam = A.crc_threshold(sub, ALPHA, grid_of(sub))
            runs.append(apply(lam, test))
        m = np.array(runs)
        out[n] = {"miss_mean": float(m[:, 0].mean()), "miss_p90": float(np.quantile(m[:, 0], 0.9)),
                  "violation_rate": float((m[:, 0] > ALPHA).mean()), "coverage_mean": float(m[:, 1].mean())}
    return out


def methods_for(src, tgt):
    """Scorers trained on the source and on the target, for features present in both."""
    both = {**src, **tgt}
    sota = A.available(both, SOTA)
    sets = {"Ours (9 signals)": FEATURES}
    if sota:
        sets["Published signals combined"] = sota
        sets["Ours + published"] = FEATURES + [k for k in sota if k not in FEATURES]
    m = {}
    for name, feats in sets.items():
        m[name] = {"src": A.scorer("model", A.train_combiner(src, feats), feats),
                   "tgt": A.scorer("model", A.train_combiner(tgt, feats), feats)}
    m["Sample variance (raw)"] = {"src": A.scorer("var"), "tgt": A.scorer("var")}
    return m


def run_pair(label, src_dir, tgt_dir, rng):
    src, tgt = A.load_all(src_dir), A.load_all(tgt_dir)
    n_cal = sum(1 for (i, c), d in tgt.items() if c == "clean" and str(d["split"]) == "calib")
    # conformal risk control needs n/(n+1)*R + 1/(n+1) <= alpha, so with alpha = 0.1 at least 10 images
    ns = sorted({n for n in (10, 12, 14, 16) if n <= n_cal} | {n_cal})
    res = {"n_target": {s: sum(1 for (i, c), d in tgt.items() if c == "clean" and str(d["split"]) == s)
                        for s in ("train", "calib", "test")},
           "error_rate": {}, "methods": {}}
    for name, f in methods_for(src, tgt).items():
        res["methods"][name] = {}
        for cond in ("clean", "corrupt"):
            test_src_model = items(tgt, "test", cond, f["src"])
            test_tgt_model = items(tgt, "test", cond, f["tgt"])
            res["error_rate"][cond] = float(np.mean([(e > A.EPS).mean() for e, _ in test_src_model]))
            cal_src = items(src, "calib", cond, f["src"])
            cal_tgt_src_model = items(tgt, "calib", cond, f["src"])
            cal_tgt = items(tgt, "calib", cond, f["tgt"])
            lam_in = A.crc_threshold(cal_tgt, ALPHA, grid_of(cal_tgt))
            lam_zero = A.crc_threshold(cal_src, ALPHA, grid_of(cal_src))
            mi, ci = apply(lam_in, test_tgt_model)
            mz, cz = apply(lam_zero, test_src_model)
            ev_in = A.evaluate_scores(tgt, "test", cond, f["tgt"])
            ev_tr = A.evaluate_scores(tgt, "test", cond, f["src"])
            res["methods"][name][cond] = {
                "in_domain": {"auroc": ev_in["auroc"], "ause": ev_in["ause"], "miss": mi, "coverage": ci},
                "zero_shot": {"auroc": ev_tr["auroc"], "ause": ev_tr["ause"], "miss": mz, "coverage": cz},
                "recalibrated": recal_curve(cal_tgt_src_model, test_src_model, ns, rng),
            }
            print(f"[{label}] {name:28s} {cond:7s} in-domain AUROC {ev_in['auroc']:.3f} miss {mi:.3f} kept {ci:.3f} | "
                  f"zero-shot AUROC {ev_tr['auroc']:.3f} miss {mz:.3f} kept {cz:.3f}", flush=True)
    return res


def figure(R, path):
    labels = list(R)
    fig, axes = plt.subplots(2, len(labels), figsize=(5.6 * len(labels), 7.6), squeeze=False)
    name = "Ours (9 signals)"
    for j, lab in enumerate(labels):
        r = R[lab]["methods"][name]["clean"]
        rec = r["recalibrated"]
        ns = sorted(int(n) for n in rec)
        ax = axes[0, j]
        bars = [("in-domain", r["in_domain"]["miss"], GREEN), ("zero-shot\n(KITTI threshold)", r["zero_shot"]["miss"], ORANGE)]
        bars += [(f"recal. n={n}", rec[n]["miss_mean"], BLUE) for n in ns]
        x = np.arange(len(bars))
        ax.bar(x, [b[1] for b in bars], color=[b[2] for b in bars])
        for xi, b in zip(x, bars):
            ax.text(xi, b[1] + 0.005, f"{b[1]:.1%}", ha="center", fontsize=8)
        ax.axhline(ALPHA, color=INK, ls="--", lw=1)
        ax.text(len(bars) - 0.5, ALPHA + 0.004, "target α = 10%", ha="right", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars], fontsize=8, rotation=20, ha="right")
        ax.set_ylabel("miss rate on Middlebury test"); ax.set_title(f"{lab}: wrong pixels accepted")
        ax = axes[1, j]
        ax.plot(ns, [rec[n]["violation_rate"] for n in ns], "o-", color=BLUE, lw=2, label="P(miss > α) over random draws")
        ax.plot(ns, [rec[n]["coverage_mean"] for n in ns], "s-", color=GREY, lw=2, label="view kept as trusted")
        ax.axhline(r["in_domain"]["coverage"], color=GREEN, ls=":", lw=1.5, label="kept, in-domain")
        ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels(ns)
        ax.set_xlabel("target images used to recalibrate the threshold"); ax.set_ylim(0, 1)
        ax.legend(fontsize=8, frameon=False); ax.set_title("Recalibration with a few target images")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def report(R, path):
    L = ["# v4: cross-domain transfer (KITTI -> Middlebury)", "",
         f"Target guarantee: at most α = {ALPHA:.0%} of the wrong pixels accepted as trusted. "
         f"Recalibration rows average {DRAWS} random draws of n calibration images.", ""]
    for lab, r in R.items():
        L += [f"## {lab}", "", f"Middlebury images per split: {r['n_target']}; wrong-pixel rate on test: "
              + ", ".join(f"{k} {v:.1%}" for k, v in r["error_rate"].items()), ""]
        for cond in ("clean", "corrupt"):
            L += [f"### depth prior: {cond}", "",
                  "| method | AUROC in-domain | AUROC zero-shot | miss in-domain | miss zero-shot | kept in-domain | kept zero-shot |",
                  "|---|---|---|---|---|---|---|"]
            for name, m in r["methods"].items():
                a, z = m[cond]["in_domain"], m[cond]["zero_shot"]
                L.append(f"| {name} | {a['auroc']:.3f} | {z['auroc']:.3f} | {a['miss']:.1%} | {z['miss']:.1%} | "
                         f"{a['coverage']:.1%} | {z['coverage']:.1%} |")
            L += ["", "Recalibration of the KITTI-trained model (ours, 9 signals):", "",
                  "| n target images | mean miss | 90th pct miss | P(miss > α) | kept |", "|---|---|---|---|---|"]
            for n, v in r["methods"]["Ours (9 signals)"][cond]["recalibrated"].items():
                L.append(f"| {n} | {v['miss_mean']:.1%} | {v['miss_p90']:.1%} | {v['violation_rate']:.0%} | {v['coverage_mean']:.1%} |")
            L.append("")
    open(path, "w").write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True, help='"label=source_out_dir,target_out_dir"')
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    R = {}
    for p in a.pair:
        label, dirs = p.split("=", 1)
        src, tgt = dirs.split(",")
        R[label] = run_pair(label, src, tgt, rng)
    json.dump(R, open(os.path.join(a.out, "transfer.json"), "w"), indent=2, default=float)
    figure(R, os.path.join(a.out, "fig_transfer.png"))
    report(R, os.path.join(a.out, "REPORT_transfer.md"))
    print(open(os.path.join(a.out, "REPORT_transfer.md")).read())


if __name__ == "__main__":
    main()
