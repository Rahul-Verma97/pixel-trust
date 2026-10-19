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


BOOT = int(__import__('os').environ.get('PT_BOOT', 500))
SUB = 4000   # pixels per image used for the bootstrap AUROC


def test_groups(data, cond):
    """Scene id of each test image, in the same order as items() (mirrored pairs share a scene)."""
    return np.array([i // 2 if str(d.get("dataset", "")) == "middlebury" else i
                     for (i, c), d in sorted(data.items()) if c == cond and str(d["split"]) == "test"])


def bootstrap(per, groups, rng):
    """95% intervals over test scenes (resampled with replacement) for AUROC, miss and kept,
    plus the paired AUROC gap between ours and the published combination."""
    from sklearn.metrics import roc_auc_score
    ug = np.unique(groups)
    n_img = len(groups)
    sel = [rng.choice(len(per[next(iter(per))]["items"][j][0]), SUB, replace=False) for j in range(n_img)]
    sub = {m: [(v["items"][j][0][sel[j]] > A.EPS, v["items"][j][1][sel[j]]) for j in range(n_img)] for m, v in per.items()}
    stats = {m: {"auroc": [], "miss": [], "kept": []} for m in per}
    gap = []
    for _ in range(BOOT):
        g = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == x) for x in g])
        au = {}
        for m, v in per.items():
            y = np.concatenate([sub[m][j][0] for j in idx]); sc = np.concatenate([sub[m][j][1] for j in idx])
            au[m] = roc_auc_score(y, sc) if y.any() and (~y).any() else np.nan
            stats[m]["auroc"].append(au[m])
            mi, kp = apply(v["lam"], [v["items"][j] for j in idx])
            stats[m]["miss"].append(mi); stats[m]["kept"].append(kp)
        if "Ours (9 signals)" in au and "Published signals combined" in au:
            gap.append(au["Ours (9 signals)"] - au["Published signals combined"])
    q = lambda a: [float(np.nanquantile(a, 0.025)), float(np.nanquantile(a, 0.975))]
    out = {m: {k: q(v) for k, v in st.items()} for m, st in stats.items()}
    if gap:
        out["auroc_gap_ours_minus_published"] = {"ci": q(gap), "p_gap_le_0": float(np.mean(np.array(gap) <= 0))}
    return out


SPLITS = int(__import__('os').environ.get('PT_SPLITS', 200))


def scene_units(data, cond, fn):
    """Calibration units = scenes (the two mirrored pairs of a scene are not independent), over
    every non-train image. Each unit keeps its sorted wrong-pixel scores, its sorted scores and a
    small score sample for the threshold grid."""
    by = {}
    for (i, c), d in sorted(data.items()):
        if c != cond or str(d["split"]) == "train":
            continue
        g = i // 2 if str(d.get("dataset", "")) == "middlebury" else i
        e, sc = d["err"].astype(np.float32).ravel(), fn(d)
        by.setdefault(g, []).append((e, sc))
    units = []
    rng = np.random.default_rng(1)
    for g, lst in sorted(by.items()):
        e = np.concatenate([x for x, _ in lst]); sc = np.concatenate([y for _, y in lst])
        units.append({"wrong": np.sort(sc[e > A.EPS]), "all": np.sort(sc),
                      "sample": sc[rng.choice(len(sc), min(3000, len(sc)), replace=False)]})
    return units


def _unit_miss(u, lam):
    n = len(u["wrong"])
    return np.searchsorted(u["wrong"], lam, side="right") / n if n else np.zeros_like(np.atleast_1d(lam), float)


def split_eval(units, n_cal, rng, fixed_lam=None):
    """Random scene-level calibration/test splits. Conformal risk control promises the miss rate
    is <= alpha *on average over such splits*; a single fixed split can land above it.
    With fixed_lam (e.g. the KITTI threshold) no calibration happens and only the test draw varies."""
    G = len(units)
    miss, kept, lams = [], [], []
    for _ in range(SPLITS):
        perm = rng.permutation(G)
        cal, test = perm[:n_cal], perm[n_cal:]
        if fixed_lam is None:
            grid = np.unique(np.quantile(np.concatenate([units[j]["sample"] for j in cal]), np.linspace(0, 1, 2000)))
            risk = np.mean([_unit_miss(units[j], grid) for j in cal], axis=0)
            ok = n_cal / (n_cal + 1) * risk + 1 / (n_cal + 1) <= ALPHA
            lam = float(grid[np.flatnonzero(ok).max()]) if ok.any() else float(grid[0]) - 1.0
        else:
            lam = fixed_lam
        lams.append(lam)
        miss.append(float(np.mean([_unit_miss(units[j], lam) for j in test])))
        kept.append(float(np.mean([np.searchsorted(units[j]["all"], lam, side="right") / len(units[j]["all"]) for j in test])))
    miss, kept = np.array(miss), np.array(kept)
    return {"n_scenes": G, "n_cal": int(n_cal) if fixed_lam is None else 0, "splits": SPLITS,
            "miss_mean": float(miss.mean()), "miss_p90": float(np.quantile(miss, 0.9)),
            "violation_rate": float((miss > ALPHA).mean()), "kept_mean": float(kept.mean())}


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
    per = {"clean": {}, "corrupt": {}}
    per_in = {"clean": {}, "corrupt": {}}
    methods_for_cache = methods_for(src, tgt)
    for name, f in methods_for_cache.items():
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
            per[cond][name] = {"items": test_src_model, "lam": lam_zero}
            per_in[cond][name] = {"items": test_tgt_model, "lam": lam_in}
            ev_in = A.evaluate_scores(tgt, "test", cond, f["tgt"])
            ev_tr = A.evaluate_scores(tgt, "test", cond, f["src"])
            res["methods"][name][cond] = {
                "in_domain": {"auroc": ev_in["auroc"], "ause": ev_in["ause"], "miss": mi, "coverage": ci},
                "zero_shot": {"auroc": ev_tr["auroc"], "ause": ev_tr["ause"], "miss": mz, "coverage": cz},
                "recalibrated": recal_curve(cal_tgt_src_model, test_src_model, ns, rng),
            }
            print(f"[{label}] {name:28s} {cond:7s} in-domain AUROC {ev_in['auroc']:.3f} miss {mi:.3f} kept {ci:.3f} | "
                  f"zero-shot AUROC {ev_tr['auroc']:.3f} miss {mz:.3f} kept {cz:.3f}", flush=True)
    res["bootstrap_zero_shot"] = {cond: bootstrap(per[cond], test_groups(tgt, cond), rng) for cond in per}
    res["bootstrap_in_domain"] = {cond: bootstrap(per_in[cond], test_groups(tgt, cond), rng) for cond in per_in}
    # guarantee over random scene-level splits of all non-train target scenes
    res["splits"] = {}
    for name, f in methods_for_cache.items():
        res["splits"][name] = {}
        for cond in ("clean", "corrupt"):
            u_tgt = scene_units(tgt, cond, f["tgt"])
            u_src = scene_units(tgt, cond, f["src"])
            G = len(u_tgt)
            n_cal = max(10, G // 2)
            cal_src = items(src, "calib", cond, f["src"])
            lam_zero = A.crc_threshold(cal_src, ALPHA, grid_of(cal_src))
            res["splits"][name][cond] = {
                "in_domain": split_eval(u_tgt, n_cal, rng),
                "recalibrated": split_eval(u_src, n_cal, rng),
                "zero_shot": split_eval(u_src, 0, rng, fixed_lam=lam_zero),
            }
            r = res["splits"][name][cond]
            print(f"[{label}] splits {name:28s} {cond:7s} in-domain miss {r['in_domain']['miss_mean']:.3f} "
                  f"(viol {r['in_domain']['violation_rate']:.2f}, kept {r['in_domain']['kept_mean']:.3f}) | recal "
                  f"{r['recalibrated']['miss_mean']:.3f} | zero-shot {r['zero_shot']['miss_mean']:.3f}", flush=True)
    print(f"[{label}] bootstrap:", json.dumps(res["bootstrap_zero_shot"]["clean"])[:1500], flush=True)
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
            for key, title in (("bootstrap_in_domain", "In-domain (Middlebury-trained and calibrated)"),
                               ("bootstrap_zero_shot", "Zero-shot (KITTI-trained and calibrated)")):
              b = r.get(key, {}).get(cond)
              if b:
                L += ["", f"{title}, 95% intervals over Middlebury test scenes ({BOOT} resamples):", "",
                      "| method | AUROC | miss | kept |", "|---|---|---|---|"]
                for m, v in b.items():
                    if m.startswith("auroc_gap"):
                        continue
                    L.append(f"| {m} | {v['auroc'][0]:.3f}–{v['auroc'][1]:.3f} | {v['miss'][0]:.1%}–{v['miss'][1]:.1%} | "
                             f"{v['kept'][0]:.1%}–{v['kept'][1]:.1%} |")
                if "auroc_gap_ours_minus_published" in b:
                    g = b["auroc_gap_ours_minus_published"]
                    L.append(f"\nAUROC gap, ours minus published combined: {g['ci'][0]:+.3f} to {g['ci'][1]:+.3f} "
                             f"(share of resamples with gap <= 0: {g['p_gap_le_0']:.1%})")
            sp = r.get("splits")
            if sp:
                any_m = next(iter(sp.values()))[cond]
                L += ["", f"Guarantee over {any_m['in_domain']['splits']} random scene-level splits of the "
                      f"{any_m['in_domain']['n_scenes']} non-train scenes ({any_m['in_domain']['n_cal']} scenes calibrate, "
                      "the rest test; calibration unit = scene):", "",
                      "| method | setting | mean miss | 90th pct miss | P(miss > α) | kept |", "|---|---|---|---|---|---|"]
                for m, v in sp.items():
                    for key, lab2 in (("in_domain", "in-domain"), ("recalibrated", "KITTI model, recalibrated"),
                                      ("zero_shot", "KITTI model + KITTI threshold")):
                        x = v[cond][key]
                        L.append(f"| {m} | {lab2} | {x['miss_mean']:.1%} | {x['miss_p90']:.1%} | {x['violation_rate']:.0%} | {x['kept_mean']:.1%} |")
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
