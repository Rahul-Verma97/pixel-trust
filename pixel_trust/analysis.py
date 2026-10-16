"""Stage 2 (CPU): evaluate trust signals, calibrate with conformal risk control, run the T2/T3 analyses."""
from __future__ import annotations

import glob
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .pipeline import GROUPS

EPS = 0.08               # a pixel is "wrong" if its blurred colour error exceeds 8% of the range
FRACS = np.linspace(0, 0.95, 20)


def _trapz(y, x):
    return ((y[:, 1:] + y[:, :-1]) / 2 * np.diff(x)).sum(1)


def load_all(out_dir: str) -> dict:
    data = {}
    for f in sorted(glob.glob(os.path.join(out_dir, "feats", "*.npz"))):
        name = os.path.basename(f)[:-4]
        idx, cond = name.split("_")
        z = np.load(f)
        data[(int(idx), cond)] = {k: z[k] for k in z.files}
    return data


def feat_matrix(d: dict, names: list[str], var_key: str = "var") -> np.ndarray:
    cols = [d[f"f_{var_key if n == 'var' else n}"].astype(np.float32).ravel() for n in names]
    return np.stack(cols, 1)


def train_combiner(data: dict, names: list[str], var_key="var", per_img=4000, seed=0):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for (idx, cond), d in data.items():
        if str(d["split"]) != "train":
            continue
        x = feat_matrix(d, names, var_key)
        y = (d["err"].astype(np.float32).ravel() > EPS)
        sel = rng.choice(len(y), size=min(per_img, len(y)), replace=False)
        X.append(x[sel]); Y.append(y[sel])
    X, Y = np.concatenate(X), np.concatenate(Y)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, random_state=seed)
    clf.fit(X, Y)
    return clf


def scorer(kind: str, clf=None, names=None, var_key="var"):
    """Return f(d) -> per-pixel uncertainty score (higher = less trusted)."""
    if kind == "model":
        return lambda d: clf.predict_proba(feat_matrix(d, names, var_key))[:, 1]
    if kind == "oracle":
        return lambda d: d["err"].astype(np.float32).ravel()
    if kind == "random":
        return lambda d: np.random.default_rng(0).random(d["err"].size)
    # single raw feature; tiny tie-break so binary masks still rank
    return lambda d: d[f"f_{kind}"].astype(np.float32).ravel() + 1e-4 * d["f_tex"].astype(np.float32).ravel()


def sparsification(err: np.ndarray, score: np.ndarray) -> np.ndarray:
    order = np.argsort(-score, kind="stable")
    e = err[order]
    n = len(e)
    csum = np.cumsum(e[::-1])[::-1]  # sum of errors from position i to end
    out = []
    for f in FRACS:
        k = int(f * n)
        out.append(csum[k] / (n - k))
    return np.array(out)


def evaluate_scores(data: dict, split: str, cond: str, fn) -> dict:
    curves, oracle, ys, ss, per_img = [], [], [], [], []
    for (idx, c), d in data.items():
        if c != cond or str(d["split"]) != split:
            continue
        err = d["err"].astype(np.float32).ravel()
        s = fn(d)
        curves.append(sparsification(err, s))
        oracle.append(sparsification(err, err))
        ys.append(err > EPS); ss.append(s)
        per_img.append((idx, d, s))
    curves, oracle = np.array(curves), np.array(oracle)
    y, s = np.concatenate(ys), np.concatenate(ss)
    ause = float(np.mean(_trapz(curves - oracle, FRACS) / FRACS[-1]))
    return {"curve": curves.mean(0), "oracle": oracle.mean(0), "ause": ause,
            "auroc": float(roc_auc_score(y, s)) if y.any() and (~y).any() else float("nan"),
            "err_rate": float(y.mean()), "per_img": per_img}


# ---------------- conformal risk control ----------------
def _prep(e, s):
    return np.sort(s[e > EPS]), (e > EPS).sum()


def _miss_fast(prep, lams):
    srt, ny = prep
    if ny == 0:
        return np.zeros_like(lams)
    return np.searchsorted(srt, lams, side="right") / ny


def crc_threshold(cal: list, alpha: float, grid: np.ndarray) -> float:
    """Largest threshold whose corrected calibration risk stays <= alpha (Angelopoulos et al. 2024)."""
    n = len(cal)
    risk = np.mean([_miss_fast(_prep(e, s), grid) for e, s in cal], axis=0)  # non-decreasing in lambda
    ok = n / (n + 1) * risk + 1 / (n + 1) <= alpha
    if not ok.any():
        return float(grid[0]) - 1.0   # accept nothing
    return float(grid[np.flatnonzero(ok).max()])


def crc_evaluate(data: dict, cond: str, fn, alphas=(0.1, 0.2)) -> dict:
    cal, test = [], []
    for (idx, c), d in data.items():
        if c != cond:
            continue
        sp = str(d["split"])
        if sp in ("calib", "test"):
            item = (d["err"].astype(np.float32).ravel(), fn(d), d)
            (cal if sp == "calib" else test).append(item)
    allS = np.concatenate([s for _, s, _ in cal])
    grid = np.unique(np.quantile(allS, np.linspace(0, 1, 1000)))
    res = {}
    for a in alphas:
        lam = crc_threshold([(e, s) for e, s, _ in cal], a, grid)
        miss, cov, prec, cmiss = [], [], [], []
        for e, s, d in test:
            y = e > EPS
            acc = s <= lam
            miss.append(float((acc & y).sum() / max(1, y.sum())) if y.any() else 0.0)
            cov.append(float(acc.mean()))
            prec.append(float((acc & y).sum() / max(1, acc.sum())))
            m = d["corrupt"].ravel() & y
            if m.sum() > 20:
                cmiss.append(float((acc & m).sum() / m.sum()))
        res[a] = {"lambda": lam, "miss": float(np.mean(miss)), "coverage": float(np.mean(cov)),
                  "wrong_among_accepted": float(np.mean(prec)),
                  "miss_in_corrupt": float(np.mean(cmiss)) if cmiss else None}
    return res


def warp_mask_policy(data: dict, cond: str) -> dict:
    """Naive policy: trust every warped pixel, distrust every generated one."""
    miss, cov, prec = [], [], []
    for (idx, c), d in data.items():
        if c != cond or str(d["split"]) != "test":
            continue
        e = d["err"].astype(np.float32).ravel()
        acc = ~d["mask"].ravel()
        y = e > EPS
        miss.append((acc & y).sum() / max(1, y.sum()))
        cov.append(acc.mean())
        prec.append((acc & y).sum() / max(1, acc.sum()))
    return {"miss": float(np.mean(miss)), "coverage": float(np.mean(cov)), "wrong_among_accepted": float(np.mean(prec))}


def regen_table(data: dict) -> dict:
    keys = sorted({k[len("regen_err_"):] for d in data.values() for k in d if k.startswith("regen_err_")})
    rows = {"clean depth": [], "corrupted depth, no regeneration": []}
    rows.update({f"regenerate ({k})": [] for k in keys})
    for (idx, c), d in data.items():
        if c != "corrupt" or str(d["split"]) != "test":
            continue
        cm = d["corrupt"].astype(bool)
        if cm.sum() < 50:
            continue
        clean = data.get((idx, "clean"))
        if clean is not None:
            rows["clean depth"].append(_err_stats(clean["err"], cm))
        rows["corrupted depth, no regeneration"].append(_err_stats(d["err"], cm))
        for k in keys:
            rows[f"regenerate ({k})"].append(_err_stats(d[f"regen_err_{k}"], cm))
    return {k: {"mae_all": float(np.mean([r[0] for r in v])), "wrong_all": float(np.mean([r[1] for r in v])),
                "mae_corrupt": float(np.mean([r[2] for r in v])), "wrong_corrupt": float(np.mean([r[3] for r in v])),
                "n": len(v)} for k, v in rows.items() if v}


def _err_stats(err, cm):
    e = err.astype(np.float32)
    return e.mean(), (e > EPS).mean(), e[cm].mean(), (e[cm] > EPS).mean()
