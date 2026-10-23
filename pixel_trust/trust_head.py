"""Learned spatial trust head (v5).

The v1-v4 combiner (gradient-boosted trees) scores every pixel from its own nine signal values.
Wrong pixels are spatially structured: they come in blobs at disocclusion borders, along depth
edges and inside corrupted depth regions. The trust head is a small fully-convolutional U-Net
that reads the signal *maps* and predicts a per-pixel distrust logit, so a pixel's score can use
its neighbourhood. It is trained with a class-balanced binary cross-entropy on wrong / right
labels and calibrated afterwards exactly like every other score (conformal risk control).

Optional generator conditioning: a one-hot generator id is appended as constant channels, so a
single head can be trained on several generators and still learn generator-specific weighting.
"""
from __future__ import annotations

import numpy as np

EPS_WRONG = 0.08


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_net(in_ch: int, width: int = 32):
    torch, nn = _torch()

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
                             nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            w = width
            self.e1, self.e2, self.e3 = block(in_ch, w), block(w, 2 * w), block(2 * w, 4 * w)
            self.pool = nn.MaxPool2d(2)
            self.up2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
            self.d2 = block(4 * w, 2 * w)
            self.up1 = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
            self.d1 = block(2 * w, w)
            self.out = nn.Conv2d(w, 1, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            d2 = self.d2(torch.cat([self.up2(e3), e2], 1))
            d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
            return self.out(d1)[:, 0]

    return UNet()


class Normalizer:
    """Per-channel robust standardisation fitted on training images (after signed log1p)."""

    def __init__(self, names):
        self.names = list(names)
        self.mu = self.sd = None

    @staticmethod
    def _tf(x):
        return np.sign(x) * np.log1p(np.abs(x))

    def fit(self, records):
        rng = np.random.default_rng(0)
        cols = []
        for d in records:
            x = self.stack_raw(d).reshape(len(self.names), -1)
            cols.append(x[:, rng.choice(x.shape[1], min(4000, x.shape[1]), replace=False)])
        x = np.concatenate(cols, 1)
        lo, hi = np.quantile(x, 0.01, axis=1, keepdims=True), np.quantile(x, 0.99, axis=1, keepdims=True)
        x = np.clip(x, lo, hi)
        self.lo, self.hi = lo[:, :, None], hi[:, :, None]
        self.mu, self.sd = x.mean(1)[:, None, None], x.std(1)[:, None, None] + 1e-6
        return self

    def stack_raw(self, d):
        x = np.stack([np.nan_to_num(d[f"f_{n}"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                      for n in self.names])
        return self._tf(x)

    def __call__(self, d):
        x = np.clip(self.stack_raw(d), self.lo, self.hi)
        return ((x - self.mu) / self.sd).astype(np.float32)


class TrustHead:
    """Fit on a list of records (dicts from analysis.load_all); score(d) -> flat distrust map."""

    def __init__(self, names, n_gen: int = 0, width: int = 32, device=None, iters: int = 3000,
                 batch: int = 16, crop: int = 96, lr: float = 1e-3, seed: int = 0, log=print):
        torch, _ = _torch()
        self.names, self.n_gen, self.width = list(names), n_gen, width
        self.iters, self.batch, self.crop, self.lr, self.seed, self.log = iters, batch, crop, lr, seed, log
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.norm = Normalizer(names)
        self.net = None

    def _input(self, d, gen_id=None):
        x = self.norm(d)
        if self.n_gen:
            g = np.zeros((self.n_gen,) + x.shape[1:], np.float32)
            if gen_id is not None:
                g[gen_id] = 1.0
            x = np.concatenate([x, g])
        return x

    def fit(self, records, gen_ids=None):
        torch, nn = _torch()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.norm.fit(records)
        X = [self._input(d, None if gen_ids is None else gen_ids[i]) for i, d in enumerate(records)]
        Y = [(d["err"].astype(np.float32) > EPS_WRONG).astype(np.float32) for d in records]
        pos = float(np.mean([y.mean() for y in Y]))
        self.net = build_net(X[0].shape[0], self.width).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=self.iters)
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-3), device=self.device))
        c = self.crop
        self.net.train()
        for it in range(self.iters):
            xb, yb = [], []
            for _ in range(self.batch):
                j = rng.integers(len(X))
                H, W = Y[j].shape
                y0, x0 = rng.integers(0, max(1, H - c + 1)), rng.integers(0, max(1, W - c + 1))
                xs, ys = X[j][:, y0:y0 + c, x0:x0 + c], Y[j][y0:y0 + c, x0:x0 + c]
                if ys.shape != (c, c):   # narrow frames: pad the crop
                    ph, pw = c - ys.shape[0], c - ys.shape[1]
                    xs = np.pad(xs, ((0, 0), (0, ph), (0, pw)), mode="edge")
                    ys = np.pad(ys, ((0, ph), (0, pw)), mode="edge")
                if rng.random() < 0.5:
                    xs, ys = xs[:, :, ::-1], ys[:, ::-1]
                xb.append(np.ascontiguousarray(xs)); yb.append(np.ascontiguousarray(ys))
            xb = torch.from_numpy(np.stack(xb)).to(self.device)
            yb = torch.from_numpy(np.stack(yb)).to(self.device)
            loss = bce(self.net(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if (it + 1) % max(1, self.iters // 5) == 0:
                self.log(f"  trust head it {it + 1}/{self.iters} loss {loss.item():.4f}")
        self.net.eval()
        return self

    def score(self, d, gen_id=None):
        torch, _ = _torch()
        x = self._input(d, gen_id)
        H, W = x.shape[1:]
        ph, pw = (-H) % 4, (-W) % 4
        x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="edge")
        with torch.no_grad():
            s = torch.sigmoid(self.net(torch.from_numpy(x[None]).to(self.device)))[0, :H, :W]
        # tiny deterministic tie-break so plateaus still rank (as for the raw-signal scorers)
        return s.float().cpu().numpy().ravel() + 1e-6 * x[0, :H, :W].ravel()

    def scorer(self, gen_id=None):
        return lambda d: self.score(d, gen_id)

    def n_params(self):
        return sum(p.numel() for p in self.net.parameters())
