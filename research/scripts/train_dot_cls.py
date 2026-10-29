"""Train the tiny dot-vs-clutter patch classifier."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from research.tridot_temporal.dot_cls import DotClassifier


def _split(x, y, seed, val_frac=0.15):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(x))
    n_val = int(len(x) * val_frac)
    val, tr = idx[:n_val], idx[n_val:]
    return x[tr], y[tr], x[val], y[val]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    d = np.load(a.data)
    x = d["x"].astype(np.float32)
    y = d["y"].astype(np.float32)
    if x.ndim == 3:
        x = x[:, None]
    xtr, ytr, xv, yv = _split(x, y, a.seed)
    xtr_t = torch.from_numpy(xtr); ytr_t = torch.from_numpy(ytr)
    xv_t = torch.from_numpy(xv); yv_t = torch.from_numpy(yv)
    model = DotClassifier(in_ch=xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    n = len(xtr_t)
    best = {"loss": 9e9, "acc": 0.0}
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, a.batch):
            idx = perm[i:i + a.batch]
            opt.zero_grad()
            logits = model(xtr_t[idx])
            loss = F.binary_cross_entropy_with_logits(logits, ytr_t[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            lv = model(xv_t)
            vloss = float(F.binary_cross_entropy_with_logits(lv, yv_t))
            pred = (torch.sigmoid(lv) >= 0.5).float()
            acc = float((pred == yv_t).float().mean())
        print(f"ep {ep}/{a.epochs} train_loss {tot/n:.4f} val_loss {vloss:.4f} "
              f"val_acc {acc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if acc > best["acc"] or (acc == best["acc"] and vloss < best["loss"]):
            best = {"loss": vloss, "acc": acc, "epoch": ep}
            a.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "in_ch": xtr.shape[1],
                        "threshold": 0.5,
                        "meta": {"val_acc": acc, "val_loss": vloss, "epoch": ep,
                                 "seed": a.seed, "n_train": int(len(xtr)),
                                 "n_val": int(len(xv)),
                                 "data": str(a.data)}}, a.out)
    print(json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
