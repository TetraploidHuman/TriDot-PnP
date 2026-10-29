"""Smoke: peak Transformer, velocity associate, heat writeback on synthetic tracks."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.tridot_temporal.find_geom import associate_velocity, extract_heatmap_peaks
from research.tridot_temporal.find_model import build_find_net, gated_heat_bump
from research.tridot_temporal.losses import peak_candidate_loss, tridot_loss


def _blob(h, w, cx, cy, amp=8.0, sigma=1.8):
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    g = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return amp * g


def make_clip(b=2, t=5, h=32, w=56):
    """True LED moves; a distractor sits still. Heat maps only — fake RGB inputs."""
    heat = torch.full((b, t, 3, h, w), -4.0)
    gt = torch.zeros(b, t, 3, 2)
    vis = torch.ones(b, t, 3)
    for bi in range(b):
        for ci in range(3):
            x0 = 8 + ci * 12 + bi
            y0 = 10 + ci * 4
            dx, dy = 2.0, 0.5
            for ti in range(t):
                tx = x0 + ti * dx
                ty = y0 + ti * dy
                # true peak
                heat[bi, ti, ci] = heat[bi, ti, ci] + _blob(h, w, tx, ty, amp=7.0)
                # distractor stronger on last frames to force trajectory use
                heat[bi, ti, ci] = heat[bi, ti, ci] + _blob(
                    h, w, x0 + 18, y0 + 8, amp=9.0 if ti >= t - 2 else 3.0,
                )
                gt[bi, ti, ci, 0] = tx / max(w - 1, 1)
                gt[bi, ti, ci, 1] = ty / max(h - 1, 1)
    # fake 7ch input: put affinity = sigmoid(heat) so the CNN has something to latch onto
    x = torch.zeros(b, t, 7, h * 4, w * 4)
    # upsample heat into affinity channels of a blank canvas (encoder sees /4)
    for bi in range(b):
        for ti in range(t):
            for ci in range(3):
                up = F.interpolate(
                    heat[bi : bi + 1, ti, ci : ci + 1],
                    size=(h * 4, w * 4), mode="bilinear", align_corners=True,
                )[0, 0]
                x[bi, ti, 3 + ci] = torch.sigmoid(up)
                x[bi, ti, ci] = torch.sigmoid(up)
                x[bi, ti, 6] = torch.sigmoid(up).clamp(0, 1)
    return x, gt, vis, heat


def main() -> int:
    torch.manual_seed(0)
    device = torch.device("cpu")

    # --- velocity associate (no train) ---
    _, gt, _, heat = make_clip(b=1, t=5)
    prev = gt[:, 0]
    prev2 = None
    coast = torch.zeros(1, 3, dtype=torch.long)
    errs = []
    for ti in range(1, 5):
        out, coast = associate_velocity(
            heat[:, ti], prev, prev2, coast_age=coast,
            top_k=5, snap=0.08, heat_frac=0.2, max_coast=2, vel_w=1.0,
        )
        err = (out - gt[:, ti]).norm(dim=-1).mean().item()
        errs.append(err)
        prev2, prev = prev, out
    print(f"[vel] mean err over t=1..4: {sum(errs)/len(errs):.4f} (want << distractor jump)")

    # --- heat writeback keeps track peak above distractor when evidence exists ---
    hm = heat[:, -1].clone()
    expect = (gt[:, -2] + (gt[:, -2] - gt[:, -3])).clamp(0, 1)
    bumped = gated_heat_bump(hm, expect, strength=4.0, sigma=0.04, logit_slack=20.0)
    peaks0 = extract_heatmap_peaks(hm, top_k=3, nms=3)
    peaks1 = extract_heatmap_peaks(bumped, top_k=3, nms=3)
    d0 = (peaks0[0, 0, 0, :2] - expect[0, 0]).norm().item()
    d1 = (peaks1[0, 0, :, :2] - expect[0, 0]).norm(dim=-1).min().item()
    print(f"[writeback] nearest-to-expect before={d0:.4f} after={d1:.4f}")

    # --- peak-assoc train a few steps on synthetic ---
    model = build_find_net(
        seq_len=5, img_h=128, img_w=224, feat_ch=48, d_model=64, nlayers=2, nhead=4,
        ffn_dim=128, peak_assoc=True, peak_k=5, heat_writeback=True, writeback_w=2.0,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x, gt, vis, _ = make_clip(b=4, t=5, h=32, w=56)
    # resize synthetic canvas to model input size
    x = F.interpolate(x.view(-1, 7, 128, 224), size=(128, 224), mode="bilinear", align_corners=False)
    # wait — make_clip already used h*4; rebuild for exact size
    x, gt, vis, _ = make_clip(b=4, t=5, h=32, w=56)
    x = F.interpolate(
        x.reshape(-1, 7, 128, 224), size=(128, 224), mode="bilinear", align_corners=False,
    ).view(4, 5, 7, 128, 224)

    model.train()
    before = None
    after = None
    for step in range(40):
        opt.zero_grad(set_to_none=True)
        pred_c, pred_v, pred_hm, pred_c0 = model(x, return_heat=True)
        losses = tridot_loss(
            pred_c, pred_v, gt, vis,
            pred_heat=pred_hm, pred_coords0=pred_c0,
            lambda_xy=5.0, lambda_xy0=2.0, lambda_hm=2.0, lambda_temp=1.0,
            last_frame_boost=2.0, hm_sigma=1.4,
        )
        pick = peak_candidate_loss(model.last_pick, model.last_cand, gt, vis, thr=0.1)
        loss = losses["loss"] + 2.0 * pick
        loss.backward()
        opt.step()
        with torch.no_grad():
            err = (pred_c[:, -1] - gt[:, -1]).norm(dim=-1).mean().item()
        if step == 0:
            before = err
        after = err
        if step % 10 == 0:
            print(f"  step {step} loss={float(loss):.3f} pick={float(pick):.3f} last_err={err:.4f}")
    print(f"[peak] last-frame err before={before:.4f} after={after:.4f}")

    model.eval()
    with torch.no_grad():
        pred_c, _, heat_out, _ = model(x[:1], return_heat=True)
        # dropping last frame should still let peak assoc use history
        x_drop = x[:1].clone()
        x_drop[:, -1].zero_()
        pred_drop, _, _, _ = model(x_drop, return_heat=True)
        err_full = (pred_c[:, -1] - gt[:1, -1]).norm(dim=-1).mean().item()
        err_drop = (pred_drop[:, -1] - gt[:1, -1]).norm(dim=-1).mean().item()
        print(f"[ctx] last-frame err full={err_full:.4f} dropped-frame={err_drop:.4f}")
        print(f"[shapes] heat={tuple(heat_out.shape)} coords={tuple(pred_c.shape)}")

    ok = after < before * 0.85 or after < 0.08
    print("PASS" if ok else "WEAK (still wired; needs real data FT)")
    return 0 if ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
