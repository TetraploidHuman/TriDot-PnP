"""Train pure learned point finder (preprocess OK, no classical detection)."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import random
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from research.tridot_temporal.find_dataset import FindClipDataset
from research.tridot_temporal.find_model import build_find_net
from research.tridot_temporal.losses import peak_candidate_loss, tridot_loss
from research.tridot_temporal.find_offset import decode_offset, offset_loss

ROOT = Path(__file__).resolve().parents[2]

_WORKER_SEED = 0


def _seed_worker(_worker_id: int) -> None:
    """Give each DataLoader worker a deterministic, distinct RNG state.

    ``FindClipDataset`` augments with ``np.random`` (brightness/additive/noise/
    zero-rect), so without this the augmentation stream differs run to run and
    two runs of the same config are not comparable.
    """
    seed = _WORKER_SEED + _worker_id
    random.seed(seed)
    np.random.seed(seed)


def _sha256(path: Path) -> str | None:
    if not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _seed_everything(seed: int, deterministic: bool = False) -> None:
    global _WORKER_SEED
    _WORKER_SEED = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _dataset_fingerprint(dataset: Path) -> dict:
    """Hash whatever defines *which* data this run saw."""
    meta, synth = dataset / "meta", dataset / "synthetic"
    out = {}
    for name in ("splits.json", "dataset_info.json"):
        p = meta / name
        out[f"meta/{name}"] = {"sha256": _sha256(p),
                               "mtime": p.stat().st_mtime if p.exists() else None}
    for name in ("index.json", "index_realistic.json", "index_realistic_train.json",
                 "index_realistic_train_more.json", "index_realistic_train_tail.json"):
        p = synth / name
        out[f"synthetic/{name}"] = {"sha256": _sha256(p),
                                    "mtime": p.stat().st_mtime if p.exists() else None}
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    """coord_l2 on fully visible LEDs; peak_hit on partial (vis=0.5) Top-1 peaks."""
    from research.tridot_temporal.find_geom import extract_heatmap_peaks

    model.eval()
    total_xy, total_n = 0.0, 0
    hit_p, n_p = 0.0, 0
    for batch in loader:
        x = batch["image"].to(device)
        gt_c = batch["coords"].to(device)
        gt_v = batch["vis"].to(device)
        pred_c, _, heat, _ = model(x, return_heat=True)
        if getattr(model, "last_off", None) is not None:
            pc = decode_offset(heat[:, -1], model.last_off[:, -1], top_k=1, nms=5)
        else:
            pc = pred_c[:, -1]
        gc, gv = gt_c[:, -1], gt_v[:, -1]
        mask = gv > 0.9
        if mask.any():
            err = torch.linalg.norm(pc - gc, dim=-1)
            total_xy += err[mask].sum().item()
            total_n += int(mask.sum().item())
        partial = (gv > 0.4) & (gv < 0.9)
        if partial.any():
            peaks = extract_heatmap_peaks(heat[:, -1], top_k=1, nms=5)
            top = peaks[:, :, 0, :2]
            d = torch.linalg.norm(top - gc, dim=-1)
            hit_p += ((d <= 0.02) & partial).sum().item()
            n_p += int(partial.sum().item())
    return {
        "coord_l2": total_xy / max(total_n, 1),
        "n": total_n,
        "partial_hit": hit_p / max(n_p, 1),
        "n_partial": n_p,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "TriDotPose")
    p.add_argument("--out", type=Path, default=ROOT / "research" / "results" / "find_v3")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=5)
    p.add_argument("--img-h", type=int, default=256)
    p.add_argument("--img-w", type=int, default=448)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-clips", type=int, default=16)
    p.add_argument("--lambda-xy", type=float, default=8.0)
    p.add_argument("--lambda-xy0", type=float, default=20.0,
                   help="supervise soft-argmax coords from heat (anti-collapse)")
    p.add_argument("--lambda-hm", type=float, default=10.0)
    p.add_argument("--lambda-peak", type=float, default=0.0,
                   help="peak-margin loss: GT heat must beat strongest false peak (TopK recall)")
    p.add_argument("--peak-margin", type=float, default=0.5,
                   help="logit margin for --lambda-peak")
    p.add_argument("--hm-sigma", type=float, default=1.4,
                   help="Gaussian GT heatmap sigma for every sample")
    p.add_argument("--peak-exclude", type=int, default=3,
                   help="pixels around GT ignored when measuring the false peak")
    p.add_argument("--lambda-sep", type=float, default=2.0)
    p.add_argument("--lambda-eq", type=float, default=0.15,
                   help="equilateral penalty — teach triangle geometry in coords")
    p.add_argument("--lambda-temp", type=float, default=0.0,
                   help="supervise inter-frame motion (pred Δ vs gt Δ)")
    p.add_argument("--offset-head", action="store_true", default=False,
                   help="add per-color 2-D sub-cell offset head (supervised by --lambda-off)")
    p.add_argument("--lambda-off", type=float, default=2.0,
                   help="weight of the offset-head L1 loss")
    p.add_argument("--max-train-clips", type=int, default=0,
                   help="subsample the training clip list (0 = off); for CPU-sized runs")
    p.add_argument("--max-train-seqs", type=int, default=0,
                   help="subsample distinct training sequences (0 = off)")
    p.add_argument("--max-val-clips", type=int, default=0,
                   help="subsample the validation clip list (0 = off)")
    p.add_argument("--max-val-seqs", type=int, default=0,
                   help="subsample distinct validation sequences (0 = off)")
    p.add_argument("--no-shuffle", dest="shuffle", action="store_false", default=True,
                   help="keep clip order (fewer sequence reloads on CPU)")
    p.add_argument("--no-disk-cache", dest="disk_cache", action="store_false",
                   default=True,
                   help="do not write/read datasets/.../find_cache npz (saves disk)")
    p.add_argument("--xy-boost-after", type=int, default=40)
    p.add_argument("--xy-boost", type=float, default=2.0)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--reset-best", action="store_true", default=False,
                   help="ignore resumed val_l2 when selecting best.pt (domain FT)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--train-subset", default="train",
                   help="splits.json keys, comma-ok: train,hard_train")
    p.add_argument("--val-subset", default="val",
                   help="splits.json keys, comma-ok: val,hard_val")
    p.add_argument("--cache-videos", action="store_true", default=False,
                   help="preload all sequences into RAM (use when find_cache fits in memory)")
    p.add_argument("--num-workers", type=int, default=-1,
                   help="DataLoader workers; -1 = auto (4 on cuda if cache warm, else 0)")
    p.add_argument("--lru-max", type=int, default=256,
                   help="in-memory sequence LRU when --cache-videos is off")
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--amp", action="store_true", default=False,
                   help="CUDA automatic mixed precision (faster on modern GPUs)")
    p.add_argument("--frame-drop-p", type=float, default=0.0,
                   help="train-time last-frame occlusion dropout (forces trajectory use)")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--ffn-dim", type=int, default=256)
    p.add_argument("--feat-ch", type=int, default=72)
    p.add_argument("--partial-init", action="store_true", default=False,
                   help="load matching-shaped tensors from --resume (for scaled models)")
    p.add_argument("--freeze-backbone", action="store_true", default=False,
                   help="freeze encoder+heat_head+aff_to_heat; train fuse/temporal/refine only")
    p.add_argument("--freeze-temporal", action="store_true", default=False,
                   help="freeze temporal/refine/token; train encoder+heat only (TopK heat FT)")
    p.add_argument("--heat-fuse", action="store_true", default=False,
                   help="causal cross-frame heatmap fusion (real trajectory in heat space)")
    p.add_argument("--no-conf-gate", action="store_true", default=False,
                   help="disable confidence gating inside CausalHeatFuse")
    p.add_argument("--refine-scale", type=float, default=0.06,
                   help="temporal token residual scale on coords")
    p.add_argument("--peak-assoc", action="store_true", default=False,
                   help="causal Transformer selects a Top-K peak using history")
    p.add_argument("--peak-k", type=int, default=5)
    p.add_argument("--lambda-pick", type=float, default=1.0,
                   help="CE on the peak nearest to GT (used with --peak-assoc)")
    p.add_argument("--heat-writeback", action="store_true", default=False,
                   help="bump returned heat at the velocity expect when heat is already there")
    p.add_argument("--writeback-w", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0,
                   help="seed for python/numpy/torch + DataLoader workers")
    p.add_argument("--deterministic", action="store_true", default=False,
                   help="also force cuDNN deterministic kernels (slower)")
    args = p.parse_args(argv)
    if args.peak_assoc and args.frame_drop_p <= 0:
        args.frame_drop_p = 0.35
    if args.peak_assoc and args.lambda_temp <= 0:
        args.lambda_temp = 1.0
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    _seed_everything(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)

    train_ds = FindClipDataset(
        args.dataset, args.train_subset, args.seq_len, args.img_h, args.img_w, True,
        args.max_clips, cache_videos=args.cache_videos, lru_max=args.lru_max,
        frame_drop_p=args.frame_drop_p, disk_cache=args.disk_cache,
    )
    if args.max_train_seqs and len(train_ds.anns) > args.max_train_seqs:
        rng = random.Random(args.seed)
        keep = set(rng.sample(sorted(train_ds.anns), args.max_train_seqs))
        train_ds.clips = [c for c in train_ds.clips if c.ann_path.stem in keep]
        train_ds.anns = {k: v for k, v in train_ds.anns.items() if k in keep}
        print(f"[find] subsampled train seqs -> {len(train_ds.anns)}", flush=True)
    if args.max_train_clips and len(train_ds.clips) > args.max_train_clips:
        rng = random.Random(args.seed)
        train_ds.clips = rng.sample(train_ds.clips, args.max_train_clips)
        print(f"[find] subsampled train clips -> {len(train_ds.clips)}", flush=True)
    val_ds = FindClipDataset(
        args.dataset, args.val_subset, args.seq_len, args.img_h, args.img_w, False,
        max(6, args.max_clips // 2), cache_videos=args.cache_videos, lru_max=args.lru_max,
        frame_drop_p=0.0,
    )
    if args.max_val_seqs and len(val_ds.anns) > args.max_val_seqs:
        # NOTE: fixed RNG on purpose -- a per-seed validation subset makes
        # val_l2 incomparable between seeds (found the hard way in the
        # multi-seed offset runs: 0.049 vs 0.11 was mostly the subset).
        rng = random.Random(0)
        keep = set(rng.sample(sorted(val_ds.anns), args.max_val_seqs))
        val_ds.clips = [c for c in val_ds.clips if c.ann_path.stem in keep]
        val_ds.anns = {k: v for k, v in val_ds.anns.items() if k in keep}
        print(f"[find] subsampled val seqs -> {len(val_ds.anns)}", flush=True)
    if args.max_val_clips and len(val_ds.clips) > args.max_val_clips:
        rng = random.Random(args.seed + 1)
        val_ds.clips = rng.sample(val_ds.clips, args.max_val_clips)
        print(f"[find] subsampled val clips -> {len(val_ds.clips)}", flush=True)
    cache_dir = args.dataset / "synthetic" / "find_cache" / f"{args.img_h}x{args.img_w}"
    cache_warm = cache_dir.exists() and any(cache_dir.glob("*.npz"))
    if args.num_workers >= 0:
        nw = args.num_workers
    elif device.type == "cuda" and (args.cache_videos or cache_warm):
        nw = 8
    else:
        nw = 0
    dl_kw: dict = dict(num_workers=nw, pin_memory=(device.type == "cuda"),
                       worker_init_fn=_seed_worker)
    if nw > 0:
        dl_kw["persistent_workers"] = True
        dl_kw["prefetch_factor"] = args.prefetch_factor
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=args.shuffle,
        generator=gen, **dl_kw
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, **dl_kw)
    use_amp = bool(args.amp and device.type == "cuda")
    print(
        f"[find] workers={nw} cache_videos={args.cache_videos} "
        f"cache_warm={cache_warm} frame_drop_p={args.frame_drop_p} amp={use_amp}",
        flush=True,
    )

    net_kw = dict(
        seq_len=args.seq_len, img_h=args.img_h, img_w=args.img_w, variant="transformer",
        d_model=args.d_model, nlayers=args.nlayers, nhead=args.nhead,
        ffn_dim=args.ffn_dim, feat_ch=args.feat_ch,
        heat_fuse=args.heat_fuse, conf_gate=(not args.no_conf_gate),
        refine_scale=args.refine_scale,
        peak_assoc=args.peak_assoc, peak_k=args.peak_k,
        heat_writeback=args.heat_writeback, writeback_w=args.writeback_w,
        offset_head=args.offset_head,
    )
    if args.resume and args.partial_init and Path(args.resume).exists():
        _ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        _raw = _ck.get("cfg") or {}
        if not isinstance(_raw, dict):
            _raw = vars(_raw)
        _arch = {k: _raw[k] for k in (
            "seq_len", "img_h", "img_w", "d_model", "nlayers", "nhead",
            "ffn_dim", "feat_ch", "heat_fuse", "conf_gate", "refine_scale",
            "peak_assoc", "peak_k", "heat_writeback", "writeback_w",
            "writeback_sigma",
        ) if k in _raw}
        net_kw.update(_arch)
        net_kw["offset_head"] = args.offset_head
        print(f"[find] arch from {Path(args.resume).name}: {_arch}", flush=True)
    model = build_find_net(**net_kw).to(device)
    start_ep = 1
    best = 1e9
    best_l2 = 1e9
    best_hit = -1.0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        if args.partial_init:
            src = ck["model"]
            dst = model.state_dict()
            loaded = 0
            for k, v in src.items():
                if k in dst and dst[k].shape == v.shape:
                    dst[k] = v
                    loaded += 1
            model.load_state_dict(dst)
            best = 1e9
            print(f"[find] partial-init {args.resume} tensors={loaded}/{len(dst)}", flush=True)
        else:
            model.load_state_dict(ck["model"])
            if args.reset_best:
                best = 1e9
                best_l2 = 1e9
                best_hit = -1.0
                print(f"[find] resumed {args.resume} (reset-best)", flush=True)
            else:
                best = float(ck.get("val", {}).get("coord_l2", best))
                best_l2 = best
                best_hit = float(ck.get("val", {}).get("partial_hit", -1.0))
                print(f"[find] resumed {args.resume} best_l2={best:.4f}", flush=True)
    if args.freeze_backbone:
        for name, p_ in model.named_parameters():
            if name.startswith(("encoder.", "heat_head.", "aff_to_heat.")):
                p_.requires_grad_(False)
        n_train = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
        n_all = sum(p_.numel() for p_ in model.parameters())
        print(f"[find] freeze-backbone trainable={n_train}/{n_all}", flush=True)
    if args.freeze_temporal:
        for name, p_ in model.named_parameters():
            if name.startswith((
                "temporal.", "token_proj.", "pos.",
                "refine_coord.", "refine_vis.", "heat_fuse.", "peak_ctx.",
            )):
                p_.requires_grad_(False)
        n_train = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
        n_all = sum(p_.numel() for p_ in model.parameters())
        print(f"[find] freeze-temporal trainable={n_train}/{n_all}", flush=True)
    opt = torch.optim.AdamW(
        (p_ for p_ in model.parameters() if p_.requires_grad),
        lr=args.lr, weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"[find] params={model.count_parameters()} train={len(train_ds)} val={len(val_ds)}", flush=True)
    # Write provenance *before* training so a crashed/killed run is still traceable.
    (out / "manifest.json").write_text(json.dumps({
        "status": "running",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv,
        "seed": args.seed,
        "host": socket.gethostname(),
        "versions": {"python": platform.python_version(),
                     "torch": torch.__version__,
                     "numpy": np.__version__,
                     "cuda": torch.version.cuda},
        "device": str(device),
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "params": model.count_parameters(),
        "model_cfg": dataclasses.asdict(model.cfg),
        "n_train_clips": len(train_ds),
        "n_val_clips": len(val_ds),
        "dataset": _dataset_fingerprint(Path(args.dataset)),
        "resume": str(args.resume) if args.resume else None,
        "resume_sha256": _sha256(Path(args.resume)) if args.resume else None,
    }, indent=2), encoding="utf-8")

    history = []
    for epoch in range(start_ep, args.epochs + 1):
        model.train()
        t0 = time.time()
        running, steps = 0.0, 0
        for batch in train_loader:
            non_block = device.type == "cuda"
            x = batch["image"].to(device, non_blocking=non_block)
            gt_c = batch["coords"].to(device, non_blocking=non_block)
            gt_v = batch["vis"].to(device, non_blocking=non_block)
            lam = args.lambda_xy
            if args.xy_boost_after and epoch > args.xy_boost_after:
                lam *= args.xy_boost
            # early epochs: lean harder on heat / coords0 so peaks separate first
            hm = args.lambda_hm * (1.5 if epoch <= 25 else 1.0)
            xy0 = args.lambda_xy0 * (1.3 if epoch <= 25 else 1.0)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_c, pred_v, pred_hm, pred_c0 = model(x, return_heat=True)
                losses = tridot_loss(
                    pred_c, pred_v, gt_c, gt_v,
                    pred_heat=pred_hm, pred_coords0=pred_c0,
                    lambda_xy=lam, lambda_xy0=xy0, lambda_hm=hm,
                    lambda_sep=args.lambda_sep, lambda_eq=args.lambda_eq,
                    lambda_temp=args.lambda_temp,
                    lambda_peak=args.lambda_peak,
                    peak_margin=args.peak_margin,
                    peak_exclude=args.peak_exclude,
                    last_frame_boost=2.5, hm_sigma=args.hm_sigma,
                )
                if args.offset_head and args.lambda_off > 0 and model.last_off is not None:
                    off_l = offset_loss(model.last_off, gt_c, gt_v)
                    losses["loss"] = losses["loss"] + args.lambda_off * off_l
                    losses["loss_off"] = off_l.detach()
                if args.peak_assoc and model.last_pick is not None and args.lambda_pick > 0:
                    losses["loss"] = losses["loss"] + args.lambda_pick * peak_candidate_loss(
                        model.last_pick, model.last_cand, gt_c, gt_v,
                    )
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            running += float(losses["loss"].item())
            steps += 1
        sched.step()
        val = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch, "loss": running / max(steps, 1),
            "val_l2": val["coord_l2"],
            "partial_hit": val["partial_hit"],
            "sec": time.time() - t0,
        }
        if "loss_off" in losses:
            row["loss_off"] = float(losses["loss_off"])
            print(f"  off_loss={row['loss_off']:.4f}", flush=True)
        history.append(row)
        print(
            f"  epoch {epoch}/{args.epochs} loss={row['loss']:.4f} "
            f"val_l2={row['val_l2']:.4f} partial_hit={row['partial_hit']:.3f} "
            f"({row['sec']:.1f}s)",
            flush=True,
        )
        ckpt = {"model": model.state_dict(), "cfg": model.cfg.__dict__, "epoch": epoch, "val": val}
        torch.save(ckpt, out / "last.pt")
        # Prefer partial-occlusion Top1 recall, but do not keep a checkpoint
        # whose fully-visible error has clearly regressed.
        l2_ok = val["coord_l2"] <= best_l2 * 1.04
        better_partial = val["partial_hit"] > best_hit + 1e-4
        better_l2 = val["coord_l2"] < best_l2 and val["partial_hit"] >= best_hit - 1e-4
        if l2_ok and (better_partial or better_l2):
            best_l2 = min(best_l2, val["coord_l2"])
            best_hit = max(best_hit, val["partial_hit"])
            best = val["coord_l2"]
            torch.save(ckpt, out / "best.pt")
    if not (out / "best.pt").exists() and (out / "last.pt").exists():
        # domain FT may never beat an easy-domain resumed best — keep last as best
        import shutil
        shutil.copy2(out / "last.pt", out / "best.pt")
        if history:
            best = float(history[-1]["val_l2"])
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    manifest = {
        "status": "finished",
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv,
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
        "host": socket.gethostname(),
        "versions": {"python": platform.python_version(),
                     "torch": torch.__version__,
                     "numpy": np.__version__,
                     "cuda": torch.version.cuda},
        "device": str(device),
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "params": model.count_parameters(),
        "model_cfg": dataclasses.asdict(model.cfg),
        "n_train_clips": len(train_ds),
        "n_val_clips": len(val_ds),
        "dataset": _dataset_fingerprint(Path(args.dataset)),
        "resume": str(args.resume) if args.resume else None,
        "resume_sha256": _sha256(Path(args.resume)) if args.resume else None,
        "best_pt_sha256": _sha256(out / "best.pt"),
        "last_pt_sha256": _sha256(out / "last.pt"),
        "best_val_l2": best,
        "history": history,
        "selection_policy": (
            "best.pt kept when val coord_l2 <= best_l2*1.04 AND "
            "(partial_hit strictly better OR coord_l2 strictly better with "
            "partial_hit not worse)"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[find] best_val_l2={best:.4f} → {out/'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
