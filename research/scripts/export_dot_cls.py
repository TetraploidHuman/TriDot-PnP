"""Export the dot-vs-clutter patch CNN to a dependency-free flat binary.

The Android app has no torch / ONNX Runtime in its offline Gradle cache, but the
net is tiny (23.6k params, 3 conv layers).  We fold BatchNorm into the conv
weights and dump every tensor as little-endian float32 in a fixed order, which a
~80-line Kotlin forward pass can consume.  An ONNX copy is written as a
reference for numeric parity checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.tridot_temporal.dot_cls import DotClassifier

PATCH = 15
ORDER = ["c1w", "c1b", "c2w", "c2b", "c3w", "c3b", "fcw", "fcb"]


def fold_bn(conv_w: torch.Tensor, conv_b: torch.Tensor, bn) -> tuple[np.ndarray, np.ndarray]:
    gamma = bn.weight.detach().cpu().numpy()
    beta = bn.bias.detach().cpu().numpy()
    mean = bn.running_mean.detach().cpu().numpy()
    var = bn.running_var.detach().cpu().numpy()
    eps = float(bn.eps)
    scale = gamma / np.sqrt(var + eps)
    w = conv_w.detach().cpu().numpy() * scale[:, None, None, None]
    b = (conv_b.detach().cpu().numpy() - mean) * scale + beta
    return np.ascontiguousarray(w, dtype=np.float32), np.ascontiguousarray(b, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ckpt = torch.load(str(a.ckpt), map_location="cpu", weights_only=False)
    model = DotClassifier(in_ch=int(ckpt.get("in_ch", 1)))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    c1w, c1b = fold_bn(model.net[0].weight, model.net[0].bias, model.net[1])
    c2w, c2b = fold_bn(model.net[4].weight, model.net[4].bias, model.net[5])
    c3w, c3b = fold_bn(model.net[8].weight, model.net[8].bias, model.net[9])
    fcw = model.head.weight.detach().cpu().numpy().astype(np.float32).reshape(-1)
    fcb = model.head.bias.detach().cpu().numpy().astype(np.float32)
    arrays = {"c1w": c1w, "c1b": c1b, "c2w": c2w, "c2b": c2b,
              "c3w": c3w, "c3b": c3b, "fcw": fcw, "fcb": fcb}
    a.out.mkdir(parents=True, exist_ok=True)
    off = {}
    buf = bytearray()
    for name in ORDER:
        arr = np.ascontiguousarray(arrays[name], dtype="<f4")
        off[name] = {"offset": len(buf), "shape": list(arr.shape)}
        buf += arr.tobytes()
    (a.out / "dot_cls.bin").write_bytes(bytes(buf))
    meta = {"patch": PATCH, "in_ch": int(ckpt.get("in_ch", 1)),
            "input_scale": 1.0 / 255.0, "threshold": float(ckpt.get("threshold", 0.5)),
            "tensors": off, "bytes": len(buf),
            "source_ckpt": str(a.ckpt)}
    (a.out / "dot_cls.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    onnx_path = a.out / "dot_cls.onnx"
    dummy = torch.zeros(1, int(ckpt.get("in_ch", 1)), PATCH, PATCH)
    torch.onnx.export(model, dummy, str(onnx_path), input_names=["patch"],
                      output_names=["logit"], opset_version=13,
                      dynamic_axes={"patch": {0: "n"}, "logit": {0: "n"}},
                      dynamo=False)
    print(json.dumps({"bin": str(a.out / "dot_cls.bin"), "bytes": len(buf),
                      "onnx": str(onnx_path)}))


if __name__ == "__main__":
    main()
