"""把「远距/小光斑」的真实合成帧导出成 JVM 单测固件。

用途：Kotlin 侧的 `FarSpotDetectorTest` 不能依赖 Android Bitmap，也不能在测试里跑
Python 镜头退化模型，所以这里预先把数据集里几张典型的远距帧（干净的 + 经过
`phone_far` 退化的）写成 PPM（P6 RGB）固件，坐标取 640 宽（与端侧 maxDim 一致），
并附上 GT 点位（同样缩放到 640 宽）。

固件本身很小（640×360×3 ≈ 690KB/张），只用于单测，不进 APK。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from research.tridot_temporal import degrade

DATASET = Path("datasets/TriDotPose")
OUT = Path("composeApp/src/androidUnitTest/resources/far_fixtures")
WIDTH = 640
FACTORS = ("far", "clutter", "screen", "handheld")
MAX_EDGE_PX = 120.0


def find_frames(limit_per_factor: int = 1) -> list[tuple[str, str, int]]:
    splits = json.loads((DATASET / "meta/splits.json").read_text())
    chosen: list[tuple[str, str, int]] = []
    per_factor: dict[str, int] = {}
    for seq in splits["hard_test"]:
        ann = json.loads((DATASET / "synthetic/annotations" / f"{seq}.json").read_text())
        factor = ann["info"]["factor"]
        if factor not in FACTORS:
            continue
        if per_factor.get(factor, 0) >= limit_per_factor:
            continue
        fx = ann["camera"]["fx"]
        edge_mm = ann["info"]["edge_length_mm"]
        for idx, frame in enumerate(ann["frames"]):
            if idx < 1 or not frame.get("all_visible"):
                continue
            if not ann["frames"][idx - 1].get("all_visible"):
                continue
            depth = float(np.linalg.norm(np.asarray(frame["pose_gt"]["translation_mm"], dtype=np.float64)))
            edge_px = fx * edge_mm / depth
            if edge_px <= MAX_EDGE_PX:
                chosen.append((seq, factor, idx))
                per_factor[factor] = per_factor.get(factor, 0) + 1
                break
    return chosen


def read_frame(seq: str, index: int) -> tuple[np.ndarray, dict]:
    ann = json.loads((DATASET / "synthetic/annotations" / f"{seq}.json").read_text())
    video = DATASET / "synthetic/videos" / f"{seq}.mp4"
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read {seq} frame {index}")
    return frame, ann


def write_ppm(path: Path, bgr: np.ndarray) -> None:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    with path.open("wb") as fh:
        fh.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        fh.write(rgb.tobytes())


def resize(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    height = int(round(h * width / w))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    chosen = find_frames()
    cases = []
    for seq, factor, index in chosen:
        frame, ann = read_frame(seq, index)
        gt_full = np.asarray(ann["frames"][index]["keypoints"], dtype=np.float64).reshape(3, 3)
        prev_full = np.asarray(ann["frames"][index - 1]["keypoints"], dtype=np.float64).reshape(3, 3)
        prev_pose = ann["frames"][index - 1]["pose_gt"]
        cur_pose = ann["frames"][index]["pose_gt"]
        scale = WIDTH / frame.shape[1]

        clean = resize(frame, WIDTH)
        write_ppm(OUT / f"{seq}_f{index}_clean.ppm", clean)

        degraded_full = degrade.apply(frame.copy(), degrade.preset("phone_far"), seed=index)
        degraded = resize(degraded_full, WIDTH)
        write_ppm(OUT / f"{seq}_f{index}_phone_far.ppm", degraded)

        cases.append(
            {
                "seq_id": seq,
                "factor": factor,
                "frame_index": index,
                "width": int(clean.shape[1]),
                "height": int(clean.shape[0]),
                "gt_points_640": [[round(float(gt_full[i, 0]) * scale, 3), round(float(gt_full[i, 1]) * scale, 3)] for i in range(3)],
                "pred_points_640": [[round(float(prev_full[i, 0]) * scale, 3), round(float(prev_full[i, 1]) * scale, 3)] for i in range(3)],
                "prev_rotation": [[round(float(v), 6) for v in row] for row in prev_pose["rotation_matrix"]],
                "prev_translation_mm": [round(float(v), 3) for v in prev_pose["translation_mm"]],
                "cur_translation_mm": [round(float(v), 3) for v in cur_pose["translation_mm"]],
                "cur_rotation": [[round(float(v), 6) for v in row] for row in cur_pose["rotation_matrix"]],
                "edge_px_640": round(float(np.linalg.norm(gt_full[0, :2] - gt_full[1, :2])) * scale, 2),
                "presets": ["clean", "phone_far"],
            }
        )

    (OUT / "cases.json").write_text(json.dumps({"width": WIDTH, "cases": cases}, ensure_ascii=False, indent=2))
    # Kotlin 单测不引入 JSON 库：同时写一份纯文本清单
    lines = ["# seq factor frame width height edge_px_640 preset gt(x6) pred(x6)"]
    pose_lines = []
    for case in cases:
        pts = " ".join(f"{x:.2f} {y:.2f}" for x, y in case["gt_points_640"])
        for preset in case["presets"]:
            pred = " ".join(f"{x:.2f} {y:.2f}" for x, y in case["pred_points_640"])
            lines.append(
                f"{case['seq_id']} {case['factor']} {case['frame_index']} {case['width']} {case['height']} "
                f"{case['edge_px_640']:.2f} {preset} {pts} {pred}"
            )
    (OUT / "cases.txt").write_text("\n".join(lines) + "\n")
    # 上一帧位姿（R 行主序 9 + t mm 3），供 Kotlin 侧做候选位姿校验
    for case in cases:
        r = " ".join(f"{v:.6f}" for row in case["prev_rotation"] for v in row)
        t = " ".join(f"{v:.3f}" for v in case["prev_translation_mm"])
        for preset in case["presets"]:
            pose_lines.append(f"{case['seq_id']} {preset} {r} {t}")
    (OUT / "poses.txt").write_text("\n".join(pose_lines) + "\n")
    print(f"wrote {len(cases)} cases to {OUT}")
    for case in cases:
        print(case["seq_id"], case["factor"], "edge_px_640 =", case["edge_px_640"])


if __name__ == "__main__":
    main()
