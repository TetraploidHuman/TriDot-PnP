"""Generate research/results/classic_cv3/REPORT.md from the far-bench JSONs.

    PYTHONPATH=. research/.venv/bin/python -m research.scripts.classic_cv3_report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "research/results/far_bench"
PROBE = ROOT / "research/results/real_probe"
OUT = ROOT / "research/results/classic_cv3"
PRESETS = ("clean", "phone_easy", "phone_far")
METHOD_LABEL = {
    "b0": "B0 (HSV 绝对阈值)",
    "classic2": "classic2 (v2 前端)",
    "classic3_contrast": "v3 对比度前端 (floor12)",
    "classic3_union": "v3 对比度 ∪ v2 (floor12)",
}


def load(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def fmt(v, digits=3):
    return "—" if v is None else f"{v:.{digits}f}"


def table_overall() -> list[str]:
    rows = ["| 方法 | clean | phone_easy | phone_far |",
            "| --- | --- | --- | --- |"]
    data = {p: load(BENCH / f"{p}_12_v3") for p in PRESETS}
    methods = ["b0", "classic2", "classic3_contrast", "classic3_union"]
    for m in methods:
        cells = []
        for p in PRESETS:
            d = data[p]
            if not d or m not in d["methods"]:
                cells.append("—")
                continue
            cells.append(fmt(d["methods"][m]["det10_all"]))
        rows.append(f"| {METHOD_LABEL.get(m, m)} | " + " | ".join(cells) + " |")
    return rows


def table_buckets(preset: str) -> list[str]:
    d = load(BENCH / f"{preset}_12_v3")
    if not d:
        return [f"> `{preset}_12_v3` 未生成。"]
    rows = ["| 方法 | det@10 all | <20px | 20–40px | 40–80px | 80+px |",
            "| --- | --- | --- | --- | --- | --- |"]
    for m, sm in d["methods"].items():
        b = sm.get("buckets", {})

        def cell(k):
            x = b.get(k, {})
            if x.get("det10_all") is None:
                return "—（n=0）" if x.get("n", 0) == 0 else "—"
            return f"{x['det10_all']:.3f} (n={x['n']})"

        rows.append("| {} | **{}** | {} | {} | {} | {} |".format(
            METHOD_LABEL.get(m, m), fmt(sm.get("det10_all")),
            cell("0-20"), cell("20-40"), cell("40-80"), cell("80+")))
    return rows


def table_real_probe() -> list[str]:
    d = load(PROBE / "summary.json")
    if not d or "detection" not in d:
        return ["> 真实视频探测未运行（`research/scripts/real_video_probe.py`）。"]
    st = d.get("real_stats", {})
    sc = d.get("synth_clean_stats", {})
    lines = [
        "| 场景 | mean V | 灰度 σ | Laplacian 方差 | 色差 p99.5 |",
        "| --- | --- | --- | --- | --- |",
        f"| 真实视频 `{d.get('video')}` | {st.get('mean_v')} | {st.get('gray_std')} | "
        f"{st.get('laplacian_var')} | {st.get('chroma_p995')} |",
        f"| 合成 clean（同长度） | {sc.get('mean_v')} | {sc.get('gray_std')} | "
        f"{sc.get('laplacian_var')} | {sc.get('chroma_p995')} |",
        "",
        "| 退化档 | 方法 | 检出三点 | 点漂移中位 (px) | 视边长误差中位 (px) |",
        "| --- | --- | --- | --- | --- |",
    ]
    rows = d.get("detection", []) + d.get("detection_degraded", [])
    for r in rows:
        lines.append("| {} | {} | {}/{} | {} | {} |".format(
            r["method"].split("+")[-1] if "+" in r["method"] else "原始",
            "classic2" if r["method"].startswith("classic2") else "classic3",
            r["detected_3_spots"], r["n_frames"],
            fmt(r.get("point_drift_px_med"), 2), fmt(r.get("edge_err_px_med"), 2)))
    lines += ["", "> 参考帧 = classic3 在未退化真实帧上的三点（其自身视边长 152px，60/60 帧一致）。"
              "点漂移 = 与参考帧同色点的像素距离中位。"]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", type=Path, default=OUT / "REPORT.md")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    parts = [
        "# 经典 CV v3：远距/小光斑检测（A–F 计划执行报告）",
        "",
        "## 0. 一句话结论",
        "",
        "真实手机取景（4:2:0 色度 + ISP 降噪 + 曝光衰减 + 模糊）下，v2 的**绝对亮度阈值**在远距失效："
        "同一批 12 条 hard_test 序列，clean 时 classic2 det@10 all 0.815，phone_far 掉到约 0.36。"
        "v3 把前端换成**颜色差 + 形态学顶帽的局部对比度**（阈值由高频 MAD 噪声自适应决定，去掉了绝对亮度门），"
        "用上一帧追踪位姿推出光斑像素半径做**尺度自适应**的核与窗口，对 1–3px 小点做 **ROI ×4 上采样亚像素质心**，"
        "并在某色候选缺失时用**亮度对比度候选 + 颜色合理性否决**兜底。"
        "结果（12 序列同帧同解码对照）：phone_far 上 det@10 all 从 classic2 的 0.340 提到 **0.777**，"
        "phone_easy 从 0.361 提到 0.814，干净集不降反升（0.824 vs 0.815）；"
        "代价是每帧耗时约 78ms（纯对比度）→151ms（并集），App 端需要降分辨率/金字塔才能上机。"
        "另外在 App 采集侧（E）加了高分辨率分析 / 关闭 ISP 降噪 / 锁定白平衡三个开关。"
        "在唯一一条真实视频（近景桌面、无标注）上做内容级验证：把它退化到三档后，"
        "classic2 仍「检出三点」但点漂移 260–318px（锁到干扰块上、视边长 152→414–473px 的假三角形），"
        "v3 同条件点漂移 0.37–1.08px。",
        "",
        "## 1. 诊断：远距为什么会失效",
        "",
        "合成 hard_test 中 `edge_px ≈ fx·L/depth`：fx≈1108、L=100mm ⇒ 1m 111px、3m 37px、5m 22px、10m 11px；"
        "点半径约 0.09·edge_px，5m 时 2–3px、10m 时 1–2px。",
        "",
        "App 侧 `BrightSpotDetector.kt` 用绝对亮度阀门（`minPixelBrightness=10`、`minTotalBrightness=20`、"
        "`dynThreshold=max(30, 0.7·maxBrightness)`、`minRefineRadius=2`）：小点被 4:2:0 色度下采样抹成灰色、"
        "被 ISP 降噪抹掉高光、AE 拉低曝光后整体亮度不足 ⇒ 在阈值前就消失。",
        "",
        "离线分桶（classic2、全分辨率、hard_test 全可见帧，按 GT 视边长）："
        "<64px 0.511 (n=176)、64–100px 0.676 (n=657)、≥100px 0.746 (n=1694)；"
        "漏检归因：153 帧有候选但拼错、576 帧前端就没有候选。",
        "",
        "## 2. 退化模型与评测口径（F）",
        "",
        "`research/tridot_temporal/degrade.py` 按真实成像链路建模：光学（离焦）→ 传感器（降采样、曝光增益/偏移、"
        "散粒+读出噪声、4:2:0 色度往返）→ ISP（中值降噪、锐化、JPEG）。随机项按 `(preset, frame_index)` 确定性播种。"
        "四个 preset：`clean` / `phone_easy` / `phone_far` / `phone_harsh`；9 个单测覆盖恒等、确定性、"
        "色度保亮度、降采样降峰、降噪抹 1px、JPEG=100 恒等。",
        "",
        "评测口径沿用 `eval_v2`：det@10 all（分母=全部帧，遮挡帧算失败）、逐点 recall@10、pose coverage、"
        "t_err 中位/p90、延迟；`far_bench.py` 再按 GT 视边长分桶。所有数字都来自同一解码、同一 PnP、"
        "无任何在线门控或后处理手调。",
        "",
        "### 2.1 三档退化下的总体 det@10 all（12 序列 hard_test）",
        "",
        *table_overall(),
        "",
        "### 2.2 phone_far 分桶（回答「多远开始失效」）",
        "",
        *table_buckets("phone_far"),
        "",
        "### 2.3 clean 分桶（回退发生在哪）",
        "",
        *table_buckets("clean"),
        "",
        "## 3. 实现（A–D）",
        "",
        "* **A 局部对比度前端**：颜色差图 `clip(channel − max(others))` → 形态学顶帽（`MORPH_TOPHAT`，核由 B 给出）"
        "→ 阈值 `thr = max(contrast_floor, k_sigma·σ_noise, rel_frac·p99.5)`，`σ_noise = 1.4826·MAD`；"
        "绝对亮度门全部去掉。",
        "* **B 尺度自适应**：`r_px ≈ 0.09·edge_px`（上一帧追踪位姿给出），顶帽核 `2·(2r)+1`（≤31），"
        "精修窗口 1.5–2×r_px、同色去重半径随 r 缩放；无位姿时退化为多尺度 (3,7,15,31) 取最大响应。",
        "* **C 小点亚像素**：连通域 <4px 时，在周围取 patch 双三次 ×4 上采样，在上采样后的颜色差图上做加权质心。",
        "* **D 检测-颜色解耦**：某颜色候选为空时，用亮度对比度候选补位，并用其它两色 map 的强度做"
        "「颜色合理性」否决（≥40 且 ≥1.6×自身强度 ⇒ 该点属于别的颜色）。",
        "",
        "## 4. 相机采集侧（E）",
        "",
        "`AppTuningSettings` 新增三个开关，`App.kt` 的「采集与远距」设置段暴露：",
        "",
        "| 开关 | 行为 | 备注 |",
        "| --- | --- | --- |",
        "| 高分辨率分析 | `ImageAnalysis` 目标分辨率 1920×1080（否则 1280×720） | 小点更多像素，代价 CPU |",
        "| 关闭 ISP 降噪 | `NOISE_REDUCTION_MODE = OFF` | 保住 1–2px 高光 |",
        "| 锁定白平衡 | `CONTROL_AWB_LOCK = true` | 避免 AWB 漂移吃掉色差 |",
        "",
        "三者都在**重新进入预览后生效**（`ImageAnalysis` 分辨率在绑定用例时确定）。"
        "`compileDebugKotlinAndroid` 通过；APK 已重打，dex 里能看到这三个字段名。",
        "",
        "## 5. 真实视频（无标注）上的内容级验证",
        "",
        "现有唯一真实视频 `datasets/TriDotPose/real/videos/smoke_real_desk.mp4`（60 帧，桌面近景、含干扰块）。"
        "它没有标注，但可以回答两件事：真实内容与哪档退化最像；退化后前端是否还锁在正确的点上。",
        "",
        *table_real_probe(),
        "",
        "结论：真实视频的色差 p99.5 是合成 clean 的 2.4 倍、Laplacian 方差 2.7 倍（桌面木纹 + 干扰块），"
        "并不像电话远距场景。把它当作「真实内容」的替身加上三档退化后："
        "classic2 在 60/60 帧仍然「检出三点」，但点漂移中位 260–318px（锁到干扰块上了，视边长从 152px 变成 414–473px，"
        "是一个完整的**假三角形**）；v3 同条件下点漂移 0.37–1.08px、视边长误差 <0.7px。"
        "说明在真实图像内容上，绝对阈值前端遇到曝光/噪声变化会整体漂移，而对比度前端不会。",
        "",
        "## 6. 局限与后续",
        "",
        "1. **不是真机**：退化模型仍是仿真的，假设零畸变、无卷帘、无镜头脏污；真机 1–8m 距离扫描才能定论。",
        "2. **单用对比度会掉 clean**：纯对比度前端在 clean 上是 0.783（v2 0.815）；"
        "并集模式（对比度 ∪ v2 绝对阈值）回到 0.824，说明两者互补而不是替代。"
        "注意 B0 在 phone_far 上 0.426 反而高于 classic2 的 0.340 —— 更「笨」的固定 HSV 阈值在退化下比调过的 v2 更稳。",
        "3. **Kotlin 前端未移植**：App 端目前只有 E（采集侧）与之前的 PnP/时序修复；"
        "把颜色差 + 顶帽 + 局部对比度阈值搬进 `BrightSpotDetector` 需要自实现可分离 min/max 滤波（App 无 OpenCV），"
        "建议在 640×360 分析分辨率上做，成本可控。",
        "4. **延迟**：Python 全分辨率 v3 约 78ms/帧（对比度）/151ms/帧（并集），手机 AArch64 单核会更慢；"
        "真机建议 2–3 级金字塔 + 预测 ROI。",
        "5. **遮挡帧**：det@10 all 仍把 `occlude_hand`（0 帧三点全可见）记为 0；2 点 + 已知几何是下一步。",
        "",
        "## 7. 复现命令",
        "",
        "```bash",
        "# 三档退化 × 分桶评测（12 序列）",
        "for p in clean phone_easy phone_far; do",
        "  D=\"\"; [ \"$p\" != clean ] && D=\"--degrade $p\"",
        "  PYTHONPATH=. research/.venv/bin/python -m research.scripts.far_bench \\",
        "    --subset hard_test --limit-seq 12 --methods b0,classic2 \\",
        "    --v3-configs contrast_f12_contrast:12:contrast,contrast_f12_union:12:union \\",
        "    $D --out research/results/far_bench/${p}_12_v3",
        "done",
        "",
        "# 真实视频内容级验证",
        "PYTHONPATH=. research/.venv/bin/python -m research.scripts.real_video_probe",
        "",
        "# 报告",
        "PYTHONPATH=. research/.venv/bin/python -m research.scripts.classic_cv3_report",
        "",
        "# 单测",
        "PYTHONPATH=. research/.venv/bin/python -m unittest discover -s research/tests",
        "```",
        "",
    ]
    args.md.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {args.md} ({len(parts)} lines)")


if __name__ == "__main__":
    main()
