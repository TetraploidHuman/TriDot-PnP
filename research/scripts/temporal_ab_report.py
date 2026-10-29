#!/usr/bin/env python
"""Render research/results/temporal_ab/REPORT.md from the A/B summaries.

Usage:
    research/.venv/bin/python research/scripts/temporal_ab_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "results" / "temporal_ab"


def fmt(v, digits=1):
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def pct(v, digits=1):
    if v is None:
        return "—"
    return f"{100.0 * v:.{digits}f}%"


def table(rows, headers):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def method_rows(summary):
    rows = []
    for name in ("fixed", "tracked", "disamb"):
        m = summary["methods"].get(name)
        if m is None:
            continue
        rows.append([
            name,
            fmt(m["t_err_mm"]["median"]),
            fmt(m["t_err_mm"]["p90"]),
            pct(m["t_err_mm"]["within_50mm"]),
            fmt(m["rot_err_deg"]["median"]),
            fmt(m["rot_err_deg"]["p90"]),
            pct(m["rot_err_deg"]["within_20deg"]),
            f'{m["rot_flips_gt60"]["count"]}/{m["rot_flips_gt60"]["frames"]}',
            fmt(m["step_mm_s"]["median"]),
            fmt(m["latency_ms"]["median"], 2),
        ])
    return rows


HEADERS = ["method", "t_err med (mm)", "t_err p90", "t≤50mm",
           "rot_err med (°)", "rot_err p90", "rot≤20°", "flips>60°",
           "step med (mm/s)", "latency (ms)"]


def factor_table(summary):
    facs = sorted(set(summary["per_factor"]["fixed"]) | set(summary["per_factor"]["tracked"]))
    rows = []
    for f in facs:
        a = summary["per_factor"]["fixed"].get(f, {})
        b = summary["per_factor"]["tracked"].get(f, {})
        rows.append([
            f,
            fmt(a.get("t_err_med")), fmt(b.get("t_err_med")),
            f'{a.get("flips", 0)}', f'{b.get("flips", 0)}',
            f'{a.get("frames", 0)}',
        ])
    return table(rows, ["factor", "t_err med fixed", "t_err med tracked",
                        "flips fixed", "flips tracked", "frames"])


def section(subset, summary):
    n_seq = summary.get("n_seq")
    n_frames = summary.get("n_frames")
    parts = [f"### {subset}（{n_seq} 序列 / {n_frames} 帧）", "",
             table(method_rows(summary), HEADERS), ""]
    if summary["methods"].get("fixed") and summary["methods"].get("tracked"):
        a = summary["methods"]["fixed"]
        b = summary["methods"]["tracked"]
        parts.append(
            "变化：flips {fa}→{fb}；t_err 中位 {ta}→{tb} mm；"
            "rot_err 中位 {ra}→{rb}°；latency {la}→{lb} ms。".format(
                fa=a["rot_flips_gt60"]["count"], fb=b["rot_flips_gt60"]["count"],
                ta=fmt(a["t_err_mm"]["median"]), tb=fmt(b["t_err_mm"]["median"]),
                ra=fmt(a["rot_err_deg"]["median"]), rb=fmt(b["rot_err_deg"]["median"]),
                la=fmt(a["latency_ms"]["median"], 2), lb=fmt(b["latency_ms"]["median"], 2)))
        parts.append("")
    if summary.get("per_factor"):
        parts += [factor_table(summary), ""]
    return "\n".join(parts)


def main() -> int:
    summaries = {}
    for subset in ("test", "hard_test"):
        path = OUT / f"{subset}_summary.json"
        if path.exists():
            summaries[subset] = json.loads(path.read_text())
    if not summaries:
        raise SystemExit(f"no summaries under {OUT}")

    doc = ["# 时序滤波 + 双解消歧 A/B（B1 检测器固定，仅换解算器）", "",
           "同一批 `B1Detector(grid_size=32)` 检测结果分别喂给：", "",
           "- **fixed**：`AppFixedSolver` —— 修好的三点 PnP（Newton+LM），无时序、无选根；",
           "- **tracked**：`AppTrackedSolver` —— 多初值求全部几何根 + 时序一致性选根 + alpha-beta 滤波 + 离群滑行（默认上架配置：alpha=0.85, gate=60mm/8%）；",
           "- **disamb**：同一套选根/门限，但输出原始测量（不滤波）—— 用来隔离「选根」与「滤波」各自的贡献。", "",
           "指标说明：`t_err` 为目标原点平移误差（vs GT `pose_gt.translation_mm`）；"
           "`rot_err` 为旋转测地角误差；`flips>60°` 统计相邻**有效帧**之间旋转跳变 >60° 的次数"
           "（镜像解翻转会让姿态瞬间翻面而距离几乎不变）；`step` 为帧间质心位移速度（抖动代理）；"
           "`latency` 只含纯解算（detect 已固定相同）。", ""]
    for subset, summary in summaries.items():
        doc.append(section(subset, summary))
    def _m(sub, name):
        return summaries[sub]["methods"][name] if sub in summaries else None

    def _line(sub):
        a, b = _m(sub, "fixed"), _m(sub, "tracked")
        if not a or not b:
            return None
        return (f"- **{sub}**：flips {a['rot_flips_gt60']['count']}→{b['rot_flips_gt60']['count']}"
                f"（{a['rot_flips_gt60']['frames']} 个有效帧间隔），"
                f"rot_err 中位 {a['rot_err_deg']['median']:.1f}°→{b['rot_err_deg']['median']:.1f}°，"
                f"rot≤20° {100*a['rot_err_deg']['within_20deg']:.1f}%→{100*b['rot_err_deg']['within_20deg']:.1f}%；"
                f"t_err 中位 {a['t_err_mm']['median']:.1f}→{b['t_err_mm']['median']:.1f} mm，"
                f"p90 {a['t_err_mm']['p90']:.1f}→{b['t_err_mm']['p90']:.1f} mm，"
                f"t≤50mm {100*a['t_err_mm']['within_50mm']:.1f}%→{100*b['t_err_mm']['within_50mm']:.1f}%；"
                f"帧间位移中位 {a['step_mm_s']['median']:.0f}→{b['step_mm_s']['median']:.0f} mm/s；"
                f"纯解算延迟 {a['latency_ms']['median']:.2f}→{b['latency_ms']['median']:.2f} ms/帧。")

    doc += ["## 结论", ""]
    for sub in ("test", "hard_test"):
        line = _line(sub)
        if line:
            doc.append(line)
    doc += [
        "",
        "1. **选根是姿态指标的主要来源**：`tracked` 与 `disamb` 的旋转指标几乎一致，"
        "说明收益来自「跨帧锁定同一几何解」而不是滤波本身；镜像解翻面会让姿态瞬间错 40–150°，"
        "这正是之前端上 6DOF 姿态跳动的根因。",
        "2. **滤波主要降抖动**：`step`（帧间质心位移速度）大幅下降，说明输出平滑；"
        "代价是快加速时略有滞后——默认 α=0.85 已把滞后压到接近测量值，"
        "需要更平滑可在 App 里调低「滤波平滑强度」。",
        "3. **精度不牺牲**：easy 集 t_err 中位从 7.6 降到 5.5 mm；hard 集持平（84 vs 80 mm，"
        "但 p90 从 1025 降到 926 mm）。距离本就不受双解影响（两解距离差几 mm~几十 mm），"
        "受影响的是姿态与抖动。",
        "4. **开销可忽略**：纯解算 0.9→4.9 ms/帧；检测端 B1 每帧 ~188 ms（grid 32），"
        "所以整机帧率不变。",
        "5. **首帧先验的局限**：无历史时用「最接近单位旋转」选根，在 hard_test 双根帧上与 GT "
        "一致率 81.6%；选错时靠旋转连续性会锁死在镜像解上（表里 far/hardmix 的 rot≤20° 提升有限即此故）。"
        "下一步可加「两候选并行假设 + 残差比较」或让用户轻点一下第 1 帧。",
        "",
        "## 复现", "",

        "```bash",
        "research/.venv/bin/python -m research.scripts.temporal_ab --subset test",
        "research/.venv/bin/python -m research.scripts.temporal_ab --subset hard_test",
        "research/.venv/bin/python research/scripts/temporal_ab_report.py",
        "```",
        "",
        "端侧对应实现：`composeApp/src/androidMain/kotlin/org/example/tridotpnp/PnPTemporalTracker.kt`，"
        "接入点 `PnPDistanceCalculator.calculate3PointPnPTracked(...)`，"
        "开关在 App「校准与 PnP」设置里的「时序滤波与双解消歧」。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(doc) + "\n")
    print(f"wrote {OUT / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
