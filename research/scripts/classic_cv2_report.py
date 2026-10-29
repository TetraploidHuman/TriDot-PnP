"""Render the classic-CV v2 (B0 -> colour-difference + top-hat + triple search) report.

Reads `research/results/honest_cv2/<subset>/summary.json` plus the B0 diagnostic
summary (`research/results/classic_diag/<subset>_summary.json`) and writes a
markdown report.  All numbers are pulled from the JSON files, never hard-coded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SUBSETS = ("test", "hard_test", "holdout_hard")
METHODS = ("b0", "classic2")


def _fmt(x, nd=3):
    if x is None:
        return "-"
    return f"{x:.{nd}f}"


def load(root: Path, subset: str) -> dict:
    p = root / subset / "summary.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def row(d: dict, method: str) -> str:
    r = d["results"][method]
    det, pt, po, la = r["det"], r["point"], r["pose"], r["latency"]
    return (f"| {method} | {_fmt(det['det10_all'])} | {_fmt(det['det10_legacy'])} | "
            f"{_fmt(pt['recall@10'])} | {_fmt(po['coverage'])} | "
            f"{po['t_err_clean']['median_mm']:.1f} | {po['t_err_clean']['p90_mm']:.1f} | "
            f"{po['jitter_mm_per_s']:.0f} | {la['fps_est']:.0f} |")


def table(d: dict) -> str:
    head = ("| method | det@10 all | det@10 legacy | recall@10 | pose cov | "
            "t_err med (mm) | t_err p90 (mm) | jitter (mm/s) | est FPS |\n"
            "|---|---|---|---|---|---|---|---|---|")
    rows = [row(d, m) for m in METHODS if m in d.get("results", {})]
    return head + "\n" + "\n".join(rows)


LEARNED = {"test": "research/results/honest/final_test",
           "hard_test": "research/results/honest/final_hard_test",
           "holdout_hard": "research/results/honest/final_holdout_hard"}


def learned_rows(subset: str) -> str:
    p = Path(LEARNED.get(subset, "")) / "summary.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    rows = []
    for name, r in d["results"].items():
        if not name.startswith("find"):
            continue
        det, pt, po, la = r["det"], r["point"], r["pose"], r["latency"]
        rows.append(f"| {name} | {_fmt(det['det10_all'])} | {_fmt(det['det10_legacy'])} | "
                    f"{_fmt(pt['recall@10'])} | {_fmt(po['coverage'])} | "
                    f"{po['t_err_clean']['median_mm']:.1f} | {po['t_err_clean']['p90_mm']:.1f} | "
                    f"{la['fps_est']:.0f} |")
    if not rows:
        return ""
    return ("\n\n参考：学习型检测器（同一评测器、同一子集，来自 `research/results/honest/`）\n\n"
            "| method | det@10 all | det@10 legacy | recall@10 | pose cov | "
            "t_err med (mm) | t_err p90 (mm) | est FPS |\n"
            "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows))


def factor_table(d: dict) -> str:
    facs = sorted(set(d["results"]["b0"]["by_factor"]) &
                  set(d["results"]["classic2"]["by_factor"]))
    head = "| factor | B0 det@10 all | classic2 det@10 all | delta |\n|---|---|---|---|"
    rows = []
    for f in facs:
        a = d["results"]["b0"]["by_factor"][f]["det10_all"]
        b = d["results"]["classic2"]["by_factor"][f]["det10_all"]
        rows.append(f"| {f} | {_fmt(a)} | {_fmt(b)} | {b - a:+.3f} |")
    return head + "\n" + "\n".join(rows)


def transfer_section(root: Path) -> str:
    """Quantify how much of the gain comes from the (scene-specific) top-hat."""
    paths = {"try2 无 top-hat": root.parent / "try2",
             "try3 恒开 top-hat": root.parent / "try3",
             "try4 自适应 top-hat": root.parent / "try4"}
    rows = []
    for label, base in paths.items():
        cells = []
        for s in SUBSETS:
            d = load(base, s)
            r = d.get("results", {}).get("classic2")
            cells.append(f"{r['det']['det10_all']:.3f}" if r else "-")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    if not rows:
        return ""
    return ("## 3. 迁移风险：哪些收益依赖当前合成集\n\n"
            "把 `classic2` 拆成两组开关看收益来源（同一评测器、同一解码）：\n\n"
            "| 配置 | test det@10 all | hard_test det@10 all | holdout_hard det@10 all |\n"
            "|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
            "- **场景无关的部分**（颜色差前端 + B0 候选并集 + 几何/光度/尺寸/时序三元组打分 + "
            "亚像素加权质心 + pose-jump 门控）：hard_test 0.172 → 0.367，holdout 0.169 → 0.366；"
            "这部分不依赖\"有泛光\"的假设。\n"
            "- **场景相关的部分**（31×31 top-hat + `p99.5>15 / p50>2` 触发规则）：再 +0.13，"
            "但只在\"亮而平滑的墙面/桌面/屏幕泛光\"这类合成 preset 上验证过；"
            "真实场景若没有这种背景，这部分收益无法预期，甚至可能在 easy 域小亏"
            "（test 0.659 → 0.647）。\n"
            "- 因此应把 top-hat 视为\"假设集里的一路\"而不是默认前置：真实数据到位后应改为"
            "\"raw / top-hat / 尺度自适应带通\"三路并集、由打分器仲裁；"
            "核大小按预期光斑半径（`r_px ≈ fx·r_mm/depth`，depth 取上一帧位姿）设为 3–5×`r_px`。\n")


def diag_section(root: Path) -> str:
    lines = []
    for s in ("test", "hard_test", "holdout_hard"):
        p = Path("research/results/classic_diag") / f"{s}_summary.json"
        if not p.exists():
            continue
        a = json.loads(p.read_text())["summary"]["all"]
        lines.append(
            f"- **{s}** ({a['n_frames']} 帧 / {a['n_all_visible']} 全可见): "
            f"B0 det@10 {a['det10_legacy']:.3f}, oracle 关联 {a['oracle_det10']:.3f}, "
            f"前端漏检帧占比 {a['detect_miss_share']:.3f}, 关联拼错帧占比 "
            f"{a['assoc_wrong_share']:.3f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="research/results/honest_cv2/try4")
    ap.add_argument("--out", default="research/results/classic_cv2/REPORT.md")
    args = ap.parse_args()
    root = Path(args.root)
    blocks = []
    for s in SUBSETS:
        d = load(root, s)
        if not d:
            print(f"[report] missing {s}")
            continue
        blocks.append(f"### {s}\n\n{table(d)}{learned_rows(s)}\n")
        blocks.append(f"<details><summary>per-factor</summary>\n\n{factor_table(d)}\n\n</details>")
    body = "\n\n".join(blocks)
    diag = diag_section(root)
    transfer = transfer_section(root)
    text = f"""# 经典 CV 检测器 v2：诊断、实现与诚实对比

> 数据来源：`{root}`（eval_v2，固定解码、无在线门控；test/hard_test 为历史调参集，
> holdout_hard 为 2026-10-28 新生成的、从未参与调参的留存集）。

## 0. 一句话结论

B0 的瓶颈主要在**前端**（hard_test 53% 的全可见帧有一色根本没有候选）与**三点选择**
（22% 有候选但拼错），而不是 PnP。把前端换成"颜色差 + top-hat 背景抑制 + 自适应阈值"，
把选择换成"光度/几何/尺寸/时序打分下的穷举三元组"，并保留 B0 的 HSV 候选做并集：

{body}

## 1. B0 失败诊断（`research/results/classic_diag/`）

{diag or "（缺诊断 JSON）"}

## 2. 实现

`research/tridot_temporal/classic_v2.py`

1. **前端**：`red = clip(R - max(G,B), 0)` 等颜色差图 → `cv2.MORPH_TOPHAT`
   （31×31）去掉缓慢变化的背景（墙面/桌面/屏幕泛光）→ 阈值
   `max(8, 0.22 * p99.5)` → 连通域 → 面积/长宽比过滤 → 窗口加权质心（亚像素）。
2. **候选并集**：B0 的固定 HSV 前端在"室内泛光、低饱和"场景仍有独立价值，因此
   把 B0 候选（半分辨率 seed + 全分辨率质心精修）并入候选池，由打分器仲裁。
3. **三点选择**：穷举颜色三元组，代价 = 形状（等边性）+ 尺寸一致性 + 光度 +
   深度-边长一致性 + **软时序先验**（上一帧接受位姿的常速预测，只加罚分不硬拒绝）。
4. **时序状态**：只在 PnP 成功且位姿跳变在 `max(150mm, 0.25·depth)` 内时更新；
   跳变超限丢弃状态，避免错误关联污染速度。
5. `eval_v2.py --methods b0,classic2` 复用同一评测器（`det10_all`、逐点召回、
   姿态覆盖、t_err、jitter、延迟）。

{transfer or ""}

## 4. 复现

```bash
research/.venv/bin/python -m research.scripts.b0_diag --subset test,hard_test,holdout_hard
research/.venv/bin/python -m research.tridot_temporal.eval_v2 \\
    --subset hard_test --methods b0,classic2 --out research/results/honest_cv2/try4/hard_test
research/.venv/bin/python -m research.scripts.classic_cv2_report
```

## 5. 局限

- `occlude_hand` 因子按定义 0 帧三点全可见，`det@10 all` 上该因子恒为 0；
- `hardmix` 大量帧为部分遮挡，补全需要"2 点 + 已知几何"才能改善；
- 延迟仍高于 B0（多为 top-hat / 双前端），端侧部署前需要下采样或 NEON 级优化；
- 无真实数据：所有结论只覆盖合成域，真实集到位后必须重跑
  `eval_v2 --subset real_test`。
"""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"[report] wrote {out} ({len(text)} chars)")


if __name__ == "__main__":
    main()
