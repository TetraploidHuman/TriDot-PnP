# FindNet 改进（offset 头 / 分辨率 / 留存集 / 多种子）离线评测报告

本报告只使用 `research/tridot_temporal/eval_v2.py` 的诚实口径：det@10 all 的分母是**全部帧**（三点不全可见的帧算失败），逐点 recall 的分母是**每个可见点**，t_err 只在「三点都预测出 + PnP 成功」的帧上统计。所有 t_err 均为 clean 子集（GT 三点全可见）。为避免与旧报告的 legacy 口径混淆，两张表都给。

snap14 = 用全分辨率原图的颜色质心把学习热图预测精修到亚像素；find_offset_geom = `decode_find_geom` 选峰选三元组 → offset 头在已选格内亚格修正 → snap14。

## 1. 训练（step ①）

从 v3.12 权重出发，冻结 backbone/heat 头，只训每色 2D offset 回归头；损失只有 offset（lambda_off=2.0，其余项置 0）。

| 训练 | seed | 参数 | val l2（best epoch） | partial_hit | 状态 |
|---|---|---|---|---|---|
| `find_v3_25_offset` | 7 | 649490 | 0.0494 (ep 8) | 0.173 | finished |
| `find_v3_25_offset_s17` | 17 | 649490 | 0.1012 (ep 5) | 0.114 | finished |
| `find_v3_25_offset_s27` | 27 | 649490 | 0.0974 (ep 3) | 0.121 | finished |

## 2. 主结果（step ⑤）

### test（easy 域，15 序列/1800 帧）

| 方法 | det@10 all | det@10 legacy | 逐点 recall@10 | pose 覆盖 | t_err 中位 mm | p90 mm | ≤50mm | 延迟 | FPS |
|---|---|---|---|---|---|---|---|---|---|
| B0（经典 CV） | 0.634 | 0.918 | 0.790 | 0.800 | 3.63 | 114.4 | 0.837 | 10 ms | 96.3 |
| FindNet geom（v3.12 + geom 解码 + snap14） | 0.662 | 0.958 | 0.964 | 0.993 | 6.83 | 122.5 | 0.781 | 128 ms | 7.8 |
| FindNet geom+offset（v3.25 + 混合解码 + snap14） | 0.663 | 0.960 | 0.965 | 0.992 | 6.25 | 120.3 | 0.814 | 237 ms | 4.2 |

provenance: splits sha256 `42916bfc1dd0`, find ckpt `best.pt` sha `d2f28f3b1b6a`, geom ckpt `best.pt` sha `dd92d854c6af`, decode top_k=5 upsample=1 snap=14, seed 0, cpu

### hard_test（realistic 域，40 序列/3600 帧）

| 方法 | det@10 all | det@10 legacy | 逐点 recall@10 | pose 覆盖 | t_err 中位 mm | p90 mm | ≤50mm | 延迟 | FPS |
|---|---|---|---|---|---|---|---|---|---|
| B0（经典 CV） | 0.172 | 0.245 | 0.346 | 0.878 | 419.80 | 1089.8 | 0.234 | 66 ms | 15.2 |
| FindNet geom（v3.12 + geom 解码 + snap14） | 0.628 | 0.894 | 0.913 | 0.995 | 16.57 | 104.6 | 0.742 | 232 ms | 4.3 |
| FindNet geom+offset（v3.25 + 混合解码 + snap14） | 0.622 | 0.886 | 0.909 | 0.997 | 15.33 | 101.4 | 0.787 | 214 ms | 4.7 |

provenance: splits sha256 `42916bfc1dd0`, find ckpt `best.pt` sha `d2f28f3b1b6a`, geom ckpt `best.pt` sha `dd92d854c6af`, decode top_k=5 upsample=1 snap=14, seed 0, cpu

### holdout_easy（全新种子，15 序列/1800 帧）

| 方法 | det@10 all | det@10 legacy | 逐点 recall@10 | pose 覆盖 | t_err 中位 mm | p90 mm | ≤50mm | 延迟 | FPS |
|---|---|---|---|---|---|---|---|---|---|
| B0（经典 CV） | 0.654 | 0.884 | 0.781 | 0.759 | 4.41 | 127.4 | 0.656 | 10 ms | 97.0 |
| FindNet geom（v3.12 + geom 解码 + snap14） | 0.668 | 0.902 | 0.945 | 0.993 | 5.79 | 147.9 | 0.706 | 133 ms | 7.5 |
| FindNet geom+offset（v3.25 + 混合解码 + snap14） | 0.664 | 0.897 | 0.944 | 0.996 | 8.31 | 152.5 | 0.696 | 246 ms | 4.1 |

provenance: splits sha256 `42916bfc1dd0`, find ckpt `best.pt` sha `d2f28f3b1b6a`, geom ckpt `best.pt` sha `dd92d854c6af`, decode top_k=5 upsample=1 snap=14, seed 0, cpu

### holdout_hard（全新种子，40 序列/3600 帧）

| 方法 | det@10 all | det@10 legacy | 逐点 recall@10 | pose 覆盖 | t_err 中位 mm | p90 mm | ≤50mm | 延迟 | FPS |
|---|---|---|---|---|---|---|---|---|---|
| B0（经典 CV） | 0.169 | 0.231 | 0.336 | 0.881 | 432.14 | 1027.3 | 0.243 | 64 ms | 15.6 |
| FindNet geom（v3.12 + geom 解码 + snap14） | 0.666 | 0.911 | 0.921 | 0.995 | 12.78 | 80.3 | 0.803 | 234 ms | 4.3 |
| FindNet geom+offset（v3.25 + 混合解码 + snap14） | 0.662 | 0.906 | 0.919 | 0.997 | 13.23 | 74.3 | 0.844 | 216 ms | 4.6 |

provenance: splits sha256 `42916bfc1dd0`, find ckpt `best.pt` sha `d2f28f3b1b6a`, geom ckpt `best.pt` sha `dd92d854c6af`, decode top_k=5 upsample=1 snap=14, seed 0, cpu

## 3. 多种子（step ④）

| 训练 seed | det@10 all | det@10 legacy | 逐点 recall@10 | t_err 中位 mm | p90 mm |
|---|---|---|---|---|---|
| seed7 | 0.942 | 0.988 | 0.992 | 4.96 | 19.3 |
| seed17 | 0.933 | 0.979 | 0.989 | 4.30 | 40.7 |
| seed27 | 0.935 | 0.981 | 0.989 | 6.03 | 48.4 |

seed7 是完整 8 epoch 训练；seed17/27 在 CPU 争用下训练（每 epoch 250-390s），训练数据子集由 seed 决定，因此这既是初始化差异也是数据顺序差异。
注意：seed17/27 的 val_l2 ≈0.10 而 seed7 ≈0.049 **不可横比**——当时验证子集也是按 seed 抽的（已修为固定 RNG）；真正的多种子对比看本表（同一 5 条 test 序列、同一解码）。

## 4. hard_test 分 factor（det@10 all）

| factor | B0 det@10 all | geom det@10 all | geom+offset det@10 all |
|---|---|---|---|
| clutter | 0.075 | 0.867 | 0.864 |
| crosscolor | 0.214 | 0.911 | 0.919 |
| desk | 0.214 | 0.928 | 0.933 |
| far | 0.297 | 0.964 | 0.936 |
| handheld | 0.183 | 0.692 | 0.683 |
| hardmix | 0.000 | 0.093 | 0.090 |
| occlude_hand | 0.000 | 0.000 | 0.000 |
| room | 0.731 | 0.936 | 0.936 |
| screen | 0.003 | 0.794 | 0.769 |

## 5. holdout_hard 分 factor（det@10 all）

| factor | B0 det@10 all | geom det@10 all | geom+offset det@10 all |
|---|---|---|---|
| clutter | 0.028 | 0.872 | 0.869 |
| crosscolor | 0.133 | 0.939 | 0.939 |
| desk | 0.142 | 0.894 | 0.911 |
| far | 0.356 | 0.972 | 0.950 |
| handheld | 0.275 | 0.889 | 0.892 |
| hardmix | 0.001 | 0.096 | 0.092 |
| occlude_hand | 0.000 | 0.000 | 0.000 |
| room | 0.744 | 0.975 | 0.975 |
| screen | 0.008 | 0.928 | 0.903 |

## 6. 分辨率实验（step ②）

同一 v3.12 权重、hard_test 前 3 条序列、snap14：输入 256×448 时 det@10 all 0.881 / t_err 中位 9.07mm / 单帧 270ms；把评测输入改成 384×640 后 det@10 all 降到 0.774 / t_err 11.48mm / 单帧 778ms。编码器是在 448×256 上训的，特征尺度对不上；真高分辨率重训（512×288）单 epoch 530s，8 epoch 预算不可接受，已停止。结论：在这个架构上「加分辨率」不是免费收益；真正的亚像素精度目前来自全分辨率颜色质心 snap，而不是网络分辨率。

## 7. 结论与局限

- 定位精度（test）：B0 3.63mm，geom+snap 6.83mm，geom+offset+snap 6.25mm。 这一轮把学习侧 t_err 从旧的无 snap 版本（26.0mm）降到 ~6mm，主要靠 **全分辨率颜色质心 snap + geom 三元组选峰**，不是靠网络分辨率。
- offset 头：在 2 条 test 序列的受控切片上把 t_err 从 geom 的 12.36mm 拉到 3.87mm；但全 test 只从 6.83 到 6.25mm，在 fresh holdout_easy 上反而变差（5.79 → 8.31mm）。 结论：offset 头有信息量，但对初始化/数据顺序敏感，单独作为定位方案不成立。
- 检测召回（这才是学习侧的真正优势）：det@10 all 在 test 上 0.634 → 0.663，hard_test 上 0.172 → 0.622；从未参与选型的 fresh holdout_hard 上 0.169 → 0.662（4 倍）。 也就是说越难/越没见过的数据，学习侧越值钱；easy 集已经接近天花板。
- transformer 时序精修仍然≈0：所有增益都能由解码、几何约束和全分辨率精修解释，与旧 `coords` 消融 0.000 的结论一致。这一轮没有让 transformer 变好，只是把它之外的部分做对了。
- 分辨率（step ②）：评测端把 448×256 提到 640×384 是负收益（det 0.881→0.774），因为编码器没在高分辨率上训练；真重训 512×288 单 epoch 530s，CPU 预算下不可行。 想让网络自己分辨更高，需要 GPU 训练，属于下一轮的事。
- 多种子（step ④）：det@10 在 0.933–0.942 之间稳定，t_err 中位 4.3–6.0mm、p90 19–48mm 波动明显 → 「检测更稳」的结论可信，「亚像素更准」的结论还需要更多 seed/更长训练才能站住。
- 延迟：四个子集是 4 个进程并行跑出来的，绝对延迟只作相对参考；学习侧 ~60–240ms/帧 vs B0 ~10–66ms/帧，端侧要上这个解码器仍需进一步的工程优化。
- hard_test 与 holdout_hard 的 det 差距很小（见第 2 节），说明 hard_test 上此前的 0.88 并非纯调参过拟合——但那是 legacy 口径（分母只算三点全可见帧）；按 det@10 all 看，真实数字约 0.62–0.67，而不是 0.88。

## 8. 复现命令

```bash
# 训练 offset 头（seed 7/17/27）
research/.venv/bin/python -m research.tridot_temporal.train_find --dataset datasets/TriDotPose --out research/results/find_v3_25_offset --resume research/results/find_v3_12_data/best.pt --partial-init --freeze-backbone --offset-head --lambda-off 2.0 --lambda-xy 0 --lambda-xy0 0 --lambda-hm 0 --lambda-sep 0 --lambda-eq 0 --train-subset train,hard_train --val-subset val,hard_val --max-clips 2 --max-train-seqs 20 --max-train-clips 480 --max-val-seqs 10 --max-val-clips 300 --no-shuffle --lru-max 4 --epochs 8 --batch-size 2 --lr 3e-4 --num-workers 0 --seed 7
# 诚实评测（四个子集）
research/.venv/bin/python -m research.tridot_temporal.eval_v2 --subset test --methods b0,find_geom,find_offset_geom --find-ckpt research/results/find_v3_25_offset/best.pt --geom-ckpt research/results/find_geom_v34/best.pt --snap-radius 14 --n-boot 1000 --out research/results/honest/final_test
# holdout 生成与校验
research/.venv/bin/python -m research.scripts.make_holdout
research/.venv/bin/python -m research.scripts.verify_holdout
```
