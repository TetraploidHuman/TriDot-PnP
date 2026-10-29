# 真实视频标注工具（label_tool）

给 `datasets/TriDotPose/real/videos/` 下的真实采集视频做三点标注，导出可直接进训练/评测的数据。

## 背景：源码丢失后的恢复方式

这个包的 `*.py` 源码在仓库里已丢失（只剩 `__pycache__/*.pyc`），所以：

- **服务端**（`engine` / `app` / `__main__`）由 `research/pyc_import.py` 直接从 `.pyc` 加载，
  无需反编译；`research/label_tool/__init__.py` 会自动安装该 import hook。
- **前端**（`static/index.html`、`static/style.css`、`static/app.js`）是重写的：
  原静态资源随源码一起丢失，`/` 一直 500。重写后的前端只依赖下面这套 HTTP API。

## 启动

```bash
research/.venv/bin/python -m research.label_tool --port 8899
# 浏览器打开 http://127.0.0.1:8899/
```

可选 `--stride 10` 改默认抽帧步长（也可在页面里改）。

## 工作流

1. 选择视频 → 设定抽帧步长 → 新建会话。服务端对每个抽帧跑 **B0 自动检测**并预填三点。
2. 检查：通常直接按 `A` 接受；点错时在画布上点击/拖动 R/G/B 三点再接受（每点保存一次，
   服务端要求三点齐全）。
3. 三点看不全/画面模糊的帧按 `R` 拒绝（拒绝帧不进导出）。
4. 标完后点“导出标注”。导出会：
   - 写稀疏标注 `datasets/TriDotPose/real/annotations/<seq>.json`（逐帧 9 个数字的 keypoints + 可见性 + 方法）
   - 写训练用紧凑标注 `datasets/TriDotPose/synthetic/annotations/<seq>.json`
   - 把视频复制到 `datasets/TriDotPose/synthetic/videos/`
   - 把 `<seq>` 加入 `datasets/TriDotPose/meta/splits.json` 的 `real_train`
   - 返回训练提示：`--train-subset train,hard_train,real_train`
5. 会话只存在服务端内存里，**重启前务必导出**。

快捷键：`←/→` 换帧，`A` 接受，`R` 拒绝，`1/2/3` 锁定颜色，`Esc` 解除锁定。
带参数打开可直接进会话：`http://127.0.0.1:8899/?video=smoke_real_desk.mp4&stride=10`

## HTTP API（前端与测试都依赖这套契约）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | `{ok, find_enabled, device, real_videos}` |
| GET | `/api/videos` | `{videos: [文件名…]}` |
| POST | `/api/session` | `{video, stride}` → `{id, seq_id, n_frames, width, height, counts}` |
| GET | `/api/session/<id>/summary` | n_frames/fps/stride/cursor/counts |
| GET | `/api/session/<id>/frame/<i>` | 该帧状态：`status`、`pred{ok,method,points,scores}`、`manual` |
| GET | `/api/session/<id>/frame/<i>.jpg` | 帧图像（`?image=1` 为带状态文字的可视化图） |
| POST | `/api/session/<id>/action` | `{action, index, points?}`；action=`goto`/`set_points`/`accept`/`reject` |
| POST | `/api/session/<id>/export` | 落盘标注并更新 `splits.json` |
| POST | `/api/upload` | multipart 字段 `upload.mp4`，上传新视频 |

注意：所有 action 用 **`index`**（会话内帧序号），不是源视频帧号；`set_points` 必须一次给齐三点。

## 已知限制

- **相机内参**：导出默认写 `fx=fy=1280, cx=639.5, cy=359.5`（由 1280×720、HFOV≈53° 推得），
  并不是真机标定值。真实数据的米制距离误差主要来自这里，采集时应另做一次内参标定，
  标注后用脚本改写 `camera` 块（或后续给 label_tool 加 `--fx/--fy`）。
- `find_enabled=false`：恢复后的服务端没加载 FindNet，自动预填只有 B0；这不影响标注
  （B0 在真实视频上的漏检靠手动补点）。
- 无撤销/清除手动点的 action；要重来只能重新 `set_points`。

## 测试

```bash
research/.venv/bin/python -m unittest research.tests.test_label_api -v
```

测试会起一个临时端口、用 smoke 视频走完 建会话→改点→接受→拒绝→导出 全流程，
并断言导出的 JSON 结构可被训练管线读取；结束后恢复 `splits.json` 与所有被写入的文件。
