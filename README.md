# ShellShock Live YOLO 检测与标注工具

当前项目使用 YOLO 识别己方坦克、敌方坦克、障碍物和虫洞，并结合风力与弹道求解自动瞄准。

## 启动

```powershell
.\.venv\Scripts\python.exe detect_shellshock_yolo.py
```

- `E`：触发一次截图、YOLO 识别、弹道计算和瞄准。
- `Caps Lock`：切换截图采集模式。
- `Delete`：退出脚本。

所有采集截图保存到 `train/yolo_captures`。

## 人工标注

```powershell
.\.venv\Scripts\python.exe label_yolo_captures.py --all-images --barrel-length 35
```

默认读取 `train/yolo_captures/full`。标注过程中，临时结果写入 `train/yolo_captures` 下的对应目录；按 Enter 确认当前图片后，图片和相关标注会移动到 `train/annotate`：

```text
train/yolo_captures/labels
train/yolo_captures/pose_geometry
train/yolo_captures/metadata
train/yolo_captures/wind/labels
train/yolo_captures/previews
train/annotate/images
train/annotate/labels
train/annotate/pose_geometry
train/annotate/metadata
train/annotate/previews
```

数字键 `0–9` 选择类别；新标注中 `1` 只记录 `self` 的中心关键点，不新增检测框，旧的炮管终点数据保持不变；`2` 仍可标注己方检测框，`4` 标注障碍线。左键新增或选中，右键删除，方向键微调位置，`+/-` 调整半径，`PageUp/PageDown` 切换图片，`Esc` 保存退出。

类别：`0 enemy`、`1 self_center_keypoint`（仅作为新中心关键点快捷键）、`2 self`、`3 obstacle_circle`、`4 obstacle_line`、`5 portal_orange`、`6 portal_blue`、`7 blackhole`、`8 double_damage`、`9 Triple_damage`。

## YOLO 预标注

```powershell
.\.venv\Scripts\python.exe prelabel_yolo.py
```

脚本读取 `train/yolo_captures/full`，把模型预测写入 `train/yolo_captures/labels`。已有覆盖层默认不会被覆盖，需要重新预测时加 `--overwrite`。

## 生成 Pose 训练数据

```powershell
.\.venv\Scripts\python.exe prepare_yolo_pose.py --geometry-dir train/annotate/pose_geometry
```

使用 `train/annotate_check` 中的人工标注生成新的多关键点数据集：

```powershell
.\.venv\Scripts\python.exe prepare_yolo_pose.py --raw-dir train/annotate_check --output-dir train/yolo_pose_dataset_v2 --geometry-dir train/annotate_check/pose_geometry --override-dir train/annotate_check/labels --overwrite
```

Pose 关键点固定为 2 个槽位：`self` 使用 `kp0` 记录中心点；直线挡板使用 `kp0/kp1` 记录两个端点；圆形障碍、传送门、黑洞和增益目标使用 `kp0` 记录圆心、`kp1` 记录左边缘点以计算半径。当前炮管终点尚未标注，因此 `self` 的 `kp1` 不可见。没有几何关键点的类别保留两个不可见槽位。落在截图范围外的真实关键点写为 `0 0 0`，不强行裁剪到边界制造错误监督。

使用 YOLO Pose 训练：

```powershell
.\.venv\Scripts\python.exe train_yolo_pose.py --data train/yolo_pose_dataset_v2/dataset.yaml --model yolo11n-pose.pt --project train/runs --name shellshock_yolo11n_pose_v1 --imgsz 960 --epochs 200 --batch -1 --workers 4
```

显存不足时可将 `--imgsz` 改为 `640`、`--batch` 改为 `4`，或将模型换成 `yolo11n-pose.pt`。当前 smoke training 已验证权重下载和训练链路，结果位于 `train/runs/shellshock_yolo11n_pose_smoke2`；正式训练应使用更长的 epoch 后再比较验证集指标。

## 生成风力和虫洞裁剪

完成全图采集和人工标注后执行：

```powershell
.\.venv\Scripts\python.exe extract_capture_assets.py
```

脚本读取 `train/yolo_captures/full`，并将结果写入同级的 `wind` 与 `wormholes`。可以使用 `--start` 和 `--end` 限定时间范围。

## 截图回放与轨迹调试

回放脚本在 `train/annotate/images` 中默认选择最新图片，并将 `label_yolo_captures.py` 的人工标注作为修正层：

```powershell
.\.venv\Scripts\python.exe replay_shellshock.py --source hybrid
```

`--source` 可选 `hybrid`（默认）、`yolo` 或 `annotation`。结果写入 `train/annotate/replay`，包括预测轨迹、游戏白色虚线检测、合并标注和 JSON 误差报告。

交互调试：

```powershell
.\.venv\Scripts\python.exe replay_shellshock.py --image train/yolo_captures/full/<stem>.png --source hybrid --interactive
```

鼠标左键选择标注，右键删除，方向键移动，数字键添加对应类别；`R` 重新计算，`S` 保存到人工标注文件并生成 `.bak` 备份，`Esc`/`Q` 退出。预测轨迹使用蓝色实线，截图中的游戏轨迹使用白色线段，不绘制红色误差线。

回放窗口会保持原图比例并使用黑边填充，不会拉伸画面；关闭窗口后会直接退出，不会重新弹出。快捷键：`E` 将当前鼠标位置设为目标并立即重算，`R` 反射模式，`H` 虫洞模式，`T` 普通模式，`PageUp/PageDown` 切换高低弧，`V` 显示/隐藏标注，`G` 显示/隐藏游戏虚线，`F` 切换全屏。

## 当前模型

运行入口默认使用：

```text
train/runs/shellshock_yolo11n_pose_v1/weights/best.pt
```

该 Pose 模型同时输出目标框和两个关键点；`replay_shellshock.py`、`detect_shellshock_yolo.py`、`label_yolo_captures.py` 默认使用它。旧检测模型仍可通过命令行显式传入，供历史预标注数据兼容使用。
