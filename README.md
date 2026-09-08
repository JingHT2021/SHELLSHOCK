# ShellShock Live 人工坐标瞄准与数据采集器

程序不再自动识别己方或敌方坦克。由玩家用鼠标记录 `self` 和当前目标；程序保留风向/风力识别和现有带风弹道计算。

## 启动

```powershell
.\.venv\Scripts\python.exe detect_shellshock.py
```

- `Q`：将当前鼠标的游戏客户区坐标记录为己方 `self`。
- `E`：将当前鼠标坐标更新为目标（按 `enemy` 类保存）；只要已用 Q 记录 `self`，每次 E 都会重新截图、识别风/障碍物、计算并点击瞄准圆盘。
- `R`：切换为一次反射模式。炮弹必须在检测到的圆形或直线障碍物上反射一次后命中目标；若当前截图未识别到障碍物，自动退回普通模式并在终端说明。
- `T`：切换为普通射击模式。
- `PageUp`：切换为严格的 100 力度模式，使用所有可达 100 力度解中角度最大的一条；只有一条解时也会使用该解。
- `PageDown`：恢复为最低力度弹道。
- `Esc`：退出脚本。
- 未找到游戏窗口时，终端显示错误，不会截取桌面。

所有截图只保留游戏画面 `y=0~2000`，且只保存训练数据到 `train`；不会创建 `output` 中的运行时截图、JSON 或标注图。

## JSON

```json
{
  "image": {"width": 2560, "height": 1600},
  "self": null,
  "enemies": [],
  "wind": {"value": 20, "direction": "right", "confidence": 0.9},
  "errors": []
}
```

坐标以游戏窗口左上角为原点；Q/E 所记录的坐标直接用于弹道计算。

## 弹道建议

按 E 后，终端会输出人工 `self` 到人工目标的弹道。普通模式保留既有理论解，反射模式搜索圆形和直线障碍物上的一次镜面反射；PageUp 使用现有 100 力度解中角度最高的一条，PageDown 恢复最低力度。理论角度/力度只用于生成候选：执行前会在附近搜索整数角度和整数力度，并重新模拟轨迹，最终点击和输出始终是整数。角度允许为 -90–90°，负值表示向下射击；`left`/`right` 表示炮管水平方向。风力或风向未识别时，E 按 0 风继续计算。

## YOLO Detection 数据

### 多类别人工复核

使用以下命令打开多类别圆形/线段标注器。默认直接读取 `train/yolo_captures/full`：

```powershell
.\.venv\Scripts\python.exe annotate_enemies.py --all-images
```

数字键 `0–9` 选择类别；左键点击已有标注可选中，点击空白处新增，右键删除当前类别下最近的命中标注，`-`/`+` 调整选中圆的半径，方向键微调位置，`A/Q` 向右调整角度（精调 1 度/粗调 5 度），`D/E` 向左调整角度（精调 1 度/粗调 5 度），`Z/C` 调整风力，角度和风力会跨过边界循环并自动切换方向，标题显示角度和风向，`Ctrl+鼠标滚轮` 以鼠标为中心缩放，`PageUp/PageDown` 切换图片，`B` 切换方框预览，`Esc` 保存退出。编辑结果写入 `train/annotation_overrides`，不会修改原始 `.txt`。

选择 `1` 点击炮管终点，选择 `2 self` 点击己方中心；`self` 默认圆半径为普通圆的两倍。选择 `4 obstacle_line` 后左键依次点击两个端点即可标注线段。`--barrel-length 35` 以 2560 像素宽度为基准，程序会按照当前截图宽度自动缩放，并在标题显示实际长度。线段和炮管点写入 `train/pose_geometry`。窗口会保持截图原始宽高比。

可将 Detection 数据转换为双关键点 Pose 数据：

```powershell
.\.venv\Scripts\python.exe prepare_yolo_pose.py --geometry-dir train/pose_geometry
```

输出到 `train/yolo_pose_dataset`，关键点配置为 `[2, 3]`：圆形为“圆心+边缘”，直线为“两端点”，`self` 为“中心+炮管终点”。

准备训练集时可显式指定覆盖层：

```powershell
.\.venv\Scripts\python.exe prepare_yolo_dataset.py --annotation-override-dir train/annotation_overrides
```

类别为：`0 enemy`、`1 ally`、`2 self`、`3 obstacle_circle`、`4 obstacle_line`、`5 portal_orange`、`6 portal_blue`、`7 blackhole`、`8 double_damage`、`9 Triple_damage`。

### YOLO 截图采集模式

```powershell
.\.venv\Scripts\python.exe detect_shellshock_yolo.py
```

按 Caps Lock 切换采集模式。模式开启时按 `E` 仍会执行瞄准计算，同时将全图保存到 `train/yolo_captures/full`，风力区域保存到 `wind`，每个检测到的虫洞保存到 `wormholes`。默认按屏幕左上角固定区域截图：2560 宽取 `2560×1300`，3840 宽取 `3840×1850`，其它宽度按比例推导；可用 `--capture-x/--capture-y/--capture-width/--capture-height` 覆盖。

每次 E 重新计算并保存截图时，都会生成：

```text
train/raw_cropped/20260905_230000.png
train/raw_cropped/20260905_230000.txt
train/annotated/20260905_230000.png
```

`.txt` 使用标准 YOLO Detection 的归一化格式。Q 记录的点生成 `self`（类别 2）框；E 记录的目标按 `enemy`（类别 0）框保存。默认框尺寸集中在 [training_data.py](shellshock_detector/training_data.py) 的 `DEFAULT_BOX_SIZE_AT_REFERENCE = (38.4, 28.8)`，按截图宽度缩放。类别为：`0 enemy`、`1 ally`、`2 self`、`3 obstacle_circle`、`4 obstacle_line`、`5 portal_orange`、`6 portal_blue`、`7 blackhole`。同名 `train/geometry/<timestamp>.json` 是该训练图的完整几何清单：`objects` 保存最终 YOLO 标签中的全部类别，`circles`/`lines` 保存粉色挡板的精确几何，`portals` 保存虫洞圆心、半径与蓝橙配对编号。

可对已有训练预览图追加严格识别的粉色障碍物与蓝橙虫洞标注：

```powershell
.\.venv\Scripts\python.exe annotate_pink_geometry.py --dry-run
.\.venv\Scripts\python.exe annotate_pink_geometry.py
```

粉色挡板 HSV 阈值为严格的 `(0,0,230)`–`(0,0,255)`；虫洞使用橙色 `(14,230,150)`–`(16,255,255)` 与蓝色 `(99,220,150)`–`(102,255,255)`。虫洞只接受有充分颜色圆周支持的圆拟合结果；半径接近的蓝橙圆写入相同 `pair_id`，未配对候选仍保留且 `pair_id` 为 `null`。先使用 `--dry-run` 查看候选数量；实际运行仅追加类别 3–6，保留原有 0/2 和已有 3/4 标签，并写入完整 JSON 与预览图。第二次运行不会重复追加标签。

计算使用 1920×1080 标定并按截图宽度缩放；当前风力系数来自一条 94 风实测，结果应作为首发建议。不同武器、弹跳、地形碰撞及版本差异会偏离普通炮弹模型；用多次实际落点可再校准风系数。

## OCR

风向无需外部组件。风力数值使用 Tesseract OCR；仅安装 `pytesseract` 不够，系统还必须有 `tesseract.exe`。如果没有安装，脚本会继续输出风向，但 `wind.value` 为 `null`，并在 `errors` 给出原因。

默认阈值按照 2560×1600 截图校准，并按窗口宽度缩放。当前实现面向普通绿色/红色坦克 UI；更换皮肤、强遮挡或不同 HUD 主题时应补充截图，后续可升级为 YOLO。

## 分辨率选项

默认 `--resolution auto`，直接按实际游戏窗口大小识别。也可选择 `2560x1600` 或 `3840x2160`：

```powershell
.\.venv\Scripts\python.exe detect_shellshock.py --resolution 3840x2160
```

选择预设不会强制缩放截图；它仅核对实际捕获尺寸，并在 JSON 的 `errors` 中报告不匹配，避免把显示器分辨率与游戏窗口分辨率混淆。
# 4K 与 Windows 缩放

脚本启动时会启用 Windows 的“每显示器 DPI 感知”，因此在 3840×2160 显示器使用 125%/150%/200% 缩放时，窗口坐标也会按真实物理像素获取，不会被缩成约 2560 像素。更新后请彻底退出并重新启动脚本；4K 屏幕建议使用 `--resolution 3840x2160`。
### 先用现有模型预标注，再人工审核

```powershell
.\.venv\Scripts\python.exe prelabel_yolo.py
.\.venv\Scripts\python.exe annotate_enemies.py --all-images --barrel-length 35
.\.venv\Scripts\python.exe prepare_yolo_pose.py
```

`prelabel_yolo.py` 默认读取 `train/yolo_captures/full`，把模型识别结果写到
`train/annotation_overrides`。已有覆盖层不会被覆盖；需要重新预测时加
`--overwrite`。旧模型的 `ally`（旧类别 1）会被忽略，不会误当成炮管终点。

标注器中：`1` 点击炮管终点，`2` 点击己方中心；`4` 连续点击起点和终点生成线段；
选中标注后用方向键逐像素微调，`PageUp/PageDown` 切换图片，标题中的
`RADIUS` 会实时显示 `+/-` 调整后的圆半径。

风力审核结果会保存到 `train/wind_labels`，并按同名文件关联
`train/yolo_captures/wind` 中的风力裁剪图。完成审核后可生成后续 RNN 使用的清单：

```powershell
.\.venv\Scripts\python.exe prepare_wind_rnn_dataset.py
```

输出为 `train/wind_rnn_dataset/manifest.csv`，目标字段为带方向的 `wind_signed`。
