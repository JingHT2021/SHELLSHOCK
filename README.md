# ShellShock Live 人工坐标瞄准与数据采集器

程序不再自动识别己方或敌方坦克。由玩家用鼠标记录 `self` 和当前目标；程序保留风向/风力识别和现有带风弹道计算。

## 启动

```powershell
.\.venv\Scripts\python.exe detect_shellshock.py --output-dir output
```

- `Q`：将当前鼠标的游戏客户区坐标记录为己方 `self`。
- `E`：将当前鼠标坐标记录为目标（按 `enemy` 类保存），用已记录的 `self`、当前风和当前力度模式计算并点击瞄准圆盘。
- `PageUp`：切换为严格的 100 力度高弹道；若只有低弹道则跳过，不会点击低角度解。
- `PageDown`：恢复为最低力度弹道。
- `R`：仅截图、识别风并保存训练数据，不点击。
- `Esc`：退出脚本。
- 未找到游戏窗口时，终端显示错误，不会截取桌面。

所有截图只保留游戏画面 `y=0~1800`。原有输出仍保存为 `*_raw.png`、`*_result.json`、`*_annotated.png`；训练数据额外保存到 `train`。

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

按 E 后，终端会输出人工 `self` 到人工目标的弹道。默认使用最低力度；PageUp 使用现有 100 力度解中仰角最高的一条，PageDown 恢复最低力度。角度始终为 0–90° 仰角；`left`/`right` 表示炮管方向。风力或风向未识别时，E 按 0 风继续计算。

## YOLO Detection 数据

每次 R 或 E 保存截图时，都会生成：

```text
train/raw_cropped/20260905_230000.png
train/raw_cropped/20260905_230000.txt
train/annotated/20260905_230000.png
```

`.txt` 使用标准 YOLO Detection 的归一化格式。Q 记录的点生成 `self`（类别 2）框；E 记录的目标按 `enemy`（类别 0）框保存。默认框尺寸集中在 [training_data.py](shellshock_detector/training_data.py) 的 `DEFAULT_BOX_SIZE_AT_REFERENCE = (38.4, 28.8)`，按截图宽度缩放。类别为：`0 enemy`、`1 ally`、`2 self`、`3 obstacle_circle`、`4 obstacle_line`、`5 portal_orange`、`6 portal_blue`、`7 blackhole`。

可对已有训练预览图追加严格识别的粉色障碍物标注：

```powershell
.\.venv\Scripts\python.exe annotate_pink_obstacles.py --dry-run
.\.venv\Scripts\python.exe annotate_pink_obstacles.py
```

默认 HSV 阈值为严格的 `H=147..160, S>=180, V>=180`，只接受圆环或细长条形状，优先避免误标。先使用 `--dry-run` 查看候选数量；实际运行仅追加类别 3/4，且会跳过同类别、高重叠的已有框。

计算使用 1920×1080 标定并按截图宽度缩放；当前风力系数来自一条 94 风实测，结果应作为首发建议。不同武器、弹跳、地形碰撞及版本差异会偏离普通炮弹模型；用多次实际落点可再校准风系数。

## OCR

风向无需外部组件。风力数值使用 Tesseract OCR；仅安装 `pytesseract` 不够，系统还必须有 `tesseract.exe`。如果没有安装，脚本会继续输出风向，但 `wind.value` 为 `null`，并在 `errors` 给出原因。

默认阈值按照 2560×1600 截图校准，并按窗口宽度缩放。当前实现面向普通绿色/红色坦克 UI；更换皮肤、强遮挡或不同 HUD 主题时应补充截图，后续可升级为 YOLO。

## 分辨率选项

默认 `--resolution auto`，直接按实际游戏窗口大小识别。也可选择 `2560x1600` 或 `3840x2160`：

```powershell
.\.venv\Scripts\python.exe detect_shellshock.py --resolution 3840x2160 --output-dir output
```

选择预设不会强制缩放截图；它仅核对实际捕获尺寸，并在 JSON 的 `errors` 中报告不匹配，避免把显示器分辨率与游戏窗口分辨率混淆。
# 4K 与 Windows 缩放

脚本启动时会启用 Windows 的“每显示器 DPI 感知”，因此在 3840×2160 显示器使用 125%/150%/200% 缩放时，窗口坐标也会按真实物理像素获取，不会被缩成约 2560 像素。更新后请彻底退出并重新启动脚本；4K 屏幕建议使用 `--resolution 3840x2160`。
