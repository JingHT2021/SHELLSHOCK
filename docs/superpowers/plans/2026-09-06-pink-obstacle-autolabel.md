# 粉色障碍物自动标注 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 扫描已有训练预览图，以保守 HSV 和形状过滤识别粉色圆环、长条挡板，并安全追加 YOLO 类别 3/4 标注。

**Architecture:** 在 `training_data.py` 内聚合可单测的检测、YOLO 框转换、标签合并与预览绘制函数；新增独立 CLI 仅负责编排目录遍历和统计。检测严格使用 OpenCV HSV `H=147..160, S>=180, V>=180`，再以轮廓几何区分圆环与长条。标签合并按类别和 IoU 去重，已有行不会被解析后重新格式化。

**Tech Stack:** Python 3、OpenCV (`cv2`)、NumPy、`unittest`、pytest。

---

## 文件结构

- 修改 `shellshock_detector/training_data.py`：检测配置、像素框转 YOLO、标签读取/合并、预览绘制与批处理 API。
- 新建 `annotate_pink_obstacles.py`：无副作用的参数解析和对 `train/annotated` 的批处理入口。
- 修改 `tests/test_training_data.py`：纯内存的 HSV/几何/标签合并测试。
- 新建 `tests/test_annotate_pink_obstacles.py`：临时目录端到端测试，验证磁盘标签保留与幂等。

### Task 1: 为粉色障碍物检测写失败测试

**Files:**
- Modify: `tests/test_training_data.py`
- Modify: `shellshock_detector/training_data.py`

- [ ] **Step 1: 添加期望的公共 API 与严格阈值测试**

  在 `tests/test_training_data.py` 增加：

  ```python
  import cv2
  import numpy as np

  from shellshock_detector.training_data import PinkObstacleConfig, detect_pink_obstacles


  class PinkObstacleDetectionTests(unittest.TestCase):
      def test_detects_a_strict_pink_circle_as_class_3(self):
          image = np.zeros((400, 600, 3), dtype=np.uint8)
          pink = cv2.cvtColor(np.uint8([[[153, 240, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
          cv2.circle(image, (180, 180), 70, pink, 5)

          detections = detect_pink_obstacles(image)

          self.assertEqual([(item.class_id, item.box) for item in detections], [(3, (107, 107, 146, 146))])

      def test_detects_a_strict_pink_diagonal_line_as_class_4(self):
          image = np.zeros((400, 600, 3), dtype=np.uint8)
          pink = cv2.cvtColor(np.uint8([[[153, 240, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
          cv2.line(image, (80, 300), (360, 150), pink, 8)

          detections = detect_pink_obstacles(image)

          self.assertEqual(len(detections), 1)
          self.assertEqual(detections[0].class_id, 4)
          self.assertGreater(detections[0].box[2], 250)

      def test_rejects_pink_outside_the_configured_hue_range(self):
          image = np.zeros((400, 600, 3), dtype=np.uint8)
          red_pink = cv2.cvtColor(np.uint8([[[178, 240, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
          cv2.circle(image, (180, 180), 70, red_pink, 5)

          self.assertEqual(detect_pink_obstacles(image), [])
          self.assertEqual(PinkObstacleConfig().hsv_lower, (147, 180, 180))
          self.assertEqual(PinkObstacleConfig().hsv_upper, (160, 255, 255))
  ```

- [ ] **Step 2: 运行测试，确认因 API 尚未实现而失败**

  Run: `python -m pytest tests/test_training_data.py -q`

  Expected: FAIL，导入 `PinkObstacleConfig` 或 `detect_pink_obstacles` 失败。

- [ ] **Step 3: 实现最小检测 API**

  在 `shellshock_detector/training_data.py` 增加不可变数据对象和下列函数：

  ```python
  @dataclass(frozen=True)
  class PinkObstacleConfig:
      hsv_lower: tuple[int, int, int] = (147, 180, 180)
      hsv_upper: tuple[int, int, int] = (160, 255, 255)
      reference_width: int = 1920
      circle_min_radius: float = 25.0
      circle_max_radius: float = 350.0
      circle_min_circularity: float = 0.72
      line_min_length: float = 55.0
      line_max_thickness: float = 32.0
      line_min_aspect_ratio: float = 3.0

  @dataclass(frozen=True)
  class ObstacleDetection:
      class_id: int
      box: tuple[int, int, int, int]  # x, y, width, height

  def detect_pink_obstacles(
      image: np.ndarray, config: PinkObstacleConfig = PinkObstacleConfig()
  ) -> list[ObstacleDetection]:
      ...
  ```

  实现细节：根据 `image.shape[1] / config.reference_width` 缩放几何阈值；通过 `cv2.inRange` 获取掩码，仅使用半径为 1 像素（按比例缩放）的闭运算连接圆环笔画；使用 `cv2.findContours(..., cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`。

  对圆候选，计算 `circularity = 4 * pi * area / perimeter**2`，同时要求 `0.80 <= w/h <= 1.25`、半径位于范围内；输出由 `cv2.boundingRect` 得到的整数框，类别为 3。对未归为圆的候选，使用 `cv2.minAreaRect`，要求长边、短边和长宽比满足配置；输出 `cv2.boundingRect`，类别为 4。按 `(class_id, x, y)` 排序返回。

- [ ] **Step 4: 运行检测单元测试，确认通过**

  Run: `python -m pytest tests/test_training_data.py -q`

  Expected: PASS。

- [ ] **Step 5: 若处于 Git 仓库，提交检测功能**

  ```powershell
  git add shellshock_detector/training_data.py tests/test_training_data.py
  git commit -m "feat: detect strict pink obstacle shapes"
  ```

  当前工作目录没有 `.git`，执行时跳过此步并在交付中说明。

### Task 2: 以追加和去重方式安全写入 YOLO 标签

**Files:**
- Modify: `tests/test_training_data.py`
- Modify: `shellshock_detector/training_data.py`

- [ ] **Step 1: 添加 YOLO 框与标签合并的失败测试**

  ```python
  from shellshock_detector.training_data import append_yolo_obstacle_labels, yolo_box_label_line

  class ObstacleLabelMergeTests(unittest.TestCase):
      def test_yolo_box_label_line_normalizes_an_axis_aligned_box(self):
          self.assertEqual(
              yolo_box_label_line(3, (100, 50, 200, 100), 1000, 500),
              "3 0.200000 0.200000 0.200000 0.200000",
          )

      def test_append_preserves_existing_bytes_and_is_idempotent(self):
          existing = "2 0.100000 0.200000 0.020000 0.030000\\n0 0.700000 0.500000 0.020000 0.030000\\n"
          detections = [ObstacleDetection(3, (100, 50, 200, 100)), ObstacleDetection(4, (500, 250, 300, 20))]

          merged, added = append_yolo_obstacle_labels(existing, detections, 1000, 500)
          again, added_again = append_yolo_obstacle_labels(merged, detections, 1000, 500)

          self.assertTrue(merged.startswith(existing))
          self.assertEqual(added, 2)
          self.assertEqual(added_again, 0)
          self.assertEqual(again, merged)
  ```

- [ ] **Step 2: 运行测试，确认它因函数缺失而失败**

  Run: `python -m pytest tests/test_training_data.py -q`

  Expected: FAIL，导入 `yolo_box_label_line` 或 `append_yolo_obstacle_labels` 失败。

- [ ] **Step 3: 实现最小 YOLO 框转换和幂等合并**

  追加如下公共函数：

  ```python
  def yolo_box_label_line(
      class_id: int, box: tuple[int, int, int, int], image_width: int, image_height: int
  ) -> str:
      ...

  def append_yolo_obstacle_labels(
      existing_text: str,
      detections: list[ObstacleDetection],
      image_width: int,
      image_height: int,
      duplicate_iou: float = 0.90,
  ) -> tuple[str, int]:
      ...
  ```

  `yolo_box_label_line` 必须裁剪框到图片范围，拒绝非正尺寸，输出六位小数。合并函数从已有文本读取可解析的五字段 YOLO 行作为去重候选；保留 `existing_text` 原始字节顺序，只有确定未重复的类 3/4 行才追加。重复条件为相同类别且与已有同类框的 IoU 大于或等于 `duplicate_iou`。

- [ ] **Step 4: 运行单元测试，确认通过**

  Run: `python -m pytest tests/test_training_data.py -q`

  Expected: PASS。

- [ ] **Step 5: 若处于 Git 仓库，提交标签合并功能**

  ```powershell
  git add shellshock_detector/training_data.py tests/test_training_data.py
  git commit -m "feat: append deduplicated obstacle labels"
  ```

### Task 3: 使用全部标签生成预览，并提供批处理 API

**Files:**
- Modify: `tests/test_training_data.py`
- Modify: `shellshock_detector/training_data.py`

- [ ] **Step 1: 添加磁盘操作的失败测试**

  ```python
  from pathlib import Path
  import tempfile
  from shellshock_detector.training_data import annotate_pink_obstacle_sample

  class ObstacleSampleTests(unittest.TestCase):
      def test_sample_update_keeps_existing_labels_and_creates_preview(self):
          with tempfile.TemporaryDirectory() as directory:
              root = Path(directory)
              image = np.zeros((400, 600, 3), dtype=np.uint8)
              pink = cv2.cvtColor(np.uint8([[[153, 240, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
              cv2.circle(image, (200, 200), 70, pink, 5)
              source = root / "sample.png"
              label = root / "sample.txt"
              preview = root / "preview.png"
              cv2.imwrite(str(source), image)
              label.write_text("2 0.100000 0.200000 0.020000 0.030000\\n", encoding="utf-8")

              added = annotate_pink_obstacle_sample(source, label, preview)

              self.assertEqual(added, 1)
              self.assertTrue(preview.exists())
              self.assertTrue(label.read_text(encoding="utf-8").startswith("2 0.100000 0.200000 0.020000 0.030000\\n"))
  ```

- [ ] **Step 2: 运行测试，确认函数尚未实现**

  Run: `python -m pytest tests/test_training_data.py -q`

  Expected: FAIL，导入 `annotate_pink_obstacle_sample` 失败。

- [ ] **Step 3: 实现样本更新与预览函数**

  实现：

  ```python
  def annotate_pink_obstacle_sample(
      source_image_path: Path,
      label_path: Path,
      preview_path: Path,
      config: PinkObstacleConfig = PinkObstacleConfig(),
  ) -> int:
      ...
  ```

  函数读取 `source_image_path`，执行检测，读取或创建 `label_path`，调用安全合并函数，且仅当新增行数大于零时写回标签文本。预览从源图复制并绘制标签文件中全部可解析的框：类别 0/2 采用现有黄色；类别 3 使用洋红色 `(255, 0, 255)`，类别 4 使用浅紫色 `(255, 120, 255)`；框上标注 `obstacle_circle` 或 `obstacle_line`。无论是否新增，更新预览供人工核对。

- [ ] **Step 4: 运行单元测试，确认通过**

  Run: `python -m pytest tests/test_training_data.py -q`

  Expected: PASS。

- [ ] **Step 5: 若处于 Git 仓库，提交样本更新 API**

  ```powershell
  git add shellshock_detector/training_data.py tests/test_training_data.py
  git commit -m "feat: render previews for obstacle annotations"
  ```

### Task 4: 新增 CLI 并以副本数据验证批处理

**Files:**
- Create: `annotate_pink_obstacles.py`
- Create: `tests/test_annotate_pink_obstacles.py`
- Modify: `README.md`

- [ ] **Step 1: 写 CLI 编排的失败测试**

  ```python
  import tempfile
  from pathlib import Path
  import cv2
  import numpy as np

  from annotate_pink_obstacles import annotate_directory

  def test_annotate_directory_reports_additions_without_replacing_existing_labels():
      with tempfile.TemporaryDirectory() as directory:
          root = Path(directory)
          annotated = root / "annotated"
          labels = root / "raw_cropped"
          annotated.mkdir()
          labels.mkdir()
          image = np.zeros((400, 600, 3), dtype=np.uint8)
          pink = cv2.cvtColor(np.uint8([[[153, 240, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
          cv2.line(image, (80, 300), (360, 150), pink, 8)
          cv2.imwrite(str(annotated / "scene.png"), image)
          (labels / "scene.txt").write_text("0 0.500000 0.500000 0.020000 0.020000\\n", encoding="utf-8")

          summary = annotate_directory(annotated, labels)

          assert summary == {"images": 1, "added_circle": 0, "added_line": 1, "missing_labels": 0}
          assert (labels / "scene.txt").read_text(encoding="utf-8").startswith("0 0.500000")
  ```

- [ ] **Step 2: 运行测试，确认失败**

  Run: `python -m pytest tests/test_annotate_pink_obstacles.py -q`

  Expected: FAIL，找不到模块或 `annotate_directory`。

- [ ] **Step 3: 实现 CLI 与目录遍历**

  创建 `annotate_pink_obstacles.py`，包含：

  ```python
  def annotate_directory(annotated_dir: Path, labels_dir: Path) -> dict[str, int]:
      ...

  def main() -> int:
      parser = argparse.ArgumentParser()
      parser.add_argument("--annotated-dir", type=Path, default=Path("train/annotated"))
      parser.add_argument("--labels-dir", type=Path, default=Path("train/raw_cropped"))
      args = parser.parse_args()
      summary = annotate_directory(args.annotated_dir, args.labels_dir)
      print(json.dumps(summary, ensure_ascii=False))
      return 0
  ```

  使用 `sorted(annotated_dir.glob("*.png"))` 稳定遍历；对每个图片查找同名 `.txt`。缺少 `.txt` 时累加 `missing_labels` 并不创建文件，避免意外标注无对应训练图的预览；存在标签时原地更新 `.txt`，预览也保存回该 `annotated` 图。统计从实际新增检测的类 ID 得到 `added_circle`、`added_line`。

- [ ] **Step 4: 运行新增测试和全套测试**

  Run: `python -m pytest tests/test_annotate_pink_obstacles.py -q; python -m pytest -q`

  Expected: 两个命令均 PASS。

- [ ] **Step 5: 在真实数据上先做只读预演，然后运行一次**

  先添加 CLI 的 `--dry-run` 开关：它执行识别、打印统计，但绝不写 `.txt` 或预览。运行：

  ```powershell
  .\.venv\Scripts\python.exe annotate_pink_obstacles.py --dry-run
  ```

  检查输出的图片数、缺失标签数和每类候选数；确认候选合理后，运行：

  ```powershell
  .\.venv\Scripts\python.exe annotate_pink_obstacles.py
  ```

  Expected: 原有 0/2 标签仍在；仅追加 3/4；命令输出新增数量。

- [ ] **Step 6: 在 README 记录运行命令与保守阈值策略**

  在 `README.md` 的 YOLO 数据段添加：

  ```powershell
  .\.venv\Scripts\python.exe annotate_pink_obstacles.py --dry-run
  .\.venv\Scripts\python.exe annotate_pink_obstacles.py
  ```

  写明默认阈值是严格的 `H=147..160, S>=180, V>=180`，主要用于高精度初标，需人工查看 `train/annotated` 中的紫色框后再放宽配置。

- [ ] **Step 7: 若处于 Git 仓库，提交 CLI、测试和文档**

  ```powershell
  git add annotate_pink_obstacles.py tests/test_annotate_pink_obstacles.py README.md
  git commit -m "feat: batch annotate pink obstacles"
  ```

## 交付验证清单

- [ ] `python -m pytest -q` 通过。
- [ ] `--dry-run` 不修改标签文件哈希值。
- [ ] 实际运行只追加类 3/4 行，原有类 0/2 每个文件的原始行均保持不变。
- [ ] 第二次实际运行报告新增数为 0，证明幂等。
- [ ] 随机抽查若干 `train/annotated/*.png`，确认紫色框仅覆盖粉色圆环或长条挡板。
