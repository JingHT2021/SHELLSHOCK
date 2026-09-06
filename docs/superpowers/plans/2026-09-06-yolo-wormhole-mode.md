# YOLO 虫洞模式 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一个独立 YOLO11n 瞄准入口，自动识别己方与世界几何，并在 R/H 模式以全局事件回放验证反射和强制虫洞传送。

**Architecture:** 将模型推理、世界几何、弹道事件和整数控制搜索分成纯模块。`detect_shellshock_yolo.py` 仅负责热键、截图、鼠标目标、诊断与安全点击；它不改动既有 `detect_shellshock.py`。

**Tech Stack:** Python 3、Ultralytics YOLO、OpenCV、现有 `keyboard`/Win32 截图与瞄准模块、unittest/pytest。

---

## 文件结构

- Create: `shellshock_detector/yolo_runtime.py` — 验证模型类别、调用 YOLO、输出稳定的检测框。
- Create: `shellshock_detector/world_geometry.py` — 由检测框构建坦克点、障碍物、虫洞及一对一配对。
- Create: `shellshock_detector/projectile_events.py` — 计算最早事件、虫洞传送和状态机回放。
- Create: `shellshock_detector/global_solver.py` — 枚举整数操控并依据模式约束挑选弹道。
- Create: `detect_shellshock_yolo.py` — 新的独立命令行/热键入口。
- Create: `tests/test_yolo_runtime.py`、`tests/test_world_geometry.py`、`tests/test_projectile_events.py`、`tests/test_global_solver.py`、`tests/test_yolo_hotkeys.py`。
- Modify: `README.md` — 追加独立 YOLO 入口、权重前提和键位说明，不修改旧入口说明。

### Task 1: 定义世界几何和虫洞配对

**Files:** Create `tests/test_world_geometry.py`; create `shellshock_detector/world_geometry.py`.

- [ ] **Step 1: 写入失败测试，固定检测框到世界对象的契约**

```python
from shellshock_detector.world_geometry import DetectionBox, build_world

def test_build_world_pairs_nearest_equal_radius_orange_and_blue_portals():
    world = build_world([
        DetectionBox("self", 100, 200, 40, 30, 0.95),
        DetectionBox("portal_orange", 180, 280, 80, 80, 0.90),
        DetectionBox("portal_blue", 700, 310, 82, 82, 0.91),
    ], 1000, 700)
    assert world.self_position == (120, 215)
    assert len(world.portal_pairs) == 1
    assert world.portal_pairs[0].orange.center == (220, 320)
    assert world.portal_pairs[0].blue.center == (741, 351)

def test_build_world_excludes_ambiguous_or_unmatched_portals():
    world = build_world([
        DetectionBox("portal_orange", 100, 100, 80, 80, 0.9),
        DetectionBox("portal_blue", 400, 100, 80, 80, 0.9),
        DetectionBox("portal_blue", 700, 100, 80, 80, 0.9),
    ], 1000, 700)
    assert world.portal_pairs == []
    assert world.unpaired_portals == 3
```

- [ ] **Step 2: 验证测试确因模块不存在而失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_world_geometry.py -q`

Expected: `ModuleNotFoundError: No module named 'shellshock_detector.world_geometry'`.

- [ ] **Step 3: 实现最小不可变世界模型**

```python
@dataclass(frozen=True)
class DetectionBox:
    name: str
    x: float
    y: float
    width: float
    height: float
    confidence: float

@dataclass(frozen=True)
class Portal:
    color: Literal["orange", "blue"]
    center: tuple[float, float]
    radius: float

@dataclass(frozen=True)
class PortalPair:
    orange: Portal
    blue: Portal

def build_world(boxes: list[DetectionBox], image_width: int, image_height: int) -> World:
    # 严格按名称筛选；圆形取框中心与平均半径；线段取长轴中线。
    # 仅接受唯一最高置信度 self；按半径误差排序，仅唯一最优时建立橙蓝一对。
```

包含圆形障碍、线性障碍和 `unpaired_portals`；半径误差阈值作为 `MAX_PORTAL_RADIUS_RELATIVE_ERROR = 0.15` 常量。线段、圆形与虫洞框被裁剪到图像边界；无有效 self 返回 `None`。

- [ ] **Step 4: 扩充测试覆盖唯一 self、圆/线几何和半径阈值**

```python
def test_build_world_returns_none_self_when_top_confidences_tie():
    world = build_world([
        DetectionBox("self", 0, 0, 20, 20, 0.9),
        DetectionBox("self", 50, 0, 20, 20, 0.9),
    ], 100, 100)
    assert world.self_position is None
```

- [ ] **Step 5: 运行测试并确认通过**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_world_geometry.py -q`

Expected: all tests pass.

- [ ] **Step 6: 提交该独立纯模块**

Run: `git add shellshock_detector/world_geometry.py tests/test_world_geometry.py && git commit -m "feat: build YOLO world geometry"`

Expected: one commit containing only these two files.

### Task 2: YOLO 运行时适配器与类别安全门

**Files:** Create `tests/test_yolo_runtime.py`; create `shellshock_detector/yolo_runtime.py`.

- [ ] **Step 1: 写入失败测试，使用假模型而非加载真实权重**

```python
from shellshock_detector.yolo_runtime import REQUIRED_CLASSES, validate_class_names

def test_validate_class_names_accepts_required_sparse_ids():
    names = {0: "enemy", 2: "self", 3: "obstacle_circle", 4: "obstacle_line",
             5: "portal_orange", 6: "portal_blue"}
    assert validate_class_names(names) == REQUIRED_CLASSES

def test_validate_class_names_reports_missing_required_name():
    with pytest.raises(ValueError, match="portal_blue"):
        validate_class_names({0: "enemy", 2: "self"})
```

- [ ] **Step 2: 验证失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_yolo_runtime.py -q`

Expected: import failure for `yolo_runtime`.

- [ ] **Step 3: 实现延迟加载的运行时接口**

```python
REQUIRED_CLASSES = frozenset({"self", "obstacle_circle", "obstacle_line", "portal_orange", "portal_blue"})

def validate_class_names(names: dict[int, str] | list[str]) -> frozenset[str]:
    present = set(names.values()) if isinstance(names, dict) else set(names)
    missing = sorted(REQUIRED_CLASSES - present)
    if missing:
        raise ValueError("YOLO weight is missing required classes: " + ", ".join(missing))
    return REQUIRED_CLASSES

class YoloDetector:
    def detect(self, image: np.ndarray) -> list[DetectionBox]:
        # 使用 model.predict(source=image, verbose=False, conf=self.confidence)。
        # 只将具名类别转换为 DetectionBox，并按 name、坐标、置信度确定性排序。
```

构造函数内部导入 `ultralytics.YOLO`，使纯单元测试不依赖 GPU 或实际权重。类别映射通过模型 `names` 校验后保存。

- [ ] **Step 4: 添加结果转换、置信度过滤和确定性排序测试**

```python
def test_detector_discards_unknown_class_and_low_confidence(fake_model, image):
    detector = YoloDetector("fake.pt", confidence=0.6, model_factory=lambda _: fake_model)
    assert [box.name for box in detector.detect(image)] == ["self", "portal_orange"]
```

- [ ] **Step 5: 运行测试并确认通过**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_yolo_runtime.py -q`

Expected: all tests pass without downloading a model.

- [ ] **Step 6: 提交**

Run: `git add shellshock_detector/yolo_runtime.py tests/test_yolo_runtime.py && git commit -m "feat: add checked YOLO runtime"`

Expected: one commit containing only Task 2 files.

### Task 3: 全局事件与虫洞传送回放

**Files:** Create `tests/test_projectile_events.py`; create `shellshock_detector/projectile_events.py`.

- [ ] **Step 1: 写入失败测试，定义传送后保持速度与相对位置**

```python
from shellshock_detector.projectile_events import advance_through_world
from shellshock_detector.world_geometry import Portal, PortalPair, World

def test_portal_entry_preserves_velocity_and_relative_position():
    pair = PortalPair(Portal("orange", (100, 100), 20), Portal("blue", (500, 300), 20))
    replay = advance_through_world((0, 110), (100, 0), (0, 0), World(portal_pairs=[pair]), 2.0)
    assert replay.events[0].kind == "portal"
    assert replay.events[0].exit_point == pytest.approx((500, 310))
    assert replay.events[0].velocity_after == pytest.approx((100, 0))

def test_replay_rejects_obstacle_that_occurs_before_portal():
    replay = advance_through_world((0, 100), (100, 0), (0, 0), blocked_world, 2.0)
    assert replay.terminal_kind == "obstacle"
    assert replay.portal_count == 0
```

- [ ] **Step 2: 验证失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_projectile_events.py -q`

Expected: import failure for `projectile_events`.

- [ ] **Step 3: 实现最早事件计算与状态机**

```python
@dataclass(frozen=True)
class TrajectoryEvent:
    kind: Literal["portal", "obstacle", "target", "out_of_bounds", "event_limit"]
    time: float
    point: Vector
    exit_point: Vector | None = None
    velocity_after: Vector | None = None

def advance_through_world(start: Vector, velocity: Vector, acceleration: Vector,
                          world: World, max_time: float, target: Vector | None = None,
                          allow_reflection: bool = False) -> Replay:
    # 对每一段求目标、圆、线、虫洞的正时间交点；相同时间的不同事件为 ambiguous。
    # 按最早事件推进。虫洞用 entry-to-exit 的 offset 映射并保留当前速度。
```

使用 `reflection.first_collision` 的几何原则，但实现独立事件函数，以便同一段同时比较所有障碍、所有虫洞和目标。虫洞出口后的 `PORTAL_EXIT_EPSILON` 必须计入剩余时间，并以 `MAX_EVENTS = 32` 防止无限循环。

- [ ] **Step 4: 添加多次触发、出口后受阻、同一时刻歧义和事件上限测试**

```python
def test_portal_can_be_triggered_more_than_once():
    replay = advance_through_world((0, 0), (120, 0), (0, 0), looping_but_finite_world, 5.0)
    assert replay.portal_count == 2

def test_replay_stops_at_event_limit_for_portal_loop():
    replay = advance_through_world((0, 0), (100, 0), (0, 0), infinite_loop_world, 99)
    assert replay.terminal_kind == "event_limit"
```

- [ ] **Step 5: 运行测试并确认通过**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_projectile_events.py -q`

Expected: all tests pass.

- [ ] **Step 6: 提交**

Run: `git add shellshock_detector/projectile_events.py tests/test_projectile_events.py && git commit -m "feat: replay global projectile events"`

Expected: one commit containing only Task 3 files.

### Task 4: 整数控制全局求解与模式约束

**Files:** Create `tests/test_global_solver.py`; create `shellshock_detector/global_solver.py`.

- [ ] **Step 1: 写入失败测试，先锁定 H 的强制传送规则**

```python
from shellshock_detector.global_solver import solve_integer_shot

def test_wormhole_mode_rejects_direct_hit_without_portal_event():
    result = solve_integer_shot(source=(0, 0), target=(200, 0), world=empty_world,
                                wind_value=0, wind_direction="right", image_width=1920,
                                mode="wormhole")
    assert result["status"] == "unreachable"
    assert result["reason"] == "wormhole-required"

def test_wormhole_mode_selects_integer_shot_with_portal_event():
    result = solve_integer_shot(source=(0, 0), target=(700, 0), world=paired_portal_world,
                                wind_value=0, wind_direction="right", image_width=1920,
                                mode="wormhole")
    assert result["status"] == "reachable"
    assert isinstance(result["angle_degrees"], int)
    assert result["portal_count"] >= 1
```

- [ ] **Step 2: 验证失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_global_solver.py -q`

Expected: import failure for `global_solver`.

- [ ] **Step 3: 实现控制量枚举和排序**

```python
def solve_integer_shot(source: Point, target: Point, world: World, wind_value: float,
                       wind_direction: str, image_width: int,
                       mode: Literal["normal", "reflection", "wormhole"],
                       force_power: int | None = None) -> dict[str, object]:
    # 枚举 direction in ("left", "right"), angle 0..90、power 1..100（force_power 时仅该力度）。
    # 将控制量转为初速度；调用 advance_through_world；严格依 mode 筛选。
    # 按 (target_error, power, angle, direction) 返回 JSON 安全字典。
```

R 仅接受 `reflection_count == 1` 且成功到达目标的回放；H 仅接受 `portal_count >= 1`；普通模式仅接受零非法障碍终止。候选只在目标命中半径内命中；无候选统一返回 `{"status": "unreachable", "reason": ...}`，绝不返回未验证的理论解。

- [ ] **Step 4: 添加全局遮挡、一次反射、额外碰撞和固定力度排序测试**

```python
def test_reflection_mode_rejects_second_obstacle_after_legal_bounce():
    result = solve_integer_shot(source, target, two_obstacle_world, 0, "right", 1920, "reflection")
    assert result["status"] == "unreachable"

def test_normal_mode_rejects_target_behind_first_obstacle():
    result = solve_integer_shot(source, target, blocked_world, 0, "right", 1920, "normal")
    assert result["status"] == "unreachable"
```

- [ ] **Step 5: 运行测试并确认通过**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_global_solver.py -q`

Expected: all tests pass.

- [ ] **Step 6: 提交**

Run: `git add shellshock_detector/global_solver.py tests/test_global_solver.py && git commit -m "feat: solve verified global shots"`

Expected: one commit containing only Task 4 files.

### Task 5: 新入口热键、E 目标与安全点击

**Files:** Create `tests/test_yolo_hotkeys.py`; create `detect_shellshock_yolo.py`.

- [ ] **Step 1: 写入失败测试，验证模式状态与不点击策略**

```python
from detect_shellshock_yolo import select_mode, should_click

def test_h_selects_wormhole_mode():
    assert select_mode("h") == "wormhole"

def test_unreachable_solution_never_clicks():
    assert not should_click({"status": "unreachable"}, (100, 100), (1920, 1080))

def test_reachable_solution_inside_client_clicks():
    assert should_click({"status": "reachable"}, (100, 100), (1920, 1080))
```

- [ ] **Step 2: 验证失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_yolo_hotkeys.py -q`

Expected: import failure for `detect_shellshock_yolo`.

- [ ] **Step 3: 实现可注入的新入口流程**

```python
def select_mode(key: str) -> str:
    return {"t": "normal", "r": "reflection", "h": "wormhole"}[key]

def should_click(solution: dict[str, object], click: tuple[int, int], client_size: tuple[int, int]) -> bool:
    return solution.get("status") == "reachable" and 0 <= click[0] < client_size[0] and 0 <= click[1] < client_size[1]

def aim_yolo_target(screen_target: tuple[int, int], detector: YoloDetector, mode: str, ...):
    # 截图一次；由 detector.detect(image) 和 build_world 取得 self/几何；E 的鼠标位置转客户区目标。
    # 调用 solve_integer_shot；只有 should_click 为真时才激活窗口并执行 click。
```

`main()` 接受 `--weights`、`--confidence`、`--resolution` 和 `--max-events`；注册 E/R/H/T/PageUp/PageDown/Esc。所有错误在终端输出而不注入鼠标动作。旧入口不导入也不修改。

- [ ] **Step 4: 添加单一 self 缺失、目标越界、H 无传送和 R 合法解的流程测试**

```python
def test_aim_yolo_target_does_not_click_when_self_is_missing(fake_detector, fake_capture):
    _, click = aim_yolo_target((400, 300), fake_detector, "normal", capture=fake_capture)
    assert click is None
```

- [ ] **Step 5: 运行测试并确认通过**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_yolo_hotkeys.py -q`

Expected: all tests pass without ShellShock 窗口或真实模型。

- [ ] **Step 6: 提交**

Run: `git add detect_shellshock_yolo.py tests/test_yolo_hotkeys.py && git commit -m "feat: add YOLO wormhole hotkeys"`

Expected: one commit containing only Task 5 files.

### Task 6: 文档、回归与人工验证

**Files:** Modify `README.md`; modify or create only tests required by failures.

- [ ] **Step 1: 写入失败的 README 契约测试或更新入口测试文本**

```python
def test_readme_lists_independent_yolo_entry_and_h_wormhole_mode():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "detect_shellshock_yolo.py" in text
    assert "H" in text and "至少" in text and "虫洞" in text
```

- [ ] **Step 2: 验证失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_yolo_hotkeys.py -q`

Expected: failure until README is updated.

- [ ] **Step 3: 更新 README 的独立章节**

```markdown
## YOLO 虫洞瞄准器（独立入口）

```powershell
.\.venv\Scripts\python.exe detect_shellshock_yolo.py --weights train/runs/shellshock_yolo11n_v1/weights/best.pt
```

- `E`：鼠标指向敌军后，识别己方和场景并计算瞄准。
- `R`：恰好一次反射后命中。
- `H`：至少一次虫洞传送后命中。
```

明确权重类别门、无有效解不点击、H 中障碍物会阻断路径，以及旧 `detect_shellshock.py` 不受影响。

- [ ] **Step 4: 运行新增与既有回归测试**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_world_geometry.py tests/test_yolo_runtime.py tests/test_projectile_events.py tests/test_global_solver.py tests/test_yolo_hotkeys.py tests/test_reflection.py tests/test_app.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: 运行完整套件**

Run: `./.venv/Scripts/python.exe -m pytest -q`

Expected: all tests pass; record any pre-existing failure separately rather than weakening tests.

- [ ] **Step 6: 人工安全检查**

Run: `./.venv/Scripts/python.exe detect_shellshock_yolo.py --help`

Expected: help lists `--weights`, `--confidence`, and `--max-events`; no game window or mouse action occurs.

- [ ] **Step 7: 提交文档与验证测试**

Run: `git add README.md tests/test_yolo_hotkeys.py && git commit -m "docs: explain YOLO wormhole controls"`

Expected: one commit containing only documentation and Task 6 test edits.

## 计划自检

- 规格覆盖：独立入口（Task 5）、YOLO 类别安全门（Task 2）、检测框世界几何与虫洞配对（Task 1）、多次传送和循环保护（Task 3）、R/H 全局路径约束与整数控制（Task 4）、文档及完整回归（Task 6）。
- 类型一致性：`DetectionBox` 由 `YoloDetector.detect()` 输出，`build_world()` 接收它并产生 `World`，`advance_through_world()` 和 `solve_integer_shot()` 接收 `World`，入口仅消费最终 JSON 安全结果。
- 范围：不改动 `detect_shellshock.py`、既有 `app.py` 或既有颜色阈值虫洞流程；新版本只复用其窗口、风、DPI 与瞄准盘的稳定基础设施。
