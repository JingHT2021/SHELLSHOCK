# 模块职责索引

本表按当前源码逐文件列出职责。`shellshock/` 是共享实现包；根目录脚本负责交互、命令行和训练流程。空的 `__init__.py` 只标记 Python 包；事件目录的同名文件还包含规则，已单独说明。

## 根目录入口

| 文件 | 职责 |
| --- | --- |
| `detect_shellshock_yolo.py` | 实时游戏截图、识别与瞄准入口；组织求解、诊断输出和桌面交互。 |
| `replay_shellshock.py` | 离线截图重放；合并人工标注、求解和绘制轨迹，提供交互调参及校准样本管理。 |
| `label_yolo_captures.py` | 多类别圆、线和关键点交互标注；维护标签、人工几何、风力及相关元数据。 |

## 数据与训练工具

| 文件 | 职责 |
| --- | --- |
| `extract_capture_assets.py` | 从已完成整帧截图生成风力和传送门裁剪资源。 |
| `extract_capture_crops.py` | 根据每图 YOLO 标签补提取风力与传送门裁剪，并提供数据审计。 |
| `prelabel_yolo.py` | 用 YOLO Pose 生成供人工复核的预标注覆盖文件。 |
| `prepare_yolo_pose.py` | 从人工标注准备按来源分组划分的 Pose 数据集和配置。 |
| `train_yolo_pose.py` | 训练和验证多类别 YOLO Pose 模型，整理训练指标。 |
| `measure_pose_error.py` | 在验证数据上匹配预测对象并统计关键点像素误差。 |
| `migrate_self_keypoints.py` | 为旧人工标注补齐坦克中心与炮口字段，保留已有值并同步相关标签。 |
| `prepare_wind_digit_dataset.py` | 将风力 HUD 裁剪与标签转换为分组划分的单数字训练样本。 |
| `train_wind_digit_model.py` | 训练、预测和评估固定字体风力数字分类网络。 |

## 共享包逐文件说明

### `shellshock/`

| 文件 | 职责 |
| --- | --- |
| `shellshock/__init__.py` | 共享 Python 包标记，无额外业务逻辑。 |

### `shellshock/adapters/`

| 文件 | 职责 |
| --- | --- |
| `shellshock/adapters/__init__.py` | Python 包标记。 |
| `shellshock/adapters/dpi.py` | 启用 Windows 每显示器 DPI 感知，把客户区坐标转换为物理像素截图边界。 |
| `shellshock/adapters/windows.py` | 查找游戏窗口、读取客户区位置、截取画面、确认活动窗口并执行屏幕点击。 |

### `shellshock/annotations/`

| 文件 | 职责 |
| --- | --- |
| `shellshock/annotations/__init__.py` | Python 包标记。 |
| `shellshock/annotations/bundle.py` | 优先从图片所在位置查找同名标注与元数据，兼容历史数据目录。 |
| `shellshock/annotations/conversion.py` | 在 YOLO 框、编辑器标注和 World 之间转换；合并人工增改删，读写人工场景。 |

### `shellshock/application/`

| 文件 | 职责 |
| --- | --- |
| `shellshock/application/__init__.py` | Python 包标记。 |
| `shellshock/application/scene.py` | 统一分析实时或离线图像，组合检测、风力、人工标注和几何诊断。 |
| `shellshock/application/solver.py` | 校验公共求解参数，分派普通弹道或统一事件路线搜索，并补全模式信息。 |

### `shellshock/capture/`

| 文件 | 职责 |
| --- | --- |
| `shellshock/capture/__init__.py` | Python 包标记。 |
| `shellshock/capture/storage.py` | 保存整帧截图、检测框裁剪资源和发射元数据，提供固定区域截图。 |

### `shellshock/config/`

| 文件 | 职责 |
| --- | --- |
| `shellshock/config/__init__.py` | Python 包标记。 |
| `shellshock/config/capture.py` | 集中定义截图客户区的物理像素边界常量。 |
| `shellshock/config/paths.py` | 解析项目、训练数据、权重和日志路径；支持环境变量与关联 worktree 的原数据目录。 |
| `shellshock/config/solver.py` | 定义命中容差、反射接触限制、整数邻域及路线和重放预算。 |

### `shellshock/datasets/`

| 文件 | 职责 |
| --- | --- |
| `shellshock/datasets/__init__.py` | Python 包标记。 |
| `shellshock/datasets/export.py` | 导出人工位置对应的 YOLO 样本、标签和预览；检测粉色障碍并追加去重标签。 |
| `shellshock/datasets/pose.py` | 把检测框、圆线几何及人工关键点转换为双关键点 Pose 标签，并保留不可见状态。 |
| `shellshock/datasets/yolo.py` | 校验整理检测数据、合并去重语料、划分数据集、生成五折配置并导出待复核样本。 |

### `shellshock/domain/`

| 文件 | 职责 |
| --- | --- |
| `shellshock/domain/__init__.py` | Python 包标记。 |
| `shellshock/domain/detection.py` | 定义检测对象、风力和整帧检测结果的数据类型。 |
| `shellshock/domain/world.py` | 定义与 UI 和识别无关的不可变世界、检测框、关键点、障碍、传送门和倍率区域。 |

### `shellshock/interaction/`

| 文件 | 职责 |
| --- | --- |
| `shellshock/interaction/__init__.py` | Python 包标记。 |
| `shellshock/interaction/aiming.py` | 根据角度和力度计算瞄准圆盘的屏幕点击位置，并检查输入。 |
| `shellshock/interaction/results.py` | 保存已验证候选的展示信息，管理当前结果、候选切换和点击回调。 |

### `shellshock/math2d/`

| 文件 | 职责 |
| --- | --- |
| `shellshock/math2d/__init__.py` | Python 包标记。 |
| `shellshock/math2d/ballistics.py` | 计算带风普通连续弹道、固定力度解和小邻域整数修正，并格式化弹道结果。 |
| `shellshock/math2d/continuous.py` | 提供连续时间途经点方程、到达速度、反射速度和保守可达区域判断。 |
| `shellshock/math2d/geometry.py` | 计算恒加速度轨迹、圆和线段交点、包围盒及到目标的连续最近距离。 |
| `shellshock/math2d/shots.py` | 求恒加速度下固定初速的弹道分支和最低所需速度，保留轨迹结果结构。 |

### `shellshock/perception/`

| 文件 | 职责 |
| --- | --- |
| `shellshock/perception/__init__.py` | Python 包标记。 |
| `shellshock/perception/circle_fit.py` | 在检测 ROI 中拟合标准圆，以颜色、边缘和质量约束拒绝证据不足的结果。 |
| `shellshock/perception/color_geometry.py` | 提取粉色圆线障碍及蓝橙传送门几何，生成相关标签、存储几何并绘制预览。 |
| `shellshock/perception/digit_model.py` | 定义训练与运行时共享的风力数字卷积网络结构。 |
| `shellshock/perception/digits.py` | 延迟加载训练好的 HUD 数字模型，识别固定字体数字串并返回置信度。 |
| `shellshock/perception/guide.py` | 从截图提取浅色虚线瞄准引导，返回引导轨迹及识别信息。 |
| `shellshock/perception/self_center.py` | 在己方坦克 ROI 内验证成对绿色履带横轨，拟合中间轮中心；证据不足时返回空值。 |
| `shellshock/perception/wind.py` | 定位风力面板，结合方向识别和数字识别生成风力结果。 |
| `shellshock/perception/world.py` | 把检测转换为世界几何，融合图像拟合、Pose 和框回退，配对传送门并输出诊断。 |
| `shellshock/perception/yolo.py` | 加载 Ultralytics 模型、校验类别，把推理框和关键点转换为统一 DetectionBox。 |

### `shellshock/physics/`

| 文件 | 职责 |
| --- | --- |
| `shellshock/physics/__init__.py` | Python 包标记。 |
| `shellshock/physics/engine.py` | 按事件顺序重放精确飞行段，检查意外碰撞、命中与净空，并累计倍率区域收益。 |
| `shellshock/physics/launch.py` | 由坦克中心、游戏角度和炮管长度计算实际炮口发射位置。 |

### `shellshock/physics/events/` 及其子目录

| 文件 | 职责 |
| --- | --- |
| `shellshock/physics/events/__init__.py` | 事件规则子包标记。 |
| `shellshock/physics/events/blackhole/__init__.py` | 黑洞事件包及禁止进入的半径规则；当前未模拟引力吸引。 |
| `shellshock/physics/events/portal/__init__.py` | 传送门事件包及配对映射、保持速度和入口偏移的传送位移。 |
| `shellshock/physics/events/reflection/__init__.py` | 反射事件包及接触位置、入射条件检查；不缩小实际反射圆半径。 |
| `shellshock/physics/events/reward/__init__.py` | 倍率事件包及触发半径规则；每个对象只累计一次由引擎落实。 |

### `shellshock/planning/`

| 文件 | 职责 |
| --- | --- |
| `shellshock/planning/__init__.py` | Python 包标记。 |
| `shellshock/planning/iteration.py` | 轮流消费有限候选迭代器，避免单一路线占满搜索预算。 |
| `shellshock/planning/layer_a.py` | 沿事件路线传播包围区域，使用保守连续可达外界筛除不可能路线。 |
| `shellshock/planning/layer_b.py` | 生成解析途经点种子和固定接触点的连续反射分支。 |
| `shellshock/planning/layer_c.py` | 围绕连续种子生成局部整数角度和力度候选，交错遍历分支。 |
| `shellshock/planning/normal.py` | 生成普通连续候选并重放附近整数解，支持弹道偏好、固定力度和随角度变化的炮口。 |
| `shellshock/planning/policies.py` | 统一模式名称与旧别名，拆分模式族和高低弹道偏好，根据场景选择模式。 |
| `shellshock/planning/ranking.py` | 对已物理验证的候选排序，综合倍率、事件种类、误差、净空和分支偏好。 |
| `shellshock/planning/routes.py` | 按预算生成传送、反射和倍率事件路线，为简单回退路线保留覆盖。 |
| `shellshock/planning/unified.py` | 串联 A/B/C 搜索，公平分配整数重放预算，调用共享物理引擎验证并选择结果。 |

### `shellshock/planning/portal/` 与 `shellshock/planning/reflection/`

| 文件 | 职责 |
| --- | --- |
| `shellshock/planning/portal/__init__.py` | Python 包标记，无独立传送求解器实现。 |
| `shellshock/planning/reflection/__init__.py` | Python 包标记。 |
| `shellshock/planning/reflection/contact.py` | 在固定反射接触点和法向下解析求解连续飞行时间分支，处理退化约束。 |

### `shellshock/rendering/`

| 文件 | 职责 |
| --- | --- |
| `shellshock/rendering/__init__.py` | Python 包标记。 |
| `shellshock/rendering/overlay.py` | 把验证后的轨迹画到图像上，避免把传送瞬移连接为实体飞行线段。 |
| `shellshock/rendering/trajectory.py` | 对已验证飞行段采样，用空分隔点表示瞬时传送跳跃。 |

## 旧重复实现的整合

旧 `shellshock_detector` 与 `shellshock_detector_yolo` 中重复的检测结果类型、风力识别、模式和颜色几何，分别收敛到 `domain/detection.py`、`perception/wind.py`、`planning/policies.py` 和 `perception/color_geometry.py`。截图存储和数据准备归入 `capture/`、`datasets/`，Windows 操作归入 `adapters/`。数字模型训练和推理共享 `perception/digit_model.py` 的网络定义。

旧反射、传送和混合路线中的独立重放逻辑收敛到 `physics/engine.py`，事件差异放入 `physics/events/`。路线生成、连续种子、整数候选和结果排序由 `planning/` 的对应模块协作。普通弹道保留专用解析候选入口；共享验证与绘制使用同一套飞行段表达。

实时和离线入口通过 `application/scene.py` 共享图像解释，通过 `application/solver.py` 共享求解分派。`planning/portal/` 和 `planning/reflection/` 的包标记不表示存在两套独立完整求解器。详细旧模块映射见同目录 `module-migration.json`；本索引描述当前职责，不把迁移视为逐文件一对一改名。

补充迁移：`shellshock/interaction/click_policy.py` 统一客户区点击边界；屏幕外返回 OFFSCREEN 且不执行窗口激活或鼠标操作。

A 层连续关联新增：

| 文件 | 职责 |
|---|---|
| `shellshock/math2d/intervals.py` | 扩展实数区间、连续速度/位移极值、二次不等式时间区间。 |
| `shellshock/math2d/reachability.py` | 共享接触位置、连续时间与进出速度的双向收缩；同点反射、区域约束与保守区间划分。 |

`planning/layer_a.py` 现已调用上述关联传播，替代原先仅投影筛除和反射重置的逻辑；详见 [A 层说明](layer-a-correlations.md)。
