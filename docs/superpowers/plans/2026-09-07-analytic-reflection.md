# 解析求解与反射重构执行清单

依据用户提供的《detect_shellshock_yolo 整体优化与反射模式重构计划书》实施。

目标：按 normal/wormhole/reflection 分发连续候选，再做整数精确回放。
架构：入口保留截图、识别、点击；共享 collision 几何；独立 reflection_solver / reflection_replay；保留旧入口兼容。
技术：Python、NumPy、SciPy、pytest。

- [x] 新增碰撞、closest approach、clearance、缩放参数，先测试再实现。
- [x] 虫洞 planned sequence 与双半径回放；专用解析入口和误差排序。
- [x] 线/圆连续反射解族、Pmin、固定力度根、low/high 同板选择、整数回放。
- [x] 模式规范化、快捷键、普通求解与诊断日志。
- [x] 计划场景回归、全套测试、性能测量、独立审查与使用说明。

验证和限制见 `docs/solver-refactor-validation.md`。high 的下界采用 ceil(Pmin)，允许 low 安全余量超出100时仍返回 high-only 解。游戏命中率尚未实测。

解释：计划 Test 15 的文字相互矛盾；仅最近到 0.95r 而未进入 0.90r 必须拒绝。反射模式不接受未经计划的虫洞组合路径。根据用户后续明确要求，已增加一次反射前后最多两次计划传送的连续求解与真实回放，同时保留纯反射候选。旧的通用事件回放 API 保留。
