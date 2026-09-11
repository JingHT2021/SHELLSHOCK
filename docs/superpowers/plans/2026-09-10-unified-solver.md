# Unified ShellShock implementation plan

> Execute task-by-task in the isolated worktree, with focused regression tests and independent specification/quality review.

Goal: merge the two packages, preserve current detection fixes, and share event physics and analytic candidate generation across live and saved-image entry points.

Architecture: one shellshock package; pure math2d primitives, perception and annotation adapters, physics event engine, and planning A/B/C. Existing entry scripts remain launchable. Existing surface reflection convention is preserved. Blackhole 7 remains forbidden unless the user changes this explicitly. Portals and rewards trigger at 80% radius; intended circles reflect at physical radius, intended lines use their central 90%; unintended geometry gets an absolute 10 image-pixel safety margin.

Tech stack: existing Python virtualenv, NumPy, SciPy, OpenCV, pytest; no model retraining.

- [x] Snapshot working edits into .worktrees/unified-solver; run existing pytest baseline.
- [x] Add import/path regression test, confirm it fails before relocating. Map every old module into shellshock/domain, perception, annotations, datasets, math2d, physics, planning, adapters, rendering, interaction, config. Rewrite imports through the mapping and centralize data root discovery. Remove only old source files verified present in the worktree; retain root launchers.
- [x] Add tests for high-arc candidates including power 90 and true minimum miss, then update normal solver to recompute analytic high arc at each power 90..100 and replay local integer angles.
- [x] Preserve circle-fit changes and add perception tests: synthetic standard circle with inaccurate detection box; manual center wins, valid pose center is not silently replaced by box center. Reject weak circle fits and retain diagnostics and fallback.
- [x] Add pure-math tests for three-point positive-time solutions, rotated acceleration and impossible/degenerate cases. Extract shared frames, roots, position/velocity and continuous reachability; remove velocity-quadrant rejection from active A.
- [x] Add engine tests for portal offset/velocity continuity, line/circle bounce, reward stacking, forbidden blackholes, +10 pixel unexpected geometry and central 90% line contact. Implement one event replay returning trajectory segments. Circle avoidance uses cubic distance extrema and endpoints; anticipated events may use exact intersections.
- [x] Add planning tests for reward+portal+reflection priority, portal mode allowing reflection, no full power sweep in waypoint search, and higher power preference for comparable valid circle reflections. Implement bounded analytic B candidates and local integer C; use existing fixed-contact analytic reflection formulas. Preserve explicit no-solution vs budget-exhausted diagnostics.
- [x] Route live/replay through common solve and scene services. Render verified segments, route manual previews through engine, share mode commands and annotation storage. Centralize weights, screenshot, annotation and replay paths; reuse original data directory explicitly in this worktree.
- [x] Run all pytest, import/CLI smoke checks, synthetic integration scenes and selected saved-image annotation replays. Review specification then code quality independently, fix findings, update README and migration notes. Leave original checkout unchanged.

Primary commands (run in worktree):
```powershell
C:\Users\jingh\Desktop\game\shellshock\.venv\Scripts\python.exe -m pytest -q
C:\Users\jingh\Desktop\game\shellshock\.venv\Scripts\python.exe replay_shellshock.py --help
C:\Users\jingh\Desktop\game\shellshock\.venv\Scripts\python.exe detect_shellshock_yolo.py --help
```

New tests live in tests/test_package_layout.py, test_continuous_math.py, test_event_engine.py, test_unified_planning.py, test_perception_refinement.py and the existing test_normal_solver.py. Each behavioral change starts with a failing regression. Search budgets cap route/seed/verification counts and report truncation; they are not proofs of impossibility.

Completion evidence (2026-09-11): 64 pytest cases passed; all package imports and three entrypoint/launcher help checks passed, including outside-CWD invocation. Three images × three modes replayed; final normal replay confirms 93/83 and 0.1873 px predicted miss. Independent engine/planning review passed 21 focused tests. See refactor-notes.md for conservative A / finite B / bounded C limitations; no claim of complete continuous velocity-set propagation or actual game firing validation. Worktree remains separate and original checkout source unchanged.
