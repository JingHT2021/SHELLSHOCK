# Layer A Event Filter Design

## Goal

Replace reflection-only Layer A ranking with a conservative, reusable event-path
filter. It removes only paths whose next waypoint is unreachable on an axis by
both every permitted velocity direction and acceleration.

## Architecture

`coarse_path_filter.py` owns generic waypoint/event traversal. A portal event
teleports position to its paired exit without changing possible velocity
directions. A reflection adapter contributes the midpoint and local reflection
transform (`tangent` unchanged, `normal` reversed). Reflection and wormhole
solvers provide routes to this filter; the filter contains no replay, score,
ballistic solve, collision time, or virtual-height heuristic.

## Behavior

- A normal segment is retained when each axis can move toward its target under
  at least one possible velocity sign or its acceleration sign.
- Portal sequences are checked in actual entry/exit order, including
  portal-to-portal routes.
- Reflection routes check pre-reflection events, apply the local velocity
  reflection transform, then check post-reflection events.
- Candidates contain route identity, events, and rejection diagnostics only.
  Layer B owns proxy scoring and ranking.
