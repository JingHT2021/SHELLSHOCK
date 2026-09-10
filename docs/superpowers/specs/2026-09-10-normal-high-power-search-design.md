# Normal High Power Search Design

## Goal

Change ordinary-mode `normal_high` aiming from a fixed power of 100 to an
integer power search across 90 through 100, returning the verified shot with
the smallest replay miss distance.

## Scope

- Apply only to the ordinary `normal_high` solver path.
- Preserve `normal_low`, wormhole, reflection, and explicit `force_power`
  behavior.
- Keep angle and power outputs integer-valued because they map directly to
  game controls.

## Candidate Generation

For each integer power from 90 through 100, solve the ballistic equation at
that speed and retain the high-arc branch matching the established horizontal
direction. Around that branch's rounded theoretical angle, generate the same
bounded integer-angle neighborhood already used by the normal solver. Replay
every generated `(power, angle)` candidate against the real world geometry.

If `force_power` is provided, search only that requested power and preserve the
existing high-arc branch selection behavior.

## Selection

Discard candidates rejected by replay. From the remaining candidates, first
identify the smallest `miss_distance`. Keep the existing miss-tie tolerance so
small calibration noise does not destabilize the result. Within that reliable
set, rank `normal_high` candidates by:

1. smallest `miss_distance`;
2. highest `angle_degrees`;
3. greatest `clearance`;
4. highest `power` as a deterministic final tie-breaker.

This makes accuracy authoritative while retaining a high-arc preference among
practically equivalent shots.

## Diagnostics and Failure Handling

The existing candidate and verification counters remain authoritative and now
cover all powers in the 90--100 range. If no high-arc branch or no replay-valid
candidate exists across the range, return the existing `no-verified-shot`
unreachable result with diagnostics.

## Tests

Add focused solver tests that use deterministic ballistic branches and replay
results to prove:

- `normal_high` evaluates every integer power from 90 through 100;
- the selected result has the smallest verified miss distance rather than
  merely the highest angle or power;
- explicit `force_power` continues to restrict the search to one power.

Run the focused tests first, then the full project test suite.
