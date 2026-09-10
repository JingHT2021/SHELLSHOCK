# Normal High Local Search Design

## Goal

Change ordinary-mode `normal_high` from one fixed 100-power shot to a small
integer neighborhood search that returns the replay-verified shot with the
smallest miss distance.

## Search Space

Use the high-arc solution calculated at power 100 as the single theoretical
angle center. Search:

- power: every integer from 94 through 100 (100 minus a deviation of 6);
- angle: every integer from the rounded theoretical angle minus 3 degrees
  through the rounded theoretical angle plus 3 degrees, clamped to 0--90.

Do not recompute a new theoretical angle for each power. Replay all 49
candidates when the full neighborhood is inside the legal angle range.

## Selection

Discard replay-invalid candidates. Rank the rest by smallest `miss_distance`,
then use higher angle, greater clearance, and higher power as deterministic
tie-breakers. Preserve explicit `force_power` by restricting the same angle
neighborhood to the requested power.

## Scope and Diagnostics

Only `normal_high` changes. `normal_low`, wormhole, and reflection behavior
remain unchanged. High-mode diagnostics report the power-100 high-arc theory
angle and theory power 100, plus the existing candidate and verification
counters.

## Tests

Tests verify the 94--100 power range, the fixed theoretical-angle neighborhood,
49 full-range candidates, accuracy-first selection, deterministic tie-breakers,
and explicit forced-power restriction.
