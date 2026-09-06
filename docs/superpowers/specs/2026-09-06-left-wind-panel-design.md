# Left-wind HUD panel detection

## Goal

Recognize the ShellShock Live wind panel when its arrow appears to the left of
the cloud, without changing the existing right-arrow behavior.

## Design

Panel discovery will pair the two nearby, top-centre HUD components regardless
of their left-to-right order. It will skip candidate pairs narrower than a
scaled wind-panel width instead of returning early, so unrelated HUD digits do
not prevent discovery of the real panel. The returned rectangle spans both
components, so the existing direction and OCR routines continue to operate
unchanged.

## Missing panels

When no valid panel exists, the application continues to use 0 wind and does
not block manual aiming, matching the current intended fallback.

## Verification

Add a synthetic left-arrow panel test that verifies panel discovery and a
left direction result. Keep the missing-panel fallback test unchanged.
