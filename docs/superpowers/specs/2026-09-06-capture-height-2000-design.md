# Capture height: 2000 pixels

## Goal

Extend the supported ShellShock Live playfield from the current top 1800
physical pixels to the top 2000 physical pixels, including capture, manual
aiming, and tank detection.

## Design

Use one shared game-capture-height constant with the value `2000`. The
capture routine crops screenshots to that height. Manual self/target
coordinates at `y < 2000` remain valid, and values at or below the boundary
are rejected. Tank-detection search limits use the same 2000-pixel playfield
bottom.

## Documentation and verification

Update user-facing messages and the README to state 2000 pixels. Tests will
verify that capture, aiming validation, and tank detection all use the new
boundary, while preserving rejection at `y >= 2000`.

## Scope

The height remains a fixed application value; no new command-line option is
added. Existing resolution validation and horizontal behavior are unchanged.
