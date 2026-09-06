# Outside-client aim report

## Goal

Show the computed integer angle and power when the aim-disc click point lies
outside the game client, while preserving the no-click safety boundary.

## Design

The aiming operation will return a result that distinguishes a successful
calculation from an executed click. If the click point is outside the client,
the result contains the captured paths, ballistic solution, and client click
point, but no screen click point. The command-line handler formats the
solution and prints a clear no-click warning.

## Safety and verification

The injected click callback must not run for an outside-client point. Tests
will assert both the returned angle/power solution and the absent click.
