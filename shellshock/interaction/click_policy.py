"""Safe client-area click policy shared by every shot mode."""

from __future__ import annotations

from collections.abc import Callable


Point = tuple[int, int]


def click_client_point(
    client_point: Point,
    client_origin: Point,
    client_size: Point,
    *,
    activate: Callable[[], None],
    click: Callable[[Point], None],
) -> str:
    """Click an in-client point, or quietly report that it is offscreen."""
    x, y = client_point
    width, height = client_size
    if not (0 <= x < width and 0 <= y < height):
        return "OFFSCREEN"
    activate()
    click((client_origin[0] + x, client_origin[1] + y))
    return "SUCCESS"
