"""
Artifact protocol — composable result objects that consumers can render.

Artifacts are self-contained results (regression fits, trends, statistical
bundles, ...) that know how to render themselves into different visualization
consumers. Consumers (Crossplot, Table, Composite, WellView) accept artifacts
via a single ``.add(artifact)`` method; each artifact implements only the
``_render_in_<consumer>(...)`` methods it supports. Unsupported consumers
raise a clean :class:`TypeError`.

This module belongs to the ``core`` layer and has no runtime dependency on
visualization. Render-method type hints reference matplotlib via
``TYPE_CHECKING`` only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class Artifact:
    """Base class for objects added to visualization consumers via ``.add()``.

    Subclasses override only the rendering methods relevant to the consumers
    they support. Unsupported consumers fall through to the base
    implementation, which raises :class:`TypeError` with a clear message.

    Examples
    --------
    >>> class MyArtifact(Artifact):
    ...     def _render_in_crossplot(self, ax):
    ...         ax.axhline(self.y, color="red")
    ...
    >>> art = MyArtifact()
    >>> crossplot.add(art)         # works
    >>> wellview.add(art)          # raises TypeError
    """

    def _render_in_crossplot(self, ax: Axes, **kwargs: Any) -> None:
        raise TypeError(
            f"{type(self).__name__} cannot be rendered in a Crossplot. "
            f"It does not implement _render_in_crossplot."
        )

    def _render_in_table(self, table: Any, **kwargs: Any) -> Any:
        raise TypeError(
            f"{type(self).__name__} cannot be rendered in a Table. "
            f"It does not implement _render_in_table."
        )

    def _render_in_wellview(self, view: Any, **kwargs: Any) -> None:
        raise TypeError(
            f"{type(self).__name__} cannot be rendered in a WellView. "
            f"It does not implement _render_in_wellview."
        )
