"""
Table panel rendering for visualization consumers.

Attaches a DataFrame to the same figure as a parent axes (typically a
Crossplot scatter) so the combined output saves as a single SVG/PNG. Used
by ``Crossplot.add_table_panel(...)``; the helpers here are pure functions
operating on a matplotlib ``Figure`` + ``Axes`` so they can be reused by
other consumers (Composite, future SCAL deliverable templates) without
coupling to any one class.

Defaults aim at the SCAL/DG3 deliverable style:
* NaN → ``"N/A"``.
* MultiIndex column labels flatten via ``" | "``.
* MultiIndex row labels visually merge repeated outer levels (the outer
  cell is blanked when the same as the previous row, which mimics the
  merged-cell look without needing matplotlib cell merging).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _value_to_str(value: Any, formatter: Callable[[Any], Any] | str | None) -> str:
    """Render one cell value through an optional formatter, with NaN → ``N/A``."""
    if pd.isna(value):
        return "N/A"
    if formatter is None:
        return str(value)
    if callable(formatter):
        return str(formatter(value))
    # Treat strings as a format spec, e.g. ".4f"
    try:
        return format(value, formatter)
    except (TypeError, ValueError):
        return str(value)


def _format_cells(df: pd.DataFrame, formatters: Mapping[str, Any] | None) -> list[list[str]]:
    """Apply formatters column-by-column and produce a 2D list of strings."""
    formatters = formatters or {}
    rows: list[list[str]] = []
    for _, row in df.iterrows():
        rows.append([_value_to_str(row[col], formatters.get(col)) for col in df.columns])
    return rows


def _column_labels(df: pd.DataFrame) -> list[str]:
    if isinstance(df.columns, pd.MultiIndex):
        return [" | ".join(str(level) for level in col) for col in df.columns]
    return [str(c) for c in df.columns]


def _row_labels_with_visual_merge(df: pd.DataFrame) -> list[str] | None:
    """Build row labels; for MultiIndex, blank repeated outer-level cells.

    Returns ``None`` for a default integer index so the table isn't padded
    with row numbers.
    """
    if isinstance(df.index, pd.MultiIndex):
        labels: list[str] = []
        last_row: tuple = ()
        for row in df.index:
            row = tuple(row)
            parts: list[str] = []
            for i, level in enumerate(row):
                # Outer levels (everything except the innermost) are blanked
                # when they match the previous row at every position up to
                # and including this level.
                inner_index = len(row) - 1
                if (
                    i < inner_index
                    and i < len(last_row)
                    and tuple(row[: i + 1]) == tuple(last_row[: i + 1])
                ):
                    parts.append("")
                else:
                    parts.append(str(level))
            labels.append(" / ".join(p for p in parts if p))
            last_row = row
        return labels
    if isinstance(df.index, pd.RangeIndex):
        return None
    return [str(i) for i in df.index]


def render_table_panel(
    fig: Figure,
    parent_ax: Axes,
    df: pd.DataFrame,
    position: str = "bottom",
    title: str | None = None,
    formatters: Mapping[str, Any] | None = None,
    table_fraction: float = 0.30,
    fontsize: int = 9,
) -> Axes:
    """Attach a DataFrame as a rendered table panel inside ``fig``.

    Grows the figure dimension along the panel's axis (height for
    ``"bottom"``, width for ``"right"``) so the parent axes is not squished.
    Repositions ``parent_ax`` and adds a new axes for the table.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure that owns ``parent_ax``.
    parent_ax : matplotlib.axes.Axes
        The existing axes (e.g. the scatter on a Crossplot) the panel
        should attach to.
    df : pandas.DataFrame
        Rows × columns of values to render.
    position : {"bottom", "right"}, default "bottom"
        Where to place the panel relative to ``parent_ax``.
    title : str, optional
        Title rendered above the table.
    formatters : dict, optional
        Per-column formatter spec. Each value may be a callable
        ``f(value) -> str`` or a Python format spec like ``".4f"``.
        Columns without an entry use ``str(value)``.
    table_fraction : float, default 0.30
        Fraction of the figure dimension reserved for the panel.
        Must be strictly between 0 and 1.
    fontsize : int, default 9
        Cell font size; the title is rendered one point larger.

    Returns
    -------
    matplotlib.axes.Axes
        The new axes containing the table — useful for further tweaks
        (e.g. setting per-cell colors).
    """
    if position not in {"bottom", "right"}:
        raise ValueError(f"position must be 'bottom' or 'right', got {position!r}")
    if not 0 < table_fraction < 1:
        raise ValueError(f"table_fraction must be in (0, 1), got {table_fraction}")

    fig_w, fig_h = fig.get_size_inches()
    parent_bbox = parent_ax.get_position()

    # Figure-fraction spacing knobs. ``gap`` is the explicit visual breathing
    # room between the parent axes (scatter, including its tick labels) and
    # the table panel. Without it, tall tables — even drawn at
    # ``loc="upper center"`` — collide with the chart's bottom labels.
    outer_margin = 0.04
    gap = 0.10

    if position == "bottom":
        new_h = fig_h / (1 - table_fraction)
        fig.set_size_inches(fig_w, new_h, forward=True)

        panel_bottom = outer_margin
        panel_height = table_fraction - outer_margin
        parent_y0 = panel_bottom + panel_height + gap
        parent_height = 1 - parent_y0 - outer_margin
        parent_ax.set_position([parent_bbox.x0, parent_y0, parent_bbox.width, parent_height])
        table_ax = fig.add_axes([parent_bbox.x0, panel_bottom, parent_bbox.width, panel_height])
    else:  # "right"
        new_w = fig_w / (1 - table_fraction)
        fig.set_size_inches(new_w, fig_h, forward=True)

        # Reserve the rightmost ``table_fraction`` of the new figure width
        # for the panel, keeping the original left margin and a gap before
        # the panel. The parent shrinks in figure-fraction so the absolute
        # inch width is roughly preserved (figure also grew).
        new_parent_width = (1 - table_fraction) - parent_bbox.x0 - gap
        parent_ax.set_position(
            [parent_bbox.x0, parent_bbox.y0, new_parent_width, parent_bbox.height]
        )
        panel_left = parent_bbox.x0 + new_parent_width + gap
        panel_width = (1 - panel_left) - outer_margin
        table_ax = fig.add_axes([panel_left, parent_bbox.y0, panel_width, parent_bbox.height])

    table_ax.axis("off")

    cell_text = _format_cells(df, formatters)
    col_labels = _column_labels(df)
    row_labels = _row_labels_with_visual_merge(df)

    if not cell_text:
        if title:
            table_ax.set_title(title, fontsize=fontsize + 1, fontweight="bold", pad=10)
        return table_ax

    table = table_ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        rowLabels=row_labels,
        # Anchor the table at the top of its panel so a tall table grows
        # downward into the bottom margin rather than upward into the chart.
        loc="upper center",
        cellLoc="right",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    # Less aggressive row scaling than 1.5×; tall rows were the main
    # contributor to the table extending beyond its panel on dense DataFrames.
    table.scale(1.0, 1.3)

    if title:
        # pad=4 keeps the title close to the panel's top edge so it lives
        # inside the gap between the chart and the table rather than
        # creeping into the chart's tick-label area.
        table_ax.set_title(title, fontsize=fontsize + 1, fontweight="bold", pad=4)

    return table_ax
