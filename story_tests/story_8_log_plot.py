"""
Story 8 — Single-well log-plot composition with WellView.

A geologist wants the standard well-log display for one well: a depth axis,
a GR track, a density/neutron overlay, a porosity track with reservoir
fill, and a facies track that uses the same palette set once at the
manager level. Acceptance is that ``WellView(well, template=...)``
produces the deliverable image with no additional matplotlib code, and
honours ``Property.colors`` and ``Property.labels`` for the discrete
facies fill.

Run with: ``python story_tests/story_8_log_plot.py``
"""

from __future__ import annotations

from pathlib import Path

from synthetic_data import build_manager, ensure_local_package_on_path

ensure_local_package_on_path()

import matplotlib

matplotlib.use("Agg")  # headless

from logsuite import Template, WellView

OUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    manager = build_manager()
    well = manager.well_Well_A  # Manager-substrate access

    # Build a five-track template: depth | GR | RHOB-NPHI | PHIE | Facies.
    template = Template("story_8")
    template.add_track(track_type="depth", width=0.4)
    template.add_track(
        track_type="continuous",
        title="GR",
        logs=[{"name": "GR", "x_range": [0, 150], "color": "darkgreen"}],
        width=1.0,
    )
    template.add_track(
        track_type="continuous",
        title="RHOB / NPHI",
        logs=[
            {"name": "RHOB", "x_range": [1.95, 2.70], "color": "red"},
            # NPHI is plotted right-to-left by convention so the standard
            # crossover signature appears in cleaner zones.
            {"name": "NPHI", "x_range": [0.45, -0.05], "color": "blue"},
        ],
        width=1.2,
    )
    template.add_track(
        track_type="continuous",
        title="PHIE",
        logs=[{"name": "PHIE", "x_range": [0.0, 0.35], "color": "navy"}],
        fill={
            "left": {"curve": "PHIE"},
            "right": {"value": 0},
            "color": "lightblue",
            "alpha": 0.5,
        },
        width=1.0,
    )
    template.add_track(
        track_type="discrete",
        title="Facies",
        logs=[{"name": "Facies"}],
        width=0.5,
    )

    # WellView reads the same Property.colors and Property.labels we set
    # for Stories 1/2/7 — same palette, different rendering surface.
    view = WellView(
        well,
        depth_range=(2510, 2545),
        template=template,
        figsize=(10, 8),
    )

    out = OUT_DIR / "output_story_8.png"
    view.save(str(out))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
