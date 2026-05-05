"""
Story 7 — Plot-deliverable composition.

A geologist wants the standard SCAL/DG3 deliverable shape: a poroperm
crossplot with per-facies regression lines and a summary statistics table
attached to the same figure, exported as a single SVG suitable for paste
into PowerPoint or Word — without writing matplotlib glue.

Run with: ``python story_tests/story_7_scal_deliverable.py``
"""

from __future__ import annotations

from pathlib import Path

from synthetic_data import build_manager, ensure_local_package_on_path

ensure_local_package_on_path()

import matplotlib

matplotlib.use("Agg")  # headless

from logsuite import Crossplot

OUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    manager = build_manager()

    # Build the per-facies summary table (Story 3 / Story 6 path).
    stats = manager.PHIE.filter("Facies").stats(
        return_df=True,
        flat_columns=True,
        methods=["mean", "percentile_10", "percentile_50", "percentile_90"],
    )
    stats = stats.rename(
        columns={"p10": "P10", "p50": "P50", "p90": "P90"},
    )

    # Crossplot with per-facies regressions in Petrel form, legend in
    # the upper-left so the table panel can sit underneath.
    xplot = Crossplot(
        manager,
        x="PHIE",
        y="PERM",
        color="Facies",
        y_log=True,
        title="Synthetic poroperm — DG3 deliverable shape",
        figsize=(9, 6),
        equation_format="petrel",
        decimals=3,
    )
    xplot.plot()
    xplot.add_regression_per(
        "Facies",
        "exponential",
        min_samples=10,
        legend_loc="upper left",
    )

    # Attach the statistics table as a panel on the bottom.
    xplot.add_table_panel(
        stats,
        position="bottom",
        title="Per-facies PHIE summary",
        formatters={
            "mean": ".4f",
            "P10": ".4f",
            "P50": ".4f",
            "P90": ".4f",
        },
    )

    out = OUT_DIR / "output_story_7.svg"
    xplot.save(str(out))
    print(f"Saved combined deliverable to {out}")
    print(f"Stats table:\n{stats.to_string(index=False)}")


if __name__ == "__main__":
    main()
