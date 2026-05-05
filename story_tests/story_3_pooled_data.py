"""
Story 3 — pooled raw-data extraction across wells.

A reservoir geologist wants a single long-form DataFrame of plug-level
values plus their grouping columns, pooled across every well in the
manager. Goal: replicate ``stats()``-style depth-weighted percentiles
externally and merge with metadata, without writing a per-well loop.

Run with: ``python story_tests/story_3_pooled_data.py``
"""

from __future__ import annotations

from synthetic_data import build_manager, ensure_local_package_on_path

ensure_local_package_on_path()

import numpy as np


def main() -> None:
    manager = build_manager()

    # The story expression: one chain, one DataFrame.
    raw = manager.PHIE.filter("Facies").data(weighted=True)

    print("Long-format pooled DataFrame:")
    print(raw.head(10).to_string(index=False))
    print(f"\nrows: {len(raw)}, columns: {list(raw.columns)}\n")

    # External depth-weighted statistics per facies.
    print("Pooled depth-weighted PHIE statistics by facies:")
    print(f"{'Facies':<10}{'wt-mean':>10}{'P10':>10}{'P50':>10}{'P90':>10}{'samples':>10}")
    for facies_label in ["Tight", "Medium", "Clean"]:
        sub = raw[raw["Facies"] == facies_label]
        wt = sub["Weight"].to_numpy()
        v = sub["PHIE"].to_numpy()
        wt_mean = float(np.average(v, weights=wt))
        p10, p50, p90 = (float(x) for x in np.percentile(v, [10, 50, 90]))
        print(f"{facies_label:<10}{wt_mean:>10.4f}{p10:>10.4f}{p50:>10.4f}{p90:>10.4f}{len(sub):>10}")

    # Cross-check with the built-in flat_columns DataFrame.
    print("\nstats(return_df=True, flat_columns=True) for comparison:")
    df_stats = manager.PHIE.filter("Facies").stats(
        return_df=True,
        flat_columns=True,
        methods=["mean", "percentile_10", "percentile_50", "percentile_90"],
    )
    print(df_stats.to_string(index=False))


if __name__ == "__main__":
    main()
