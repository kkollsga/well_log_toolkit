"""
Story 6 — One-call pooled extraction of raw plug values across wells.

A reservoir geologist wants pooled raw values + grouping columns from a
single manager-level call. Three real use cases motivate this story:

1. Pooled percentile table (mean / P10 / P50 / P90 per facies).
2. Diagnostic subset — the "suspect" Clean-facies plugs with unusually
   low PHIE — for QC inspection.
3. Cross-check with the library's own ``stats(return_df=True,
   flat_columns=True)`` to confirm the in-Python numbers agree.

Today (pre-M1) each of these required its own ``for well in
manager.wells: ...`` loop. With ``manager.PHIE.filter("Facies").data()``
they're each one expression.

Run with: ``python story_tests/story_6_pooled_extraction.py``
"""

from __future__ import annotations

from synthetic_data import build_manager, ensure_local_package_on_path

ensure_local_package_on_path()

import numpy as np


def main() -> None:
    manager = build_manager()

    # ---------- Use case 1: pooled raw data with depth weights ----------
    raw = manager.PHIE.filter("Facies").data(weighted=True, warn_missing=False)
    print("Pooled long-form DataFrame:")
    print(f"  rows={len(raw)}, columns={list(raw.columns)}")
    print(raw.head().to_string(index=False))
    print()

    print("Pooled depth-weighted PHIE percentiles by facies:")
    print(
        f"  {'Facies':<10}{'samples':>10}{'wt_mean':>10}"
        f"{'P10':>10}{'P50':>10}{'P90':>10}"
    )
    for facies_label in ["Tight", "Medium", "Clean"]:
        sub = raw[raw["Facies"] == facies_label]
        weights = sub["Weight"].to_numpy()
        values = sub["PHIE"].to_numpy()
        wt_mean = float(np.average(values, weights=weights))
        p10, p50, p90 = (float(x) for x in np.percentile(values, [10, 50, 90]))
        print(
            f"  {facies_label:<10}{len(sub):>10}{wt_mean:>10.4f}"
            f"{p10:>10.4f}{p50:>10.4f}{p90:>10.4f}"
        )
    print()

    # ---------- Use case 2: diagnostic subset for QC ----------
    suspects = raw[(raw["Facies"] == "Clean") & (raw["PHIE"] < 0.10)]
    print(f"Suspect Clean-facies plugs (PHIE < 0.10): {len(suspects)} rows")
    if len(suspects):
        print(suspects[["well", "DEPT", "PHIE"]].to_string(index=False))
    else:
        print("  (none — synthetic data well-behaved at this seed)")
    print()

    # ---------- Use case 3: cross-check with the library aggregation ----------
    print("Library-side aggregation via stats(return_df=True, flat_columns=True):")
    df_stats = manager.PHIE.filter("Facies").stats(
        return_df=True,
        flat_columns=True,
        methods=["mean", "percentile_10", "percentile_50", "percentile_90"],
    )
    print(df_stats.to_string(index=False))


if __name__ == "__main__":
    main()
