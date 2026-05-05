"""
Zone × Facies poroperm workflow.

End-to-end example: three synthetic wells with a Zone (3 values) × Facies
(3 values) hierarchy; compute sums/averages of porosity per Zone and
per Facies; extract a por-perm transform for each main zone; render the
combined deliverable — a Zone-coloured poroperm crossplot with per-zone
regressions in Petrel calculator syntax, plus a Zone × Facies summary
table — into a single SVG.

Run:  python examples/zone_facies_poroperm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the cloned repo to sys.path so this script exercises the in-tree
# code rather than any older editable install elsewhere on the path.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np
import pandas as pd

from logsuite import Crossplot, ExponentialRegression, WellDataManager, set_quiet

OUT_DIR = Path(__file__).resolve().parent

ZONE_LABELS = {0: "Sand 1", 1: "Sand 2", 2: "Sand 3"}
FACIES_LABELS = {0: "Tight", 1: "Medium", 2: "Clean"}

# Per-zone poroperm trend parameters: y = a * exp(b * x).
ZONE_POROPERM = {
    0: (0.0008, 22.0),
    1: (0.005, 19.0),
    2: (0.05, 16.0),
}

# Per-facies PHIE distribution (mean, std) within any zone.
FACIES_PHI = {
    0: (0.07, 0.020),
    1: (0.13, 0.030),
    2: (0.20, 0.030),
}


def synthesize_zoned_well(seed: int, n_per_zone: int = 80) -> pd.DataFrame:
    """Three stacked zones, each with a Tight/Medium/Clean facies mix."""
    rng = np.random.default_rng(seed)
    rows = []
    depth_top = 2400.0
    for zone_code in [0, 1, 2]:
        depths = np.arange(depth_top, depth_top + n_per_zone * 0.5, 0.5)[:n_per_zone]
        facies_codes = rng.integers(0, 3, n_per_zone).astype(float)
        a, b = ZONE_POROPERM[zone_code]
        phi = np.empty(n_per_zone)
        perm = np.empty(n_per_zone)
        for f_code in [0.0, 1.0, 2.0]:
            mask = facies_codes == f_code
            n = int(mask.sum())
            mu_phi, sd_phi = FACIES_PHI[int(f_code)]
            phi[mask] = np.clip(rng.normal(mu_phi, sd_phi, n), 0.04, 0.35)
            perm[mask] = a * np.exp(b * phi[mask]) * rng.lognormal(0, 0.25, n)
        rows.append(
            pd.DataFrame(
                {
                    "DEPT": depths,
                    "PHIE": phi,
                    "PERM": perm,
                    "Zone": float(zone_code),
                    "Facies": facies_codes,
                }
            )
        )
        depth_top += n_per_zone * 0.5 + 5.0
    return pd.concat(rows, ignore_index=True)


# Which stats to surface in the rendered table panel. The full set of
# valid keys is:  ["samples", "mean", "std", "min", "max",
#                  "P10", "P50", "P90"].
# Trim or reorder to taste — every entry must also appear in
# TABLE_FORMATTERS below.
TABLE_FIELDS: dict = {
    "samples": lambda v: f"{int(v)}",
    "mean": ".4f",
    "std": ".4f",
    "min": ".4f",
    "max": ".4f",
    "p10": ".4f",
    "p50": ".4f",
    "p90": ".4f",
}


def main() -> None:
    set_quiet(True)

    # ---- Manager with three wells -------------------------------------------
    manager = WellDataManager()
    for i, name in enumerate(["Well_A", "Well_B", "Well_C"]):
        manager.load_properties(
            synthesize_zoned_well(seed=i),
            well_col=None,
            well_name=name,
            source_name="petrophysics",
            type_mappings={"Zone": "discrete", "Facies": "discrete"},
            label_mappings={"Zone": ZONE_LABELS, "Facies": FACIES_LABELS},
        )

    # Project palettes — set once on the manager, read everywhere.
    manager.Zone.colors = {0: "#fb923c", 1: "#3b82f6", 2: "#10b981"}
    manager.Facies.colors = {0: "#999999", 1: "#a16207", 2: "#0d9488"}

    # ---- Per-well Zone × Facies summary (printout for visibility) ----------
    per_well = manager.PHIE.filter("Zone").filter("Facies").stats(
        return_df=True,
        flat_columns=True,
        methods=["mean", "std", "min", "max",
                 "percentile_10", "percentile_50", "percentile_90"],
    )
    print("Per-well Zone × Facies PHIE summary:")
    print(per_well.to_string(index=False))
    print()

    # ---- Pooled cross-well summary -----------------------------------------
    # ``pool=True`` aggregates across wells from the long-form data so std,
    # min, max, and percentiles are statistically correct — not averages of
    # per-well point estimates.
    pooled = manager.PHIE.filter("Zone").filter("Facies").stats(
        return_df=True, pool=True, flat_columns=True,
        methods=list(TABLE_FIELDS.keys()),
    )
    print("Pooled cross-well PHIE summary:")
    print(pooled.to_string(index=False))
    print()

    # ---- Per-zone por-perm transforms (one fit per Zone) -------------------
    fits = manager.properties(["PHIE", "PERM"]).fit_per(
        "Zone", ExponentialRegression(), equation_format="petrel", decimals=4
    )
    print("Por-perm transforms per main zone (Petrel calculator syntax):")
    print(f"  {'Zone':<10}{'R²':>8}  equation")
    for label in ZONE_LABELS.values():
        fit = fits.get(label)
        if fit is not None:
            print(f"  {label:<10}{fit.r_squared:>8.3f}  {fit.equation()}")
    print()

    # ---- Combined deliverable: crossplot + summary table -------------------
    xplot = Crossplot(
        manager,
        x="PHIE",
        y="PERM",
        color="Zone",
        y_log=True,
        title="Zone-segregated poroperm transforms — synthetic 3-well field",
        figsize=(11, 7),
        equation_format="petrel",
        decimals=3,
    )
    xplot.add_regression_per(
        "Zone", "exponential", min_samples=20, legend_loc="upper left"
    )
    xplot.add_table_panel(
        pooled,
        position="bottom",
        title="Zone × Facies — pooled cross-well PHIE",
        formatters=TABLE_FIELDS,
    )

    out = OUT_DIR / "zone_facies_poroperm.svg"
    xplot.save(str(out))
    print(f"Saved deliverable to {out}")


if __name__ == "__main__":
    main()
