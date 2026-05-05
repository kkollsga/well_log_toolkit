"""
Story 4 — Petrel calculator-syntax regression equations.

A static modeller wants to copy a fitted ``y = a * exp(b * x)`` directly
into Petrel as ``pow(10, c1*x + c0)``. Acceptance: ``equation_format=``
on both the standalone artifact and the Crossplot legend.

Run with: ``python story_tests/story_4_petrel_equations.py``
"""

from __future__ import annotations

from synthetic_data import build_manager, ensure_local_package_on_path

ensure_local_package_on_path()

from logsuite import ExponentialRegression


def main() -> None:
    manager = build_manager()

    # Build a standalone fit on the Clean facies, pooled across wells.
    fit = (
        manager.filter(where={"Facies": "Clean"})
        .properties(["PHIE", "PERM"])
        .fit(
            ExponentialRegression(),
            name="Clean — pooled",
            equation_format="petrel",
            decimals=4,
        )
    )

    print(f"Fit R² = {fit.r_squared:.4f}\n")

    print("Same fit, three forms:")
    print(f"  natural : {fit.equation(format='natural')}")
    print(f"  log10   : {fit.equation(format='log10')}")
    print(f"  petrel  : {fit.equation(format='petrel')}")
    print(f"  legend  : {fit.label()}")

    # Decimals override
    print("\nDecimals control:")
    print(f"  petrel,2 : {fit.equation(format='petrel', decimals=2)}")
    print(f"  petrel,6 : {fit.equation(format='petrel', decimals=6)}")


if __name__ == "__main__":
    main()
