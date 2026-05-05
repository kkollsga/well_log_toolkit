"""
Shared synthetic data for the user-story scripts.

Why this module:

* The user's environment has an older ``logsuite`` editable-installed at a
  different path. The ``ensure_local_package_on_path()`` helper prepends
  this clone to ``sys.path`` so the story scripts exercise the in-progress
  code rather than the pre-M1 install.
* The synthetic dataset is shared so all four story scripts speak about
  the same wells / facies / palette and produce comparable output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_local_package_on_path() -> None:
    """Prepend the cloned repository root to ``sys.path``.

    Story scripts may be invoked directly (``python story_tests/...``).
    Without this shim, an older editable install of ``logsuite`` elsewhere
    on ``sys.path`` would be imported instead of the in-progress code.
    """
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


# Facies palette used by every story script.
FACIES_PALETTE: dict[int, str] = {
    0: "#999999",  # Tight
    1: "#3b82f6",  # Medium
    2: "#10b981",  # Clean
}

FACIES_LABELS: dict[int, str] = {0: "Tight", 1: "Medium", 2: "Clean"}


def synthesize_well(seed: int, n_points: int = 100) -> pd.DataFrame:
    """Three-facies poroperm dataset with realistic per-facies trends.

    Each facies has its own (a, b) for ``y = a * exp(b * x)`` and its own
    PHIE distribution. PERM also gets a lognormal scatter so fits don't
    land on top of each other.
    """
    rng = np.random.default_rng(seed)
    depth = np.arange(2500.0, 2500.0 + n_points * 0.5, 0.5)[:n_points]
    facies = rng.integers(0, 3, n_points).astype(float)

    # (a, b, mu_phi, sd_phi) per facies code
    bases = {
        0.0: (0.0005, 24.0, 0.07, 0.025),
        1.0: (0.005, 22.0, 0.13, 0.030),
        2.0: (0.05, 20.0, 0.20, 0.035),
    }
    phi = np.empty(n_points)
    perm = np.empty(n_points)
    for code, (a, b, mu_phi, sd_phi) in bases.items():
        mask = facies == code
        n = int(mask.sum())
        phi[mask] = np.clip(rng.normal(mu_phi, sd_phi, n), 0.04, 0.35)
        perm[mask] = a * np.exp(b * phi[mask]) * rng.lognormal(0, 0.25, n)

    # Petrophysical curves derived from PHIE so the facies palette tells a
    # consistent story across crossplot and log-track displays. Used by
    # Story 8; earlier stories ignore them.
    gr = np.clip(130 - 200 * phi + rng.normal(0, 8, n_points), 20, 160)
    rhob = np.clip(2.65 - 1.65 * phi + rng.normal(0, 0.03, n_points), 1.95, 2.70)
    nphi = np.clip(phi + rng.normal(0, 0.02, n_points), 0.0, 0.45)

    return pd.DataFrame(
        {
            "DEPT": depth,
            "PHIE": phi,
            "PERM": perm,
            "Facies": facies,
            "GR": gr,
            "RHOB": rhob,
            "NPHI": nphi,
        }
    )


def build_manager(well_names: list[str] | None = None, quiet: bool = True):
    """Return a ``WellDataManager`` populated with three synthetic wells.

    The manager has Facies set as discrete with labels and a user palette
    pre-configured (Story 2 prerequisite).

    By default informational broadcast prints (``✓ Set colors for …``,
    ``✓ Loaded N properties …``) are silenced via ``logsuite.set_quiet``
    so the story scripts produce clean output. Pass ``quiet=False`` to
    see the library's status messages.
    """
    ensure_local_package_on_path()
    from logsuite import WellDataManager, set_quiet  # imported after sys.path fix

    set_quiet(quiet)

    manager = WellDataManager()
    names = well_names or ["Well_A", "Well_B", "Well_C"]
    for i, name in enumerate(names):
        manager.load_properties(
            synthesize_well(seed=i),
            well_col=None,
            well_name=name,
            source_name="petrophysics",
            type_mappings={"Facies": "discrete"},
            label_mappings={"Facies": FACIES_LABELS},
        )

    # Story 2 prerequisite: define the facies palette once on the manager.
    manager.Facies.colors = FACIES_PALETTE

    return manager
