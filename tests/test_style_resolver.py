"""Tests for visualization/style.py — discrete palette resolution."""

import pytest

from logsuite.visualization.style import (
    DEFAULT_DISCRETE_PALETTE,
    resolve_discrete_palette,
)


class TestResolveDiscretePalette:
    def test_user_colors_override_defaults(self):
        user = {1: "#aaaaaa", 2: "#bbbbbb"}
        out = resolve_discrete_palette(user, [1, 2, 3])
        assert out[1] == "#aaaaaa"
        assert out[2] == "#bbbbbb"
        # 3 falls back to first default
        assert out[3] == DEFAULT_DISCRETE_PALETTE[0]

    def test_no_user_colors_uses_defaults_in_order(self):
        out = resolve_discrete_palette(None, [10, 20, 30])
        assert out[10] == DEFAULT_DISCRETE_PALETTE[0]
        assert out[20] == DEFAULT_DISCRETE_PALETTE[1]
        assert out[30] == DEFAULT_DISCRETE_PALETTE[2]

    def test_empty_user_colors_treated_as_none(self):
        out = resolve_discrete_palette({}, [1, 2])
        assert out[1] == DEFAULT_DISCRETE_PALETTE[0]
        assert out[2] == DEFAULT_DISCRETE_PALETTE[1]

    def test_float_categories_match_int_keys(self):
        # Categories from numpy arrays are floats; user.colors keyed on int codes
        user = {1: "#3b82f6", 2: "#ef4444"}
        out = resolve_discrete_palette(user, [1.0, 2.0, 3.0])
        assert out[1.0] == "#3b82f6"
        assert out[2.0] == "#ef4444"
        assert out[3.0] == DEFAULT_DISCRETE_PALETTE[0]

    def test_fallback_index_only_increments_for_unknown_categories(self):
        user = {1: "X", 5: "Y"}
        out = resolve_discrete_palette(user, [1, 2, 5, 7])
        assert out[1] == "X"
        assert out[2] == DEFAULT_DISCRETE_PALETTE[0]
        assert out[5] == "Y"
        assert out[7] == DEFAULT_DISCRETE_PALETTE[1]

    def test_custom_default_palette(self):
        out = resolve_discrete_palette(None, [1, 2], default_palette=["#111", "#222"])
        assert out[1] == "#111"
        assert out[2] == "#222"

    def test_palette_cycles_when_exhausted(self):
        out = resolve_discrete_palette(
            None, list(range(15)), default_palette=["a", "b", "c"]
        )
        # 15 categories cycle through 3-color palette
        assert out[0] == "a"
        assert out[3] == "a"
        assert out[6] == "a"

    def test_nan_categories_skipped(self):
        out = resolve_discrete_palette(None, [1.0, float("nan"), 2.0])
        assert 1.0 in out
        assert 2.0 in out
        # NaN is skipped (would be a confusing key anyway)
        assert all(not (isinstance(k, float) and k != k) for k in out)
