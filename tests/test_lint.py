"""Lint checks that mirror CI — run locally to catch issues early."""

import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("black") is None, reason="black not installed")
def test_black_formatting():
    """Check that all source files are formatted with black."""
    result = subprocess.run(
        ["black", "--check", "logsuite/"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"black formatting check failed:\n{result.stderr}\n"
        "Run 'black logsuite/' to fix."
    )


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not installed")
def test_ruff_linting():
    """Check that ruff finds no errors."""
    result = subprocess.run(
        ["ruff", "check", "logsuite/"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"ruff check failed:\n{result.stdout}\n"
        "Run 'ruff check --fix logsuite/' to auto-fix."
    )
