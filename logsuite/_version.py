"""Version detection for logsuite."""


def _get_version():
    """Get version from installed package metadata or pyproject.toml."""
    try:
        from importlib.metadata import version

        return version("logsuite")
    except Exception:
        try:
            from pathlib import Path
            import re

            pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
            if pyproject_path.exists():
                content = pyproject_path.read_text()
                match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
                if match:
                    return match.group(1)
        except Exception:
            pass
    return "unknown"
