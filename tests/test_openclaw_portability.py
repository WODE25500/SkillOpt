"""Tests that the OpenClaw plugin paths are portable (not hardcoded)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / "plugins" / "openclaw"


def _load_slash_sleep():
    spec = importlib.util.spec_from_file_location(
        "slash_sleep", _PLUGIN_DIR / "slash_sleep.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_skill_dir_uses_home_expansion():
    """SKILL_DIR must be derived from the home directory, not an absolute user."""
    mod = _load_slash_sleep()
    expected = Path(os.path.expanduser("~/.openclaw/workspace/skills/skillopt-sleep"))
    assert str(mod.SKILL_DIR) == str(expected)


def test_skill_dir_not_hardcoded_to_ethanclaw():
    """No /home/ethanclaw residue should remain in the plugin path."""
    mod = _load_slash_sleep()
    assert "ethanclaw" not in str(mod.SKILL_DIR)
