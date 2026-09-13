"""Tests for CLI exec-path resolution (Windows bare-name .cmd shims)."""

from __future__ import annotations

from skillopt.model.backend_config import _resolve_cli_path


def test_resolve_cli_path_uses_shutil_which(monkeypatch):
    """A name found on PATH resolves to its real executable."""
    monkeypatch.setattr("shutil.which", lambda v: f"/resolved/{v}")
    assert _resolve_cli_path("codex") == "/resolved/codex"


def test_resolve_cli_path_falls_back_to_original(monkeypatch):
    """A name not on PATH (or a bare name on a host without it) passes through."""
    monkeypatch.setattr("shutil.which", lambda v: None)
    assert _resolve_cli_path("codex") == "codex"


def test_resolve_cli_path_keeps_configured_absolute_path(monkeypatch):
    """An absolute configured path that cannot be resolved is preserved."""
    monkeypatch.setattr("shutil.which", lambda v: None)
    assert _resolve_cli_path("/opt/bin/codex") == "/opt/bin/codex"


def test_resolve_cli_path_not_called_with_empty(monkeypatch):
    """Empty input is not handed to shutil.which in a way that corrupts."""
    monkeypatch.setattr("shutil.which", lambda v: None)
    assert _resolve_cli_path("") == ""
