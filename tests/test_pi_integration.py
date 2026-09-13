"""Configuration and source-routing coverage for Pi integration."""
from __future__ import annotations

import argparse
import os
from unittest import mock

from skillopt_sleep.__main__ import _add_common, _cfg_from_args
from skillopt_sleep.config import load_config
from skillopt_sleep.harvest_sources import harvest_for_config
from skillopt_sleep.types import SessionDigest


def test_cli_maps_pi_backend_source_and_paths(monkeypatch):
    parser = argparse.ArgumentParser()
    _add_common(parser)
    args = parser.parse_args(
        [
            "--backend",
            "pi",
            "--source",
            "pi",
            "--pi-home",
            "~/.pi-test",
            "--pi-path",
            "~/bin/pi",
        ]
    )
    monkeypatch.setattr("skillopt_sleep.config._user_config_path", lambda: None)

    cfg = _cfg_from_args(args)

    assert cfg.get("backend") == "pi"
    assert cfg.get("transcript_source") == "pi"
    assert cfg.get("pi_home") == os.path.abspath(os.path.expanduser("~/.pi-test"))
    assert cfg.get("pi_path") == os.path.abspath(os.path.expanduser("~/bin/pi"))
    assert cfg.pi_sessions_dir == os.path.join(
        os.path.abspath(os.path.expanduser("~/.pi-test")), "agent", "sessions"
    )


def test_explicit_pi_source_routes_only_to_pi_harvester():
    cfg = load_config(
        transcript_source="pi",
        projects="invoked",
        invoked_project="/repo/project",
        pi_home="/tmp/pi-home",
    )
    expected = [SessionDigest(session_id="pi-session", project="/repo/project")]
    with (
        mock.patch("skillopt_sleep.harvest_sources.harvest_pi", return_value=expected) as pi,
        mock.patch("skillopt_sleep.harvest_sources.harvest_codex") as codex,
        mock.patch("skillopt_sleep.harvest_sources.harvest") as claude,
    ):
        actual = harvest_for_config(cfg, since_iso="2026-01-01T00:00:00Z", limit=3)

    assert actual == expected
    pi.assert_called_once_with(
        cfg.pi_sessions_dir,
        scope="invoked",
        invoked_project="/repo/project",
        since_iso="2026-01-01T00:00:00Z",
        limit=3,
    )
    codex.assert_not_called()
    claude.assert_not_called()


def test_auto_source_does_not_silently_add_pi_precedence():
    cfg = load_config(
        transcript_source="auto",
        projects="invoked",
        invoked_project="/repo/project",
    )
    expected = [SessionDigest(session_id="claude-session", project="/repo/project")]
    with (
        mock.patch("skillopt_sleep.harvest_sources.harvest_codex", return_value=[]),
        mock.patch("skillopt_sleep.harvest_sources.harvest", return_value=expected),
        mock.patch("skillopt_sleep.harvest_sources.harvest_pi") as pi,
    ):
        assert harvest_for_config(cfg) == expected

    pi.assert_not_called()
