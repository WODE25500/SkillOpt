"""Tests for the SkillOpt WebUI security posture (bind default + public warning).

The WebUI is gradio-coupled, so we inject a minimal fake ``gradio`` module and
mock ``build_ui``/``launch`` to exercise ``main()``'s argparse + host-check
logic without the heavy ``webui`` extra.
"""

from __future__ import annotations

import sys
import types
import unittest.mock as mock

import pytest


@pytest.fixture
def webui(monkeypatch):
    fake_gradio = types.ModuleType("gradio")
    fake_gradio.themes = types.SimpleNamespace(Soft=lambda **kw: mock.MagicMock())
    monkeypatch.setitem(sys.modules, "gradio", fake_gradio)
    import skillopt_webui.app as app

    return app


def test_main_defaults_host_to_localhost(webui, monkeypatch):
    """The server must not be publicly bound by default."""
    webui_mod = webui
    launcher = mock.MagicMock()
    app_mock = mock.MagicMock()
    app_mock.launch = launcher
    monkeypatch.setattr(webui_mod, "build_ui", lambda: app_mock)
    monkeypatch.setattr(sys, "argv", ["app.py"])

    webui_mod.main()

    launcher.assert_called_once()
    _args, kwargs = launcher.call_args
    assert kwargs["server_name"] == "127.0.0.1"


def test_main_warns_on_public_host(webui, monkeypatch, capsys):
    """An explicit public bind must emit an unauthenticated-exposure warning."""
    webui_mod = webui
    launcher = mock.MagicMock()
    app_mock = mock.MagicMock()
    app_mock.launch = launcher
    monkeypatch.setattr(webui_mod, "build_ui", lambda: app_mock)
    monkeypatch.setattr(sys, "argv", ["app.py", "--host", "0.0.0.0"])

    webui_mod.main()

    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()
    _args, kwargs = launcher.call_args
    assert kwargs["server_name"] == "0.0.0.0"


def test_main_warns_on_share(webui, monkeypatch, capsys):
    """--share must emit a public-tunnel warning."""
    webui_mod = webui
    launcher = mock.MagicMock()
    app_mock = mock.MagicMock()
    app_mock.launch = launcher
    monkeypatch.setattr(webui_mod, "build_ui", lambda: app_mock)
    monkeypatch.setattr(sys, "argv", ["app.py", "--share"])

    webui_mod.main()

    captured = capsys.readouterr()
    assert "share" in captured.err.lower()
    assert "public" in captured.err.lower() or "tunnel" in captured.err.lower()


def test_main_auth_via_cli_args(webui, monkeypatch):
    """--auth-user and --auth-pass must enable Gradio basic auth."""
    webui_mod = webui
    launcher = mock.MagicMock()
    app_mock = mock.MagicMock()
    app_mock.launch = launcher
    monkeypatch.setattr(webui_mod, "build_ui", lambda: app_mock)
    monkeypatch.setattr(sys, "argv", ["app.py", "--auth-user", "admin", "--auth-pass", "s3cret"])

    webui_mod.main()

    _args, kwargs = launcher.call_args
    assert kwargs.get("auth") == ("admin", "s3cret")


def test_main_auth_via_env(webui, monkeypatch):
    """SKILLOPT_WEBUI_USER / SKILLOPT_WEBUI_PASS must enable auth without CLI args."""
    webui_mod = webui
    launcher = mock.MagicMock()
    app_mock = mock.MagicMock()
    app_mock.launch = launcher
    monkeypatch.setattr(webui_mod, "build_ui", lambda: app_mock)
    monkeypatch.setattr(sys, "argv", ["app.py"])
    monkeypatch.setenv("SKILLOPT_WEBUI_USER", "envuser")
    monkeypatch.setenv("SKILLOPT_WEBUI_PASS", "envpass")

    webui_mod.main()

    _args, kwargs = launcher.call_args
    assert kwargs.get("auth") == ("envuser", "envpass")


def test_main_no_auth_by_default(webui, monkeypatch):
    """Without auth args or env vars, no auth must be configured."""
    webui_mod = webui
    launcher = mock.MagicMock()
    app_mock = mock.MagicMock()
    app_mock.launch = launcher
    monkeypatch.setattr(webui_mod, "build_ui", lambda: app_mock)
    monkeypatch.setattr(sys, "argv", ["app.py"])
    monkeypatch.delenv("SKILLOPT_WEBUI_USER", raising=False)
    monkeypatch.delenv("SKILLOPT_WEBUI_PASS", raising=False)

    webui_mod.main()

    _args, kwargs = launcher.call_args
    assert "auth" not in kwargs or kwargs["auth"] is None


def test_scan_outputs_rejects_path_traversal(webui, tmp_path, monkeypatch):
    """The scan_outputs callback must not enumerate directories outside PROJECT_ROOT."""
    monkeypatch.setattr(webui, "PROJECT_ROOT", tmp_path)
    (tmp_path / "outputs").mkdir()
    # Every traversal / escape form is denied at consumption: no rows, no reads.
    for bad in ("/../../etc/passwd", "../outside", "outputs/../../../etc", "C:\\Windows"):
        assert webui.scan_outputs(bad) == [], f"traversal {bad!r} must be denied"


def test_scan_outputs_allows_valid_subdir(webui, tmp_path, monkeypatch):
    """scan_outputs must accept directories within PROJECT_ROOT."""
    monkeypatch.setattr(webui, "PROJECT_ROOT", tmp_path)
    (tmp_path / "outputs" / "bench1" / "run1").mkdir(parents=True)
    (tmp_path / "outputs" / "bench1" / "run1" / "config.yaml").write_text("a: 1\n", encoding="utf-8")
    rows = webui.scan_outputs("outputs")
    assert rows, "valid in-tree output area must be digested"


def test_scan_outputs_callback_consumes_within_project(webui, tmp_path, monkeypatch):
    """The registered scan_outputs callback must digest data only inside PROJECT_ROOT
    at the point data is actually read (traversal denied, in-tree consumed)."""
    monkeypatch.setattr(webui, "PROJECT_ROOT", tmp_path)
    # A traversal arg must be denied at consumption: no rows, no data read.
    assert webui.scan_outputs("/../../etc/passwd") == []
    assert webui.scan_outputs("../outside") == []
    # A valid in-tree output area is digested (config.yaml read per run dir).
    (tmp_path / "outputs/bench1/run1").mkdir(parents=True)
    (tmp_path / "outputs/bench1/run1/config.yaml").write_text("alpha: 1\n", encoding="utf-8")
    rows = webui.scan_outputs("outputs")
    assert rows, f"expected rows from a valid in-tree output area, got {rows!r}"


def test_main_rejects_incomplete_cli_auth_user_only(webui, monkeypatch):
    """--auth-user without --auth-pass must fail closed (never launch)."""
    webui_mod = webui
    launcher = mock.MagicMock()
    app_mock = mock.MagicMock()
    app_mock.launch = launcher
    monkeypatch.setattr(webui_mod, "build_ui", lambda: app_mock)
    monkeypatch.setattr(sys, "argv", ["app.py", "--host", "0.0.0.0", "--auth-user", "admin"])
    with pytest.raises(SystemExit):
        webui_mod.main()
    launcher.assert_not_called()


def test_main_rejects_incomplete_cli_auth_pass_only(webui, monkeypatch):
    """--auth-pass without --auth-user must fail closed (never launch)."""
    webui_mod = webui
    launcher = mock.MagicMock()
    app_mock = mock.MagicMock()
    app_mock.launch = launcher
    monkeypatch.setattr(webui_mod, "build_ui", lambda: app_mock)
    monkeypatch.setattr(sys, "argv", ["app.py", "--host", "0.0.0.0", "--auth-pass", "s3cret"])
    with pytest.raises(SystemExit):
        webui_mod.main()
    launcher.assert_not_called()


def test_main_rejects_incomplete_env_auth_user_only(webui, monkeypatch):
    """Only SKILLOPT_WEBUI_USER set must fail closed (never launch)."""
    webui_mod = webui
    launcher = mock.MagicMock()
    app_mock = mock.MagicMock()
    app_mock.launch = launcher
    monkeypatch.setattr(webui_mod, "build_ui", lambda: app_mock)
    monkeypatch.setattr(sys, "argv", ["app.py", "--host", "0.0.0.0"])
    monkeypatch.setenv("SKILLOPT_WEBUI_USER", "envuser")
    monkeypatch.delenv("SKILLOPT_WEBUI_PASS", raising=False)
    with pytest.raises(SystemExit):
        webui_mod.main()
    launcher.assert_not_called()


def test_main_rejects_incomplete_env_auth_pass_only(webui, monkeypatch):
    """Only SKILLOPT_WEBUI_PASS set must fail closed (never launch)."""
    webui_mod = webui
    launcher = mock.MagicMock()
    app_mock = mock.MagicMock()
    app_mock.launch = launcher
    monkeypatch.setattr(webui_mod, "build_ui", lambda: app_mock)
    monkeypatch.setattr(sys, "argv", ["app.py", "--host", "0.0.0.0"])
    monkeypatch.setenv("SKILLOPT_WEBUI_PASS", "envpass")
    monkeypatch.delenv("SKILLOPT_WEBUI_USER", raising=False)
    with pytest.raises(SystemExit):
        webui_mod.main()
    launcher.assert_not_called()
