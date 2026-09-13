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
