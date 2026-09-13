"""Real Gradio build/launch smoke test (requires the `webui` extra).

Verifies the WebUI builds (and launches where the environment allows) on the
installed Gradio without a TypeError or ignored-argument warning, and that the
theme is placed on the right object for the installed major version — as a
specific ``Soft`` schema, not merely "some non-None theme". Gradio 6 assigns a
default theme when none is passed, so ``theme is not None`` would pass even if
``Soft`` was never applied; these tests assert the exact theme.
"""

from __future__ import annotations

import pytest

pytest.importorskip("gradio")

import gradio as gr  # noqa: E402

import skillopt_webui.app as app  # noqa: E402


def _assert_soft_theme(theme) -> None:
    assert isinstance(theme, gr.themes.Soft), f"theme was not Soft: {theme!r}"


def test_webui_builds_theme_on_blocks():
    """build_ui() must not raise; for Gradio <6 the theme lives on Blocks."""
    ui = app.build_ui()
    assert ui is not None
    if app._GRADIO_MAJOR < 6:
        # Gradio 5 and below place the theme on Blocks. Assert it is the schema
        # we chose (Soft), not merely any non-None default.
        _assert_soft_theme(ui.theme)
    else:
        # Gradio 6 moved the theme to launch(); build_ui must NOT bake it into
        # the Blocks kwargs (that would be an ignored argument).
        assert ui.theme is None, "theme must live on launch kwargs for Gradio >=6"


def test_launch_kwargs_follow_production_main_path():
    # The production path applies Soft; the test must not inject its own theme.
    kwargs = app.build_launch_kwargs(
        server_name="127.0.0.1", server_port=7860, share=False
    )
    if app._GRADIO_MAJOR >= 6:
        _assert_soft_theme(kwargs.get("theme"))
        assert kwargs["theme"].name == "soft"
    else:
        assert "theme" not in kwargs, "theme must live on Blocks for Gradio <6"


def test_webui_builds_and_launches_theme():
    ui = app.build_ui()
    launch_kwargs = app.build_launch_kwargs(
        server_name="127.0.0.1", server_port=7860, share=False
    )
    launch_kwargs["prevent_thread_lock"] = True
    try:
        ui.launch(**launch_kwargs)
    except ValueError as exc:
        # Headless/sandboxed environments may not expose localhost; that is an
        # environment limitation, not a theme-compatibility bug.
        if "localhost is not accessible" in str(exc):
            pytest.skip("headless environment blocks localhost launch")
        raise
    try:
        # Assert Soft specifically: gradio >=6 would hand back a default theme
        # if Soft were not actually applied via the production kwargs.
        _assert_soft_theme(ui.theme)
    finally:
        ui.close()


def test_gradio_major_detected():
    # The constant must reflect the installed Gradio major.
    assert app._GRADIO_MAJOR == int(gr.__version__.split(".")[0])
