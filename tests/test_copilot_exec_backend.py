from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import scripts.eval_only as eval_script
import scripts.train as train_script
import skillopt.model as model
from skillopt.config import flatten_config
from skillopt.model import backend_config, copilot_backend
from skillopt.model import codex_harness as harness
from skillopt.model.common import normalize_backend_name


@pytest.fixture(autouse=True)
def restore_backend_state() -> Iterator[None]:
    optimizer_backend = backend_config.get_optimizer_backend()
    target_backend = backend_config.get_target_backend()
    copilot_path = backend_config.COPILOT_EXEC_PATH
    copilot_home = backend_config.COPILOT_EXEC_HOME
    copilot_tools = backend_config.COPILOT_EXEC_ALLOW_ALL_TOOLS
    chat_optimizer_model = backend_config.COPILOT_CHAT_OPTIMIZER_MODEL
    chat_target_model = backend_config.COPILOT_CHAT_TARGET_MODEL
    chat_timeout = backend_config.COPILOT_CHAT_TIMEOUT
    retries = backend_config.EXEC_EMPTY_RESPONSE_RETRIES
    env = {
        key: os.environ.get(key)
        for key in (
            "OPTIMIZER_BACKEND",
            "TARGET_BACKEND",
            "COPILOT_EXEC_PATH",
            "COPILOT_EXEC_HOME",
            "COPILOT_EXEC_ALLOW_ALL_TOOLS",
            "COPILOT_CHAT_OPTIMIZER_MODEL",
            "COPILOT_CHAT_TARGET_MODEL",
            "COPILOT_CHAT_TIMEOUT",
        )
    }
    yield
    # copilot_backend records call counts, and those now feed the aggregate
    # model.get_token_summary(); leaving them behind pollutes other suites.
    copilot_backend.reset_token_tracker()
    backend_config.OPTIMIZER_BACKEND = optimizer_backend
    backend_config.TARGET_BACKEND = target_backend
    backend_config.COPILOT_EXEC_PATH = copilot_path
    backend_config.COPILOT_EXEC_HOME = copilot_home
    backend_config.COPILOT_EXEC_ALLOW_ALL_TOOLS = copilot_tools
    backend_config.COPILOT_CHAT_OPTIMIZER_MODEL = chat_optimizer_model
    backend_config.COPILOT_CHAT_TARGET_MODEL = chat_target_model
    backend_config.COPILOT_CHAT_TIMEOUT = chat_timeout
    backend_config.EXEC_EMPTY_RESPONSE_RETRIES = retries
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.mark.parametrize("alias", ["copilot_exec"])
def test_aliases_normalize_to_copilot_exec(alias: str) -> None:
    assert normalize_backend_name(alias) == "copilot_exec"


@pytest.mark.parametrize("alias", ["copilot", "copilot_cli", "github_copilot"])
def test_bare_copilot_aliases_normalize_to_the_cli_authenticated_chat_backend(
    alias: str,
) -> None:
    assert normalize_backend_name(alias) == "copilot_chat"
    assert model.set_backend(alias) == "copilot_chat"
    assert backend_config.get_optimizer_backend() == "copilot_chat"
    assert backend_config.get_target_backend() == "copilot_chat"


def test_set_backend_routes_target_to_copilot_and_keeps_chat_optimizer() -> None:
    assert model.set_backend("copilot_exec") == "copilot_exec"
    # The CLI agent is the *target*; the optimizer must stay a chat model
    # because it has to emit structured skill edits.
    assert backend_config.get_target_backend() == "copilot_exec"
    assert backend_config.get_optimizer_backend() == "openai_chat"
    assert backend_config.is_target_exec_backend() is True
    assert model.get_backend_name() == "copilot_exec"


def test_chat_target_refuses_exec_backend() -> None:
    model.set_backend("copilot_exec")
    with pytest.raises(NotImplementedError):
        model.chat_target(system="s", user="u")


def test_configure_rejects_non_boolean_allow_all_tools() -> None:
    with pytest.raises(ValueError):
        model.configure_copilot_exec(allow_all_tools="sometimes")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", "1"), ("false", "0"), ("yes", "1"), ("off", "0"), ("1", "1"), ("weird", "0")],
)
def test_env_allow_all_tools_is_normalized_on_load(raw, expected, monkeypatch) -> None:
    # A user may set the env var to 'true'/'false'; loading it raw used to make
    # get_copilot_exec_config() raise. It is now normalized to '0'/'1' (unknown
    # values fall back to the safe '0'), matching the other boolean-ish flags.
    import importlib

    monkeypatch.setenv("COPILOT_EXEC_ALLOW_ALL_TOOLS", raw)
    importlib.reload(backend_config)
    try:
        assert backend_config.COPILOT_EXEC_ALLOW_ALL_TOOLS == expected
        assert backend_config.get_copilot_exec_config()["allow_all_tools"] == expected
    finally:
        monkeypatch.delenv("COPILOT_EXEC_ALLOW_ALL_TOOLS", raising=False)
        importlib.reload(backend_config)


def test_parse_jsonl_concatenates_assistant_messages_and_ignores_noise() -> None:
    raw = "\n".join(
        [
            "not json at all",
            json.dumps({"type": "tool.call", "data": {"content": "ignored"}}),
            json.dumps({"type": "assistant.message", "data": {"content": "first"}}),
            "{ broken json",
            "[]",
            "null",
            json.dumps({"type": "assistant.message", "data": "invalid"}),
            json.dumps({"type": "assistant.message", "data": {"content": "second"}}),
            json.dumps({"type": "assistant.message", "data": {}}),
        ]
    )
    assert copilot_backend.parse_copilot_jsonl(raw) == "first\nsecond"


def test_parse_jsonl_returns_empty_for_no_assistant_messages() -> None:
    assert copilot_backend.parse_copilot_jsonl('{"type":"tool.call","data":{}}') == ""
    assert copilot_backend.parse_copilot_jsonl("") == ""


def _fake_run(captured: dict, stdout: str, returncode: int = 0):
    def _run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env") or {}
        captured["cwd"] = kwargs.get("cwd")
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")

    return _run


def test_run_copilot_exec_builds_isolated_readonly_command(monkeypatch, tmp_path) -> None:
    model.set_backend("copilot")
    model.configure_copilot_exec(path="copilot", home=str(tmp_path / "home"), allow_all_tools=False)
    monkeypatch.setenv("COPILOT_ALLOW_ALL", "true")
    captured: dict = {}
    stdout = json.dumps({"type": "assistant.message", "data": {"content": "answer"}})
    monkeypatch.setattr(harness.subprocess, "run", _fake_run(captured, stdout))

    response, raw = harness.run_copilot_exec(work_dir=str(tmp_path), prompt="solve it", model="", timeout=30)

    assert response == "answer"
    assert "COPILOT CLI ATTEMPT 1" in raw
    cmd = captured["cmd"]
    # Prompt is a single argv element -- never interpolated into a shell string.
    assert "solve it" in " ".join(cmd)
    assert cmd[cmd.index("-p") + 1].endswith("solve it") or "solve it" in cmd[cmd.index("-p") + 1]
    assert "--output-format" in cmd and cmd[cmd.index("--output-format") + 1] == "json"
    # Startup isolation: no user MCP servers or custom instructions.
    assert "--disable-builtin-mcps" in cmd
    assert "--no-custom-instructions" in cmd
    # Read-only rollout must NOT bypass the CLI approval gate.
    assert "--allow-all-tools" not in cmd
    assert "COPILOT_ALLOW_ALL" not in captured["env"]
    assert captured["env"]["COPILOT_HOME"] == str(tmp_path / "home")


def test_copilot_chat_calls_reach_the_aggregate_token_summary(monkeypatch) -> None:
    # The CLI reports no token counts (documented caveat), but the call counts
    # are real and must not be dropped from the run's token snapshots.
    model.set_backend("copilot")
    model.reset_token_tracker()
    monkeypatch.setattr(copilot_backend, "_invoke", lambda *a, **k: "hi")

    model.chat_target(system="s", user="u", stage="target")

    summary = model.get_token_summary()
    assert summary["target"]["calls"] >= 1
    assert summary["_total"]["calls"] >= 1

    model.reset_token_tracker()
    assert model.get_token_summary()["_total"]["calls"] == 0


def test_config_defaults_do_not_clobber_copilot_env_settings() -> None:
    # The shipped base config must not pin this to false: trainer.py and
    # eval_only.py pass cfg.get("copilot_exec_allow_all_tools") straight into
    # configure_copilot_exec(), and a non-None value overwrites the env var --
    # which would make the documented COPILOT_EXEC_ALLOW_ALL_TOOLS=1 opt-in
    # impossible. null leaves the env (default off) in charge.
    import yaml

    root = Path(__file__).resolve().parent.parent
    with open(root / "configs" / "_base_" / "default.yaml", encoding="utf-8") as fh:
        base = yaml.safe_load(fh)
    assert base["model"]["copilot_exec_allow_all_tools"] is None
    assert base["model"]["copilot_chat_timeout"] is None

    # None must be a no-op, so prior environment-derived settings survive.
    model.configure_copilot_exec(allow_all_tools=True)
    model.configure_copilot_exec(allow_all_tools=None)
    assert backend_config.COPILOT_EXEC_ALLOW_ALL_TOOLS == "1"
    model.configure_copilot_chat(timeout=123)
    model.configure_copilot_chat(timeout=None)
    assert backend_config.COPILOT_CHAT_TIMEOUT == "123"


def test_copilot_backends_keep_a_real_deployment_fallback() -> None:
    # This table also feeds the shared Azure deployment fallback in the entry
    # points (cfg.get("optimizer_model", default_model_for_backend(backend))),
    # and the shipped base config sets no optimizer_model/target_model -- so a
    # "" here left `--backend copilot_exec` with an EMPTY optimizer deployment
    # even though that role is still a real openai_chat model. The CLI's own
    # model comes from copilot_chat_optimizer_model / _target_model instead.
    from skillopt.model.common import default_model_for_backend

    assert default_model_for_backend("copilot_exec") == "gpt-4o"
    assert default_model_for_backend("copilot_chat") == "gpt-4o"


def test_train_copilot_exec_omits_inherited_openai_target_model(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "skillopt-train",
            "--config",
            str(root / "configs" / "searchqa" / "default.yaml"),
            "--backend",
            "copilot_exec",
        ],
    )

    cfg = train_script.load_config(train_script.parse_args())

    assert cfg["target_backend"] == "copilot_exec"
    assert cfg["target_model"] == ""


def _write_copilot_yaml_config(tmp_path: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "copilot_exec.yaml"
    config_path.write_text(
        f"""\
_base_: {json.dumps(str(root / "configs" / "searchqa" / "default.yaml"))}

model:
  backend: copilot_exec

env:
  out_root: {json.dumps(str(tmp_path / "out"))}
""",
        encoding="utf-8",
    )
    return config_path


def test_train_resolves_copilot_exec_from_structured_yaml(monkeypatch, tmp_path) -> None:
    config_path = _write_copilot_yaml_config(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["skillopt-train", "--config", str(config_path)],
    )

    cfg = train_script.load_config(train_script.parse_args())

    assert cfg["optimizer_backend"] == "openai_chat"
    assert cfg["target_backend"] == "copilot_exec"
    assert cfg["target_model"] == ""


def test_train_keeps_default_structured_azure_backend(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "skillopt-train",
            "--config",
            str(root / "configs" / "searchqa" / "default.yaml"),
        ],
    )

    cfg = train_script.load_config(train_script.parse_args())

    assert cfg["model_backend"] == "azure_openai"
    assert cfg["optimizer_backend"] == "openai_chat"
    assert cfg["target_backend"] == "openai_chat"
    assert cfg["optimizer_model"] == "gpt-5.5"
    assert cfg["target_model"] == "gpt-5.5"


def test_train_resolves_copilot_exec_from_legacy_flat_yaml(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "copilot_exec_flat.yaml"
    config_path.write_text(
        f"""\
backend: copilot_exec
optimizer_backend: openai_chat
target_backend: openai_chat
optimizer_model: gpt-5.5
target_model: gpt-5.5
env: searchqa
out_root: {json.dumps(str(tmp_path / "out"))}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["skillopt-train", "--config", str(config_path)],
    )

    cfg = train_script.load_config(train_script.parse_args())

    assert cfg["model_backend"] == "copilot_exec"
    assert cfg["optimizer_backend"] == "openai_chat"
    assert cfg["target_backend"] == "copilot_exec"
    assert cfg["target_model"] == ""


def test_eval_resolves_copilot_exec_from_structured_yaml(monkeypatch, tmp_path) -> None:
    config_path = _write_copilot_yaml_config(tmp_path)
    skill_path = tmp_path / "skill.md"
    skill_path.write_text("# Test skill\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "skillopt-eval",
            "--config",
            str(config_path),
            "--skill",
            str(skill_path),
        ],
    )

    observed = {}
    monkeypatch.setattr(eval_script, "configure_azure_openai", lambda **_kwargs: None)
    monkeypatch.setattr(
        eval_script,
        "set_optimizer_backend",
        lambda value: observed.__setitem__("optimizer_backend", value),
    )
    monkeypatch.setattr(
        eval_script,
        "set_target_backend",
        lambda value: observed.__setitem__("target_backend", value),
    )
    monkeypatch.setattr(
        eval_script,
        "set_optimizer_deployment",
        lambda value: observed.__setitem__("optimizer_model", value),
    )

    class StopAfterModelConfiguration(Exception):
        pass

    def capture_target_model(value):
        observed["target_model"] = value
        raise StopAfterModelConfiguration

    monkeypatch.setattr(eval_script, "set_target_deployment", capture_target_model)

    with pytest.raises(StopAfterModelConfiguration):
        eval_script.main()

    assert observed == {
        "optimizer_backend": "openai_chat",
        "target_backend": "copilot_exec",
        "optimizer_model": "gpt-5.5",
        "target_model": "",
    }


def test_train_copilot_exec_preserves_explicit_target_model(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "skillopt-train",
            "--config",
            str(root / "configs" / "searchqa" / "default.yaml"),
            "--backend",
            "copilot_exec",
            "--target_model",
            "copilot-model-id",
        ],
    )

    cfg = train_script.load_config(train_script.parse_args())

    assert cfg["target_backend"] == "copilot_exec"
    assert cfg["target_model"] == "copilot-model-id"


def test_no_response_error_includes_cli_output(monkeypatch, tmp_path) -> None:
    # copilot_exec persists no artifacts, so a bare "returned no response"
    # would leave an empty/invalid JSONL stream undebuggable.
    model.set_backend("copilot")
    model.configure_copilot_exec(path="copilot", home="", allow_all_tools=False)
    captured: dict = {}
    noise = '{"type":"tool.call","data":{"content":"NOISE-MARKER"}}'
    monkeypatch.setattr(harness.subprocess, "run", _fake_run(captured, noise))

    with pytest.raises(RuntimeError) as exc:
        harness.run_copilot_exec(work_dir=str(tmp_path), prompt="p", model="", timeout=30)

    assert "returned no response" in str(exc.value)
    assert "NOISE-MARKER" in str(exc.value)


def test_allow_all_tools_requires_both_opt_in_and_file_edits(monkeypatch, tmp_path) -> None:
    model.set_backend("copilot")
    stdout = json.dumps({"type": "assistant.message", "data": {"content": "ok"}})

    # Opted in, but read-only rollout -> still gated.
    model.configure_copilot_exec(allow_all_tools=True)
    captured: dict = {}
    monkeypatch.setattr(harness.subprocess, "run", _fake_run(captured, stdout))
    harness.run_copilot_exec(work_dir=str(tmp_path), prompt="p", model="", timeout=30, allow_file_edits=False)
    assert "--allow-all-tools" not in captured["cmd"]

    # File edits requested but operator did not opt in -> still gated.
    model.configure_copilot_exec(allow_all_tools=False)
    captured = {}
    monkeypatch.setattr(harness.subprocess, "run", _fake_run(captured, stdout))
    harness.run_copilot_exec(work_dir=str(tmp_path), prompt="p", model="", timeout=30, allow_file_edits=True)
    assert "--allow-all-tools" not in captured["cmd"]

    # Both -> allowed.
    model.configure_copilot_exec(allow_all_tools=True)
    captured = {}
    monkeypatch.setattr(harness.subprocess, "run", _fake_run(captured, stdout))
    harness.run_copilot_exec(work_dir=str(tmp_path), prompt="p", model="", timeout=30, allow_file_edits=True)
    assert "--allow-all-tools" in captured["cmd"]


def test_nonzero_exit_raises_with_detail(monkeypatch, tmp_path) -> None:
    model.set_backend("copilot")
    captured: dict = {}
    monkeypatch.setattr(harness.subprocess, "run", _fake_run(captured, "", returncode=3))
    with pytest.raises(RuntimeError, match="exit code 3"):
        harness.run_copilot_exec(work_dir=str(tmp_path), prompt="p", model="", timeout=30)


# --- copilot trace redaction: quoted JSON embedded in non-JSON surf --------


def _fake_run_with_stderr(captured: dict, stderr: str, stdout: str = "", returncode: int = 0):
    def _run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env") or {}
        captured["cwd"] = kwargs.get("cwd")
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)

    return _run


def test_copilot_trace_redacts_prefixed_quoted_json() -> None:
    # The CLI prefixes real-stream stderr with non-JSON text (e.g. "warning:"),
    # so the whole line is NOT valid JSON and used to fall back to the
    # unquoted-keyword redactor, leaking quoted JSON secret payloads.
    raw = 'warning: {"token":"plain-secret"}\nstill here {"api_key": "ghp_abcdef"}\n'
    out = harness._redact_copilot_trace(raw)
    assert "plain-secret" not in out
    assert "ghp_abcdef" not in out
    assert "[REDACTED]" in out


def test_copilot_trace_redacts_quoted_json_across_valid_and_prefixed_lines() -> None:
    # One structurally-valid JSON line + one non-JSON prefixed line; both carry
    # a quoted secret and both must be stripped.
    raw = (
        '{"token": "first-secret"}\n'
        'warning: {"a": {"refreshToken": "second-secret"}}\n'
    )
    out = harness._redact_copilot_trace(raw)
    assert "first-secret" not in out
    assert "second-secret" not in out
    assert "[REDACTED]" in out


def test_run_copilot_exec_redacts_secret_in_stderr_trace(monkeypatch, tmp_path) -> None:
    model.set_backend("copilot")
    model.configure_copilot_exec(path="copilot", home="", allow_all_tools=False)
    captured: dict = {}
    stdout = json.dumps({"type": "assistant.message", "data": {"content": "ok"}})
    stderr = 'warning: {"token":"plain-secret"}\n'
    monkeypatch.setattr(harness.subprocess, "run", _fake_run_with_stderr(captured, stderr, stdout=stdout))

    response, raw = harness.run_copilot_exec(work_dir=str(tmp_path), prompt="p", model="", timeout=30)

    assert response == "ok"
    assert "plain-secret" not in raw
    assert "[REDACTED]" in raw


def test_run_copilot_exec_redacts_secret_in_error_detail(monkeypatch, tmp_path) -> None:
    model.set_backend("copilot")
    model.configure_copilot_exec(path="copilot", home="", allow_all_tools=False)
    captured: dict = {}
    stderr = 'warning: {"token":"plain-secret"}\n'
    monkeypatch.setattr(harness.subprocess, "run", _fake_run_with_stderr(captured, stderr, returncode=3))

    with pytest.raises(RuntimeError) as exc:
        harness.run_copilot_exec(work_dir=str(tmp_path), prompt="p", model="", timeout=30)

    assert "plain-secret" not in str(exc.value)
    assert "[REDACTED]" in str(exc.value)


def test_redact_cursor_error_redacts_quoted_json_with_spaces() -> None:
    # The unquoted keyword regex truncates at the FIRST whitespace, so a quoted
    # JSON value that contains spaces used to leak its remainder. This is the
    # shared string redactor used by the copilot fallback AND cursor stderr.
    out = harness._redact_cursor_error('warning: {"token": "ab cd ef"}')
    assert "ab cd ef" not in out
    assert "[REDACTED]" in out


def test_sanitize_cursor_trace_redacts_quoted_json_in_stderr() -> None:
    # Cursor's trace redactor routes real stderr lines through the same shared
    # `_redact_cursor_error`; it must not leak quoted JSON values with spaces.
    raw = (
        "===== CURSOR CLI ATTEMPT 1 =====\n"
        "[stderr]\n"
        'warning: {"token": "a b c"}\n'
    )
    out = harness._sanitize_cursor_trace(raw, preserve_markers=True)
    assert "a b c" not in out
    assert "[REDACTED]" in out


def test_redact_cursor_error_keeps_token_budget_diagnostics() -> None:
    # Endswith-based key detection must NOT scrub non-credential diagnostics that
    # merely contain a secret-ish substring (token_count / token_budget /
    # secret_version), while still scrubbing real credential keys.
    out = harness._redact_cursor_error(
        '{"token_count": 3, "token_budget": 10, "secret_version": "v1", '
        '"api_key": "x", "refreshToken": "y"}'
    )
    # Non-credential diagnostics with a secret-ish substring are KEPT.
    assert '"token_count": 3' in out
    assert '"token_budget": 10' in out
    assert '"secret_version": "v1"' in out
    # Real credential keys are still scrubbed.
    assert '"api_key": "x"' not in out
    assert '"refreshToken": "y"' not in out
    assert out.count("[REDACTED]") == 2


def test_redact_cursor_error_redacts_bare_bearer_key() -> None:
    # A standalone "bearer" key is a credential marker (the exact set covers it);
    # the common "Authorization": "Bearer ..." form is also scrubbed via the
    # authorization key.
    out = harness._redact_cursor_error('{"bearer": "mysecrettoken"}')
    assert '"bearer": "mysecrettoken"' not in out
    assert "[REDACTED]" in out



def test_run_target_exec_dispatches_to_copilot(monkeypatch, tmp_path) -> None:
    model.set_backend("copilot_exec")
    called: dict = {}

    def _fake(**kwargs):
        called.update(kwargs)
        return "resp", "raw"

    monkeypatch.setattr(harness, "run_copilot_exec", _fake)
    response, raw = harness.run_target_exec(work_dir=str(tmp_path), prompt="p", model="m", timeout=15)
    assert (response, raw) == ("resp", "raw")
    assert called["model"] == "m"


# --- copilot_chat: CLI-authenticated path (no separate provider API key) -----


def test_copilot_alias_selects_cli_authenticated_chat_backend() -> None:
    assert normalize_backend_name("copilot") == "copilot_chat"
    assert model.set_backend("copilot") == "copilot_chat"
    # Both halves run through the CLI, so no separate provider API key is used.
    assert backend_config.get_optimizer_backend() == "copilot_chat"
    assert backend_config.get_target_backend() == "copilot_chat"
    assert backend_config.is_optimizer_chat_backend() is True
    assert backend_config.is_target_chat_backend() is True
    assert backend_config.is_target_exec_backend() is False
    # Both roles unify to the canonical label, like claude_chat/qwen_chat, not
    # the generic "copilot_chat+copilot_chat".
    assert model.get_backend_name() == "copilot_chat"


def test_copilot_exec_still_pairs_with_a_chat_optimizer() -> None:
    # Guards the split: `copilot` fills both roles, `copilot_exec` does not.
    assert model.set_backend("copilot_exec") == "copilot_exec"
    assert backend_config.get_optimizer_backend() == "openai_chat"
    assert backend_config.get_target_backend() == "copilot_exec"


def test_chat_backend_composes_system_and_user_into_one_prompt(monkeypatch) -> None:
    model.set_backend("copilot")
    captured: dict = {}
    monkeypatch.setenv("COPILOT_ALLOW_ALL", "true")
    stdout = json.dumps({"type": "assistant.message", "data": {"content": "done"}})
    monkeypatch.setattr(copilot_backend.subprocess, "run", _fake_run(captured, stdout))

    text, usage = model.chat_target(system="SYS", user="USR", retries=1)

    assert text == "done"
    assert usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    cmd = captured["cmd"]
    prompt = cmd[cmd.index("-p") + 1]
    assert "SYS" in prompt and "USR" in prompt
    assert "--disable-builtin-mcps" in cmd and "--no-custom-instructions" in cmd
    assert "--available-tools=" in cmd
    # A chat call must never be granted unattended tool use.
    assert "--allow-all-tools" not in cmd
    assert "COPILOT_ALLOW_ALL" not in captured["env"]


def test_chat_backend_routes_optimizer_and_target_models(monkeypatch) -> None:
    model.set_backend("copilot")
    model.configure_copilot_chat(optimizer_model="opt-m", target_model="tgt-m")
    stdout = json.dumps({"type": "assistant.message", "data": {"content": "x"}})

    captured: dict = {}
    monkeypatch.setattr(copilot_backend.subprocess, "run", _fake_run(captured, stdout))
    model.chat_optimizer(system="s", user="u", retries=1)
    assert captured["cmd"][captured["cmd"].index("--model") + 1] == "opt-m"

    captured = {}
    monkeypatch.setattr(copilot_backend.subprocess, "run", _fake_run(captured, stdout))
    model.chat_target(system="s", user="u", retries=1)
    assert captured["cmd"][captured["cmd"].index("--model") + 1] == "tgt-m"


def test_chat_backend_raises_on_empty_response(monkeypatch) -> None:
    model.set_backend("copilot")
    captured: dict = {}
    monkeypatch.setattr(copilot_backend.subprocess, "run", _fake_run(captured, ""))
    with pytest.raises(RuntimeError, match="after 1 retries"):
        model.chat_target(system="s", user="u", retries=1)


def test_chat_backend_raises_on_nonzero_exit(monkeypatch) -> None:
    model.set_backend("copilot")
    captured: dict = {}
    monkeypatch.setattr(copilot_backend.subprocess, "run", _fake_run(captured, "", returncode=2))
    with pytest.raises(RuntimeError, match="exit code 2"):
        model.chat_target(system="s", user="u", retries=1)


def test_configure_copilot_chat_rejects_bad_timeout() -> None:
    with pytest.raises(ValueError):
        model.configure_copilot_chat(timeout="0")
    with pytest.raises(ValueError):
        model.configure_copilot_chat(timeout="soon")


def test_messages_variant_flattens_roles(monkeypatch) -> None:
    model.set_backend("copilot")
    captured: dict = {}
    stdout = json.dumps({"type": "assistant.message", "data": {"content": "ok"}})
    monkeypatch.setattr(copilot_backend.subprocess, "run", _fake_run(captured, stdout))

    text, _ = model.chat_target_messages(
        messages=[
            {"role": "system", "content": "SYSTEM_TEXT"},
            {"role": "user", "content": "USER_TEXT"},
        ],
        retries=1,
    )
    assert text == "ok"
    prompt = captured["cmd"][captured["cmd"].index("-p") + 1]
    assert "SYSTEM_TEXT" in prompt and "USER_TEXT" in prompt


def test_messages_variant_rejects_non_text_multipart_content(monkeypatch) -> None:
    model.set_backend("copilot")
    invoked = False

    def _unexpected_invoke(*args, **kwargs):
        nonlocal invoked
        invoked = True
        return "unexpected"

    monkeypatch.setattr(copilot_backend, "_invoke", _unexpected_invoke)

    with pytest.raises(NotImplementedError, match="unsupported multipart type 'image_url'"):
        model.chat_target_messages(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                    ],
                }
            ],
            retries=1,
        )
    assert invoked is False


@pytest.mark.parametrize("role", ["optimizer", "target"])
def test_messages_variant_returns_compat_message_when_requested(monkeypatch, role) -> None:
    from skillopt.model.common import CompatAssistantMessage

    model.set_backend("copilot")
    monkeypatch.setattr(copilot_backend, "_invoke", lambda *a, **k: "ok")

    chat_messages = getattr(model, f"chat_{role}_messages")
    message, usage = chat_messages(
        messages=[{"role": "user", "content": "hello"}],
        return_message=True,
        retries=1,
    )

    assert isinstance(message, CompatAssistantMessage)
    assert message.content == "ok"
    assert message.tool_calls == []
    assert usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


@pytest.mark.parametrize(
    ("role", "kwargs"),
    [
        (
            "optimizer",
            {"tools": [{"type": "function", "function": {"name": "search"}}]},
        ),
        ("target", {"tool_choice": "auto"}),
    ],
)
def test_messages_variant_fails_fast_for_unsupported_tools(
    monkeypatch, role, kwargs
) -> None:
    model.set_backend("copilot")
    invoked = False

    def _unexpected_invoke(*args, **call_kwargs):
        nonlocal invoked
        invoked = True
        return "unexpected"

    monkeypatch.setattr(copilot_backend, "_invoke", _unexpected_invoke)

    chat_messages = getattr(model, f"chat_{role}_messages")
    with pytest.raises(NotImplementedError, match="does not support caller-supplied tools"):
        chat_messages(
            messages=[{"role": "user", "content": "hello"}],
            retries=1,
            **kwargs,
        )
    assert invoked is False


def test_copilot_config_keys_survive_flattening() -> None:
    # Guards the 4-place wiring: config.py map, both scripts, and trainer.
    flat = flatten_config(
        {
            "model": {
                "copilot_exec_path": "/opt/copilot",
                "copilot_exec_home": "/tmp/cph",
                "copilot_exec_allow_all_tools": True,
                "copilot_chat_optimizer_model": "opt-m",
                "copilot_chat_target_model": "tgt-m",
                "copilot_chat_timeout": 900,
            }
        }
    )
    assert flat["copilot_exec_path"] == "/opt/copilot"
    assert flat["copilot_exec_home"] == "/tmp/cph"
    assert flat["copilot_exec_allow_all_tools"] is True
    assert flat["copilot_chat_optimizer_model"] == "opt-m"
    assert flat["copilot_chat_target_model"] == "tgt-m"
    assert flat["copilot_chat_timeout"] == 900


@pytest.mark.parametrize("script", ["train", "eval_only"])
def test_cli_exposes_copilot_backend_and_flags(script: str) -> None:
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / f"{script}.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "copilot_chat" in result.stdout
    assert "copilot_exec" in result.stdout
    for flag in (
        "--copilot_exec_path",
        "--copilot_exec_home",
        "--copilot_exec_allow_all_tools",
        "--copilot_chat_optimizer_model",
        "--copilot_chat_target_model",
        "--copilot_chat_timeout",
    ):
        assert flag in result.stdout
    # train.py delegates backend configuration to the trainer; eval_only wires
    # it directly. Inspect call nodes rather than source spelling/formatting.
    applier_path = (
        script_path
        if script == "eval_only"
        else root / "skillopt" / "engine" / "trainer.py"
    )
    tree = ast.parse(applier_path.read_text(encoding="utf-8"))
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {"configure_copilot_chat", "configure_copilot_exec"} <= calls
