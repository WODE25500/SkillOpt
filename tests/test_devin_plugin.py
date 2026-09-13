"""Tests for the Devin MCP plugin: tool schema, ATIF-v1.7 harvest, path expansion."""
import contextlib
import importlib
import io
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

# Allow importing from the plugin directory (mirrors tests/test_mcp_schema.py)
PLUGIN = os.path.join(os.path.dirname(__file__), "..", "plugins", "devin")
sys.path.insert(0, PLUGIN)

import harvest_devin as hw  # noqa: E402
import mcp_server  # noqa: E402

FIXTURES = os.path.join(PLUGIN, "fixtures")
INSTALLER = os.path.join(PLUGIN, "install.sh")


def _call(name="sleep_status", arguments=None, **params):
    call_params = {"name": name, "arguments": {} if arguments is None else arguments}
    call_params.update(params)
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": call_params,
    }


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _find_session_jsonl(out_dir):
    for root, _dirs, files in os.walk(os.path.join(out_dir, "projects")):
        for name in files:
            if name.endswith(".jsonl"):
                return _read_jsonl(os.path.join(root, name))
    raise AssertionError("no session jsonl written")


class TestDevinMcpSchema(unittest.TestCase):
    def test_tools_are_the_sleep_interface(self):
        names = {t["name"] for t in mcp_server.TOOLS}
        self.assertEqual(names, {"sleep_status", "sleep_dry_run", "sleep_run",
                                 "sleep_adopt", "sleep_harvest",
                                 "sleep_schedule", "sleep_unschedule"})

    def test_actions_map_to_engine_subcommands(self):
        expected = {"sleep_status": "status", "sleep_dry_run": "dry-run",
                    "sleep_run": "run", "sleep_adopt": "adopt",
                    "sleep_harvest": "harvest", "sleep_schedule": "schedule",
                    "sleep_unschedule": "unschedule"}
        for t in mcp_server.TOOLS:
            self.assertEqual(t["action"], expected[t["name"]])

    def test_backends_in_enum(self):
        backends = mcp_server._TOOL_SCHEMA["properties"]["backend"]["enum"]
        for b in ["mock", "claude", "codex", "copilot", "handoff"]:
            self.assertIn(b, backends)

    def test_schema_has_key_engine_params(self):
        # parity with plugins/copilot's schema (tests/test_plugin_sync.py)
        props = set(mcp_server._TOOL_SCHEMA["properties"].keys())
        for param in {"project", "backend", "scope", "source", "model",
                      "tasks_file", "target_skill_path", "staging", "skills",
                      "all_skills", "legacy", "max_sessions",
                      "max_tasks", "lookback_hours", "auto_adopt", "json",
                      "edit_budget", "hour", "minute"}:
            self.assertIn(param, props)

    def test_adopt_selection_schema_types(self):
        props = mcp_server._TOOL_SCHEMA["properties"]
        self.assertEqual(props["staging"]["type"], "string")
        self.assertEqual(props["skills"]["type"], "array")
        self.assertEqual(props["skills"]["items"]["type"], "string")
        self.assertEqual(props["all_skills"]["type"], "boolean")
        self.assertEqual(props["legacy"]["type"], "boolean")

    def test_adopt_forwards_fanout_selection_as_argv_without_sync(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="adopted\n", stderr=""
        )
        arguments = {
            "project": "/tmp/devin workspace",
            "staging": "/tmp/night with spaces",
            "skills": ["alpha", "--leading-dash", "space ; $(literal)"],
        }
        with mock.patch.object(
            mcp_server.subprocess, "run", return_value=completed
        ) as run:
            result = mcp_server._run_engine("adopt", arguments)

        self.assertEqual(result.text, "[engine]\nadopted")
        self.assertEqual(result.returncode, 0)
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(
            command[-7:],
            [
                "--staging", "/tmp/night with spaces",
                "--skill", "alpha",
                "--skill=--leading-dash",
                "--skill", "space ; $(literal)",
            ],
        )
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_adoption_never_performs_post_engine_copy(self):
        cases = (
            ("bare legacy success", {}, 0),
            ("explicit legacy success", {"legacy": True}, 0),
            ("bare adoption refused", {}, 2),
            ("explicit legacy failed", {"legacy": True}, 1),
            ("per-skill success", {"skills": ["alpha"]}, 0),
            ("all-skills success", {"all_skills": True}, 0),
        )
        for name, selection, returncode in cases:
            with self.subTest(name=name):
                completed = subprocess.CompletedProcess(
                    args=[], returncode=returncode, stdout="result", stderr=""
                )
                arguments = {"project": "/tmp/devin-workspace", **selection}
                with mock.patch.object(
                    mcp_server.subprocess, "run", return_value=completed
                ) as run:
                    result = mcp_server._run_engine("adopt", arguments)

                command = run.call_args.args[0]
                if selection.get("legacy"):
                    self.assertIn("--legacy", command)
                if selection.get("all_skills"):
                    self.assertIn("--all-skills", command)
                self.assertEqual(result.returncode, returncode)
                self.assertNotIn("synced", result.text)

    def test_adopt_rejects_non_array_skills_without_spawning(self):
        with mock.patch.object(mcp_server.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "skills must be an array"):
                mcp_server._run_engine("adopt", {"skills": "alpha"})

        run.assert_not_called()


class TestDevinMcpRuntimeValidation(unittest.TestCase):
    def test_malformed_request_envelopes_return_json_rpc_errors(self):
        cases = (
            (None, "request must be a JSON object"),
            ({"method": "ping"}, "jsonrpc must be '2.0'"),
            ({"jsonrpc": "2.0", "id": 1, "method": 4}, "method must be"),
            ({"jsonrpc": "2.0", "id": 1, "method": "ping", "params": False},
             "params must be an object"),
            ({"jsonrpc": "2.0", "id": [], "method": "ping"}, "id must be"),
            ({"jsonrpc": "2.0", "id": 1, "method": "ping", "extra": 1},
             "unknown request member"),
        )
        for request, message in cases:
            with self.subTest(request=request):
                response = mcp_server.handle(request)
                self.assertEqual(response["error"]["code"], -32600)
                self.assertIn(message, response["error"]["message"])
                if type(request) is dict and type(request.get("id")) not in {str, int}:
                    self.assertIsNone(response["id"])

    def test_params_and_arguments_require_known_properties_and_objects(self):
        cases = (
            (_call(extra=1), "unknown params member"),
            (_call(arguments=[]), "arguments must be an object"),
            (_call(arguments={"unknown": True}), "unknown argument"),
        )
        for request, message in cases:
            with self.subTest(request=request), mock.patch.object(
                mcp_server.subprocess, "run"
            ) as run, mock.patch.object(mcp_server, "_run_harvest") as harvest:
                response = mcp_server.handle(request)
                self.assertEqual(response["error"]["code"], -32602)
                self.assertIn(message, response["error"]["message"])
                run.assert_not_called()
                harvest.assert_not_called()

    def test_wrong_scalar_types_and_bounds_are_rejected_before_harvest(self):
        cases = (
            ({"auto_adopt": "false"}, "auto_adopt must be a boolean"),
            ({"json": "false"}, "json must be a boolean"),
            ({"progress": 1}, "progress must be a boolean"),
            ({"max_sessions": False}, "max_sessions must be an integer"),
            ({"lookback_hours": -1}, "lookback_hours must be between"),
            ({"backend": "unknown"}, "unsupported backend"),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments), mock.patch.object(
                mcp_server, "_run_harvest"
            ) as harvest, mock.patch.object(mcp_server.subprocess, "run") as run:
                response = mcp_server.handle(_call("sleep_run", arguments))
                self.assertEqual(response["error"]["code"], -32602)
                self.assertIn(message, response["error"]["message"])
                harvest.assert_not_called()
                run.assert_not_called()

    def test_schedule_bounds_and_action_specific_arguments_are_rejected(self):
        cases = (
            ("sleep_schedule", {"hour": -1}, "hour must be between"),
            ("sleep_schedule", {"minute": 60}, "minute must be between"),
            ("sleep_status", {"hour": 3}, "valid only for sleep_schedule"),
            ("sleep_status", {"legacy": False}, "valid only for sleep_adopt"),
        )
        for tool, arguments, message in cases:
            with self.subTest(tool=tool, arguments=arguments), mock.patch.object(
                mcp_server, "_run_harvest"
            ) as harvest, mock.patch.object(mcp_server.subprocess, "run") as run:
                response = mcp_server.handle(_call(tool, arguments))
                self.assertEqual(response["error"]["code"], -32602)
                self.assertIn(message, response["error"]["message"])
                harvest.assert_not_called()
                run.assert_not_called()

    def test_adoption_arrays_and_selection_modes_are_strict(self):
        cases = (
            ({"all_skills": "false"}, "all_skills must be a boolean"),
            ({"legacy": "false"}, "legacy must be a boolean"),
            ({"skills": [None]}, "skills entry must be a string"),
            ({"skills": ["\t"]}, "control characters"),
            ({"skills": ["alpha", " alpha "]}, "must be unique"),
            ({"skills": ["alpha"], "legacy": True}, "choose at most one"),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments), mock.patch.object(
                mcp_server.subprocess, "run"
            ) as run:
                response = mcp_server.handle(_call("sleep_adopt", arguments))
                self.assertEqual(response["error"]["code"], -32602)
                self.assertIn(message, response["error"]["message"])
                run.assert_not_called()

    def test_every_string_boolean_and_integer_contract_is_exact_and_bounded(self):
        for key in mcp_server._STRING_ARGS:
            action = "adopt" if key == "staging" else "status"
            with self.subTest(kind="string", key=key), self.assertRaisesRegex(
                ValueError, f"{key} must be a string"
            ):
                mcp_server._validate_tool_arguments(action, {key: 1})
        for key in mcp_server._BOOLEAN_ARGS:
            action = "adopt" if key in {"all_skills", "legacy"} else "status"
            with self.subTest(kind="boolean", key=key), self.assertRaisesRegex(
                ValueError, f"{key} must be a boolean"
            ):
                mcp_server._validate_tool_arguments(action, {key: "false"})
        for key, (minimum, maximum) in mcp_server._INTEGER_BOUNDS.items():
            action = "schedule" if key in {"hour", "minute"} else "status"
            with self.subTest(kind="integer-bool", key=key), self.assertRaisesRegex(
                ValueError, f"{key} must be an integer"
            ):
                mcp_server._validate_tool_arguments(action, {key: True})
            for value in (minimum - 1, maximum + 1):
                with self.subTest(kind="integer-bound", key=key, value=value), \
                        self.assertRaisesRegex(ValueError, f"{key} must be between"):
                    mcp_server._validate_tool_arguments(action, {key: value})
            self.assertEqual(
                mcp_server._validate_tool_arguments(action, {key: minimum})[key],
                minimum,
            )
            self.assertEqual(
                mcp_server._validate_tool_arguments(action, {key: maximum})[key],
                maximum,
            )

    def test_harvest_failure_stops_engine_and_does_not_use_stale_cache(self):
        failure = mcp_server.EngineResult("conversion failed", 7, "bad ATIF")
        with mock.patch.object(
            mcp_server, "_run_harvest", return_value=failure
        ) as harvest, mock.patch.object(mcp_server.subprocess, "run") as run:
            result = mcp_server._run_engine("status", {})

        harvest.assert_called_once_with()
        run.assert_not_called()
        self.assertEqual(result.returncode, 7)
        self.assertIn("conversion failed", result.text)
        self.assertIn("bad ATIF", result.text)

    def test_harvest_subprocess_returncode_is_preserved(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=6, stdout="conversion stopped\n", stderr="bad source\n"
        )
        with mock.patch.object(mcp_server.subprocess, "run", return_value=completed):
            result = mcp_server._run_harvest()
        self.assertEqual(result.returncode, 6)
        self.assertEqual(result.text, "conversion stopped")
        self.assertEqual(result.diagnostics, "bad source")

    def test_engine_status_maps_to_mcp_error_and_handoff_states(self):
        cases = (
            (0, False, "ok"),
            (1, True, "error"),
            (3, False, "handoff_pending"),
        )
        for returncode, is_error, status in cases:
            with self.subTest(returncode=returncode), mock.patch.object(
                mcp_server, "_run_engine",
                return_value=mcp_server.EngineResult("engine output", returncode),
            ):
                result = mcp_server.handle(_call())["result"]
                self.assertIs(result["isError"], is_error)
                self.assertEqual(result["structuredContent"]["status"], status)
                self.assertEqual(result["structuredContent"]["exit_code"], returncode)

    def test_json_stdout_is_parseable_without_harvest_or_stderr_prefixes(self):
        harvest = mcp_server.EngineResult("converted 3 sessions", 0, "harvest note")
        engine = subprocess.CompletedProcess(
            args=[], returncode=0, stdout='{"nights": 4}\n', stderr="engine note\n"
        )
        with mock.patch.object(
            mcp_server, "_run_harvest", return_value=harvest
        ), mock.patch.object(mcp_server.subprocess, "run", return_value=engine):
            run = mcp_server._run_engine("status", {"json": True})

        self.assertEqual(json.loads(run.text), {"nights": 4})
        self.assertNotIn("harvest", run.text)
        self.assertIn("converted 3 sessions", run.diagnostics)
        self.assertIn("engine note", run.diagnostics)

    def test_json_tool_result_includes_parsed_structured_output(self):
        run = mcp_server.EngineResult('{"pending": true}', 3, "answer prompts")
        with mock.patch.object(mcp_server, "_run_engine", return_value=run):
            result = mcp_server.handle(_call(arguments={"json": True}))["result"]
        self.assertEqual(result["structuredContent"]["output"], {"pending": True})
        self.assertEqual(result["structuredContent"]["status"], "handoff_pending")
        self.assertFalse(result["isError"])

    def test_main_emits_parse_error_for_malformed_json(self):
        output = io.StringIO()
        with mock.patch.object(sys, "stdin", io.StringIO("not-json\n")), \
                contextlib.redirect_stdout(output):
            self.assertEqual(mcp_server.main(), 0)
        response = json.loads(output.getvalue())
        self.assertEqual(response["error"]["code"], -32700)


class TestClaudeHomeExpansion(unittest.TestCase):
    """Regression: ~ must be expanded even when CLAUDE_HOME comes from the env
    (the documented mcp-config sets SKILLOPT_DEVIN_CLAUDE_HOME="~/...")."""

    def test_env_tilde_is_expanded(self):
        # Re-insert the devin plugin path at position 0 so importlib.reload
        # picks up this module, not plugins/copilot/mcp_server.py when both
        # test modules are loaded in the same process.
        sys.path.insert(0, PLUGIN)
        os.environ["SKILLOPT_DEVIN_CLAUDE_HOME"] = "~/.skillopt-sleep-devin"
        try:
            importlib.reload(mcp_server)
            self.assertFalse(mcp_server.CLAUDE_HOME.startswith("~"))
            self.assertEqual(mcp_server.CLAUDE_HOME,
                             os.path.expanduser("~/.skillopt-sleep-devin"))
        finally:
            del os.environ["SKILLOPT_DEVIN_CLAUDE_HOME"]
            importlib.reload(mcp_server)


@unittest.skipIf(os.name == "nt", "Devin installer and hook are POSIX shell scripts")
class TestDevinInstaller(unittest.TestCase):
    def _run_installer(self, project, home, installer=INSTALLER):
        env = os.environ.copy()
        env["HOME"] = home
        return subprocess.run(
            ["bash", installer, project],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

    @staticmethod
    def _skillopt_hook():
        config_path = os.path.join(PLUGIN, "hooks", "hooks.v1.json")
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)["SessionEnd"][0]

    def test_new_install_and_hook_marker(self):
        with tempfile.TemporaryDirectory() as d:
            project = os.path.join(d, "project with spaces")
            home = os.path.join(d, "home")
            os.makedirs(project)
            os.makedirs(home)

            self._run_installer(project, home)

            config_path = os.path.join(project, ".devin", "hooks.v1.json")
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            self.assertEqual(config["SessionEnd"], [self._skillopt_hook()])

            hook_path = os.path.join(
                project, ".devin", "hooks", "skillopt-sleep-on-session-end.sh"
            )
            self.assertTrue(os.stat(hook_path).st_mode & stat.S_IXUSR)
            env = os.environ.copy()
            env.update(HOME=home, DEVIN_PROJECT_DIR=project)
            subprocess.run([hook_path], check=True, env=env)
            marker = os.path.join(home, ".skillopt-sleep", "session-end.log")
            with open(marker, encoding="utf-8") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 1)
            self.assertTrue(lines[0].endswith(f"\t{project}\n"))

    def test_hook_is_non_blocking_without_home(self):
        env = os.environ.copy()
        env.pop("HOME", None)
        result = subprocess.run(
            [os.path.join(PLUGIN, "hooks", "on-session-end.sh")],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stderr, "")

    def test_existing_config_without_session_end_is_extended(self):
        unrelated = [
            {"matcher": "", "hooks": [{"type": "command", "command": "./pre.sh"}]}
        ]
        with tempfile.TemporaryDirectory() as d:
            project = os.path.join(d, "project")
            home = os.path.join(d, "home")
            devin_dir = os.path.join(project, ".devin")
            os.makedirs(devin_dir)
            os.makedirs(home)
            config_path = os.path.join(devin_dir, "hooks.v1.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump({"PreToolUse": unrelated}, f)

            self._run_installer(project, home)

            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            self.assertEqual(config["PreToolUse"], unrelated)
            self.assertEqual(config["SessionEnd"], [self._skillopt_hook()])

    def test_existing_hooks_are_preserved_and_reinstall_is_idempotent(self):
        existing_session_end = {
            "matcher": "existing",
            "hooks": [{"type": "command", "command": "./existing.sh"}],
        }
        unrelated = [
            {"matcher": "", "hooks": [{"type": "command", "command": "./pre.sh"}]}
        ]
        with tempfile.TemporaryDirectory() as d:
            project = os.path.join(d, "project")
            home = os.path.join(d, "home")
            devin_dir = os.path.join(project, ".devin")
            os.makedirs(devin_dir)
            os.makedirs(home)
            hooks_dir = os.path.join(devin_dir, "hooks")
            os.makedirs(hooks_dir)
            legacy_hook = os.path.join(hooks_dir, "on-session-end.sh")
            with open(legacy_hook, "w", encoding="utf-8") as f:
                f.write("#!/bin/sh\n# existing project hook\n")
            config_path = os.path.join(devin_dir, "hooks.v1.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"PreToolUse": unrelated, "SessionEnd": [existing_session_end]}, f
                )

            self._run_installer(project, home)
            self._run_installer(project, home)

            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            self.assertEqual(config["PreToolUse"], unrelated)
            self.assertIn(existing_session_end, config["SessionEnd"])
            self.assertEqual(config["SessionEnd"].count(self._skillopt_hook()), 1)
            with open(legacy_hook, encoding="utf-8") as f:
                self.assertEqual(f.read(), "#!/bin/sh\n# existing project hook\n")

    def test_malformed_existing_config_fails_without_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            project = os.path.join(d, "project")
            home = os.path.join(d, "home")
            devin_dir = os.path.join(project, ".devin")
            os.makedirs(devin_dir)
            os.makedirs(home)
            config_path = os.path.join(devin_dir, "hooks.v1.json")
            original = "{not-json\n"
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(original)

            with self.assertRaises(subprocess.CalledProcessError):
                self._run_installer(project, home)
            with open(config_path, encoding="utf-8") as f:
                self.assertEqual(f.read(), original)

    def test_registration_path_is_shell_quoted(self):
        with tempfile.TemporaryDirectory() as d:
            plugin_copy = os.path.join(
                d, "repo with spaces $dollar `tick` 'quote'", "plugins", "devin"
            )
            shutil.copytree(PLUGIN, plugin_copy)
            project = os.path.join(d, "project")
            home = os.path.join(d, "home")
            os.makedirs(project)
            os.makedirs(home)

            result = self._run_installer(
                project,
                home,
                installer=os.path.join(plugin_copy, "install.sh"),
            )

            command_line = next(
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip().startswith("-- python3 ")
            )
            self.assertEqual(
                shlex.split(command_line),
                ["--", "python3", os.path.join(plugin_copy, "mcp_server.py")],
            )


class TestDevinHarvest(unittest.TestCase):
    def test_atif_fixture_yields_gradeable_task(self):
        with tempfile.TemporaryDirectory() as out:
            n = hw.harvest_devin_transcripts(FIXTURES, out, ["/tmp/proj"])
            self.assertEqual(n, 1)

            outcomes = _read_jsonl(os.path.join(out, "outcomes.jsonl"))
            self.assertEqual(len(outcomes), 1)
            o = outcomes[0]
            self.assertEqual(o["verifier"], "tests")
            self.assertTrue(o["success"])
            self.assertIn("repro", o["reference"])

            # the converted transcript carries the grouping key on the user turn
            session = _find_session_jsonl(out)
            user_turn = next(r for r in session if r["type"] == "user")
            self.assertIn("taskKey", user_turn)


if __name__ == "__main__":
    unittest.main()
