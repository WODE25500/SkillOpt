"""Offline unit tests for the systematic-debugging scenario pack.

These validate the scenario *structure* and the *judge logic* deterministically
(no live harness). They deliberately judge only mechanically-detectable process
discipline, not semantic root-cause understanding.
"""

from __future__ import annotations

import pytest

from skillopt_sleep.adapters.superpowers import (
    SYSTEMATIC_DEBUGGING_SCENARIOS,
    _get_scenarios,
    _pytest_reproduce_fix_order,
    _score_check,
)

_SUPPORTED_OPS = {
    "contains", "not_contains", "regex", "not_regex", "not_regex_unquoted",
    "reports_test_failure", "order", "any_of", "pytest_runs", "pytest_successes",
    "pytest_failures", "pytest_after_edit", "pytest_reproduce_fix_order",
    "harness_test_passes", "protected_files_unchanged",
}


def test_get_scenarios_returns_three():
    scenarios = _get_scenarios("systematic-debugging")
    assert len(scenarios) == 3
    ids = {s["id"] for s in scenarios}
    assert ids == {"reproduce-and-verify-before-done", "failing-test-before-fix", "fix-source-not-test-gamed"}


def test_unknown_skill_raises():
    with pytest.raises(ValueError):
        _get_scenarios("no-such-skill")


@pytest.mark.parametrize(
    "check", [c for s in SYSTEMATIC_DEBUGGING_SCENARIOS for c in s["judge"]["checks"]]
)
def test_every_judge_op_is_supported(check):
    assert check["op"] in _SUPPORTED_OPS


@pytest.mark.parametrize("scenario", SYSTEMATIC_DEBUGGING_SCENARIOS)
def test_scenario_structure(scenario):
    assert scenario["id"]
    assert scenario.get("setup", {}).get("files")
    assert scenario.get("prompt")
    assert scenario["judge"]["type"] == "rule"
    assert scenario["judge"]["checks"]


def test_reproduce_and_verify_before_done_judge():
    scenario = _get_scenarios("systematic-debugging")[0]
    ok = {"pytest_failures": 1, "pytest_reproduce_fix_order": True, "harness_test_passes": True}
    assert all(_score_check(c, "", evidence=ok) for c in scenario["judge"]["checks"])
    # Never reproduced the failure -> must fail closed.
    bad = {"pytest_failures": 0, "pytest_reproduce_fix_order": True, "harness_test_passes": True}
    assert not all(_score_check(c, "", evidence=bad) for c in scenario["judge"]["checks"])
    # Reproduced, but no ordered fail-before-fix -> must fail closed.
    no_order = {"pytest_failures": 1, "pytest_reproduce_fix_order": False, "harness_test_passes": True}
    assert not all(_score_check(c, "", evidence=no_order) for c in scenario["judge"]["checks"])


def test_failing_test_before_fix_judge():
    scenario = _get_scenarios("systematic-debugging")[1]
    ok = {"pytest_failures": 1, "pytest_successes": 1, "pytest_reproduce_fix_order": True, "harness_test_passes": True}
    assert all(_score_check(c, "", evidence=ok) for c in scenario["judge"]["checks"])
    bad = {"pytest_failures": 0, "pytest_successes": 1, "pytest_reproduce_fix_order": True, "harness_test_passes": True}
    assert not all(_score_check(c, "", evidence=bad) for c in scenario["judge"]["checks"])
    # Counts pass but the ORDER is wrong (edit before fail) -> must fail closed.
    wrong_order = {"pytest_failures": 1, "pytest_successes": 1, "pytest_reproduce_fix_order": False, "harness_test_passes": True}
    assert not all(_score_check(c, "", evidence=wrong_order) for c in scenario["judge"]["checks"])


def _audit(nonce: str, lines: list[str], tmp_path) -> object:
    path = tmp_path / "pytest.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_adversarial_order_edit_fail_edit_pass_rejected(tmp_path):
    # The maintainer's adversarial case: edit -> fail -> edit -> pass. The fail is
    # AFTER the first edit, so reproduce-before-fix is violated even though the
    # last run is a pass after the last edit.
    nonce = "abc123"
    log = _audit(nonce, [
        f"{nonce} run 1",
        f"{nonce} edit math_ops.py 100",
        f"{nonce} result 1: 1",
        f"{nonce} edit math_ops.py 200",
        f"{nonce} result 1: 0",
    ], tmp_path)
    assert _pytest_reproduce_fix_order(log, nonce) is False


def test_correct_order_fail_edit_pass_accepted(tmp_path):
    nonce = "abc123"
    log = _audit(nonce, [
        f"{nonce} run 1",
        f"{nonce} result 1: 1",
        f"{nonce} edit math_ops.py 200",
        f"{nonce} result 2: 0",
    ], tmp_path)
    assert _pytest_reproduce_fix_order(log, nonce) is True


def test_pass_before_edit_rejected(tmp_path):
    nonce = "abc123"
    log = _audit(nonce, [
        f"{nonce} run 1",
        f"{nonce} result 1: 0",  # passing run with no preceding failing run
        f"{nonce} edit math_ops.py 200",
    ], tmp_path)
    assert _pytest_reproduce_fix_order(log, nonce) is False


def test_no_edit_fails_closed(tmp_path):
    nonce = "abc123"
    log = _audit(nonce, [
        f"{nonce} run 1",
        f"{nonce} result 1: 1",
    ], tmp_path)
    assert _pytest_reproduce_fix_order(log, nonce) is False


def test_fix_source_not_test_gamed_judge():
    scenario = _get_scenarios("systematic-debugging")[2]
    ok = {"harness_test_passes": True, "protected_files_unchanged": True}
    assert all(_score_check(c, "", evidence=ok) for c in scenario["judge"]["checks"])
    # Test was modified to fake a pass -> must fail closed.
    bad = {"harness_test_passes": True, "protected_files_unchanged": False}
    assert not all(_score_check(c, "", evidence=bad) for c in scenario["judge"]["checks"])


def test_watch_edits_logs_real_mtime_change(tmp_path):
    """The production event producer must log an edit when a source file's mtime
    changes (regression: an unseen path compared mt != mt and was never baselined,
    so the watcher never recorded any edit and the judge always failed closed)."""
    import os
    import threading
    import time

    from skillopt_sleep.adapters.superpowers import _watch_edits

    project = tmp_path / "proj"
    project.mkdir()
    src = project / "math_ops.py"
    src.write_text("x = 1\n", encoding="utf-8")

    audit = tmp_path / "audit.log"
    nonce = "watcherabc"
    stop = threading.Event()
    thread = threading.Thread(
        target=_watch_edits, args=(audit, project, nonce, stop, 0.02), daemon=True
    )
    thread.start()
    try:
        # Let the watcher run its first (baseline) scan, then change the mtime.
        time.sleep(0.1)
        src.write_text("x = 2\n", encoding="utf-8")
        os.utime(src, ns=(1, 10_000_000_000))  # a clearly-different mtime

        deadline = time.time() + 2.0
        while time.time() < deadline:
            if audit.exists() and f"{nonce} edit math_ops.py" in audit.read_text(encoding="utf-8"):
                break
            time.sleep(0.05)

        content = audit.read_text(encoding="utf-8") if audit.exists() else ""
        assert f"{nonce} edit math_ops.py" in content, (
            f"watcher did not log the edit after an mtime change: {content!r}"
        )
    finally:
        stop.set()
        thread.join(timeout=2)
