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
    _score_check,
)

_SUPPORTED_OPS = {
    "contains", "not_contains", "regex", "not_regex", "not_regex_unquoted",
    "reports_test_failure", "order", "any_of", "pytest_runs", "pytest_successes",
    "pytest_failures", "pytest_after_edit", "harness_test_passes",
    "protected_files_unchanged",
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
    ok = {"pytest_failures": 1, "pytest_after_edit": True, "harness_test_passes": True}
    assert all(_score_check(c, "", evidence=ok) for c in scenario["judge"]["checks"])
    # Never reproduced the failure -> must fail closed.
    bad = {"pytest_failures": 0, "pytest_after_edit": True, "harness_test_passes": True}
    assert not all(_score_check(c, "", evidence=bad) for c in scenario["judge"]["checks"])


def test_failing_test_before_fix_judge():
    scenario = _get_scenarios("systematic-debugging")[1]
    ok = {"pytest_failures": 1, "pytest_successes": 1, "harness_test_passes": True}
    assert all(_score_check(c, "", evidence=ok) for c in scenario["judge"]["checks"])
    bad = {"pytest_failures": 0, "pytest_successes": 1, "harness_test_passes": True}
    assert not all(_score_check(c, "", evidence=bad) for c in scenario["judge"]["checks"])


def test_fix_source_not_test_gamed_judge():
    scenario = _get_scenarios("systematic-debugging")[2]
    ok = {"harness_test_passes": True, "protected_files_unchanged": True}
    assert all(_score_check(c, "", evidence=ok) for c in scenario["judge"]["checks"])
    # Test was modified to fake a pass -> must fail closed.
    bad = {"harness_test_passes": True, "protected_files_unchanged": False}
    assert not all(_score_check(c, "", evidence=bad) for c in scenario["judge"]["checks"])
