"""Offline tests for the ``--compare-baseline`` wiring (no POSIX / claude CLI).

The live baseline run needs an authenticated claude CLI on a POSIX host, so it
cannot be exercised here. But the *data-flow* that the flag drives is pure and is
pinned offline by injecting ``evaluate_fn``: the baseline must be evaluated with
``candidate=None`` (i.e. WITHOUT the candidate skill), and the merge must attach
it as ``results["_baseline"]`` only when the flag is set. Errors propagate to the
CLI layer unchanged so fail-closed still fires.
"""

from __future__ import annotations

import pytest

from skillopt_sleep.adapters.superpowers import _evaluate_with_baseline


def _candidate_result():
    return {"skill": "systematic-debugging", "score": 0.8, "passed": 3, "failed": 1, "scenarios": []}


def _baseline_result():
    return {"skill": "systematic-debugging", "score": 0.4, "passed": 1, "failed": 2, "scenarios": []}


def test_compare_baseline_merges_baseline_with_candidate_none(monkeypatch):
    """When the flag is set, the baseline run must pass candidate=None and be merged."""
    calls: list[tuple] = []

    def fake_eval(skill, candidate, scenario=None, pinned_sha=None):
        calls.append((skill, candidate, scenario, pinned_sha))
        return _baseline_result() if candidate is None else _candidate_result()

    res = _evaluate_with_baseline(
        "systematic-debugging", "skills/s.md", "scenario-x", "sha", True,
        evaluate_fn=fake_eval,
    )
    assert res["_baseline"]["score"] == 0.4, "baseline must be merged"
    # Baseline evaluated WITHOUT the candidate; scenario + sha forwarded as-is.
    assert calls == [
        ("systematic-debugging", "skills/s.md", "scenario-x", "sha"),
        ("systematic-debugging", None, "scenario-x", "sha"),
    ]


def test_no_baseline_without_flag(monkeypatch):
    """Without the flag, only the candidate run happens and no _baseline key is added."""
    calls: list[tuple] = []

    def fake_eval(skill, candidate, scenario=None, pinned_sha=None):
        calls.append((skill, candidate, scenario, pinned_sha))
        return _candidate_result()

    res = _evaluate_with_baseline(
        "systematic-debugging", "skills/s.md", None, "sha", False,
        evaluate_fn=fake_eval,
    )
    assert "_baseline" not in res, "no baseline should be attached when flag is off"
    assert calls == [("systematic-debugging", "skills/s.md", None, "sha")]


def test_baseline_error_propagates_fail_closed(monkeypatch):
    """A baseline-run failure must raise (so the CLI exits non-zero), never silently pass."""
    import skillopt_sleep.adapters.superpowers as sp

    def ok_eval(skill, candidate, scenario=None, pinned_sha=None):
        if candidate is None:
            raise RuntimeError("expected-auth-cli-missing")
        return _candidate_result()

    with pytest.raises(RuntimeError):
        sp._evaluate_with_baseline(
            "systematic-debugging", "skills/s.md", "scenario-x", "sha", True,
            evaluate_fn=ok_eval,
        )
