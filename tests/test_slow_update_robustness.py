"""Tests for slow update field manipulation and longitudinal comparison robustness."""

from __future__ import annotations

import json
import os
import tempfile

from skillopt.optimizer.slow_update import (
    _is_result_success,
    _strip_all_slow_update_fields,
    build_comparison_pairs,
    extract_slow_update_field,
    has_slow_update_field,
    inject_empty_slow_update_field,
    replace_slow_update_field,
    save_comparison_pairs,
)


def test_is_result_success_handles_diverse_metrics() -> None:
    # Explicit hard boolean / float
    assert _is_result_success({"hard": 1}) is True
    assert _is_result_success({"hard": 0}) is False
    assert _is_result_success({"hard": 1.0}) is True
    assert _is_result_success({"hard": "true"}) is True

    # General score metric
    assert _is_result_success({"score": 1.0}) is True
    assert _is_result_success({"score": 0.0}) is False
    assert _is_result_success({"score": 0.8}) is True
    assert _is_result_success({"score": 0.2}) is False

    # Exact match metric
    assert _is_result_success({"exact_match": 1}) is True
    assert _is_result_success({"exact_match": 0}) is False

    # Soft metric threshold
    assert _is_result_success({"soft": 1.0}) is True
    assert _is_result_success({"soft": 0.5}) is False

    # Empty or invalid inputs
    assert _is_result_success({}) is False
    assert _is_result_success(None) is False
    assert _is_result_success("not-a-dict") is False


def test_build_comparison_pairs_categorization() -> None:
    items = [
        {"id": "task-1", "question": "Task 1 description"},
        {"id": "task-2", "question": "Task 2 description"},
        {"id": "task-3", "question": "Task 3 description"},
        {"id": "task-4", "question": "Task 4 description"},
    ]

    # task-1: improved (fail -> pass)
    # task-2: regressed (pass -> fail)
    # task-3: persistent_fail (fail -> fail)
    # task-4: stable_success (pass -> pass)
    results_prev = [
        {"id": "task-1", "score": 0.0, "predicted_answer": "wrong1"},
        {"id": "task-2", "exact_match": 1.0, "predicted_answer": "correct2"},
        {"id": "task-3", "hard": 0, "fail_reason": "timeout"},
        {"id": "task-4", "hard": 1, "predicted_answer": "correct4"},
    ]
    results_curr = [
        {"id": "task-1", "score": 1.0, "predicted_answer": "correct1"},
        {"id": "task-2", "exact_match": 0.0, "predicted_answer": "wrong2"},
        {"id": "task-3", "hard": 0, "fail_reason": "wrong_syntax"},
        {"id": "task-4", "hard": 1, "predicted_answer": "correct4"},
    ]

    pairs = build_comparison_pairs(results_prev, results_curr, items)
    assert len(pairs) == 4

    by_id = {p["id"]: p for p in pairs}
    assert by_id["task-1"]["category"] == "improved"
    assert by_id["task-1"]["prev"]["hard"] == 0
    assert by_id["task-1"]["curr"]["hard"] == 1

    assert by_id["task-2"]["category"] == "regressed"
    assert by_id["task-2"]["prev"]["hard"] == 1
    assert by_id["task-2"]["curr"]["hard"] == 0

    assert by_id["task-3"]["category"] == "persistent_fail"
    assert by_id["task-3"]["prev"]["hard"] == 0
    assert by_id["task-3"]["curr"]["hard"] == 0

    assert by_id["task-4"]["category"] == "stable_success"
    assert by_id["task-4"]["prev"]["hard"] == 1
    assert by_id["task-4"]["curr"]["hard"] == 1


def test_slow_update_field_lifecycle() -> None:
    skill = "# Main Skill\n\nRule 1: Always verify assumptions."

    assert not has_slow_update_field(skill)
    injected = inject_empty_slow_update_field(skill)
    assert has_slow_update_field(injected)
    assert extract_slow_update_field(injected) == ""

    # Idempotent inject
    assert inject_empty_slow_update_field(injected) == injected

    # Replace field with guidance
    guidance = "Avoid premature tool exit on partial output."
    updated = replace_slow_update_field(injected, guidance)
    assert has_slow_update_field(updated)
    assert extract_slow_update_field(updated) == guidance

    # Stripping all fields
    stripped = _strip_all_slow_update_fields(updated)
    assert not has_slow_update_field(stripped)
    assert stripped == skill.rstrip()


def test_save_comparison_pairs_writes_valid_json() -> None:
    pairs = [
        {
            "id": "item-1",
            "task": "Test task",
            "category": "improved",
            "prev": {"hard": 0},
            "curr": {"hard": 1},
        }
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = os.path.join(tmpdir, "comparison.json")
        save_comparison_pairs(pairs, out_file)
        assert os.path.exists(out_file)
        with open(out_file, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == pairs
