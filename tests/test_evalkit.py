"""Paired A/B evalkit: known-answer stats, seeded null calibration, RESULTS replay."""
from __future__ import annotations

import io
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from skillopt_sleep.evalkit import (
    MAX_BOOTSTRAP_DRAWS,
    MAX_MCNEMAR_DISCORDANTS,
    EvalkitError,
    bootstrap_delta_ci,
    compare,
    compare_aa,
    exact_mcnemar_p,
    format_markdown,
    mcnemar_from_counts,
    mcnemar_paired,
    reconstruct_paired_from_rates,
)
from skillopt_sleep.evalkit import (
    main as evalkit_main,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "evalkit")


def _load(name: str):
    with open(os.path.join(FIXTURE_DIR, name), encoding="utf-8") as f:
        return json.load(f)


class TestMcNemarKnownAnswer(unittest.TestCase):
    def test_textbook_2x2_chi2_and_exact(self):
        fx = _load("mcnemar_textbook.json")
        res = mcnemar_from_counts(
            fx["both_success"], fx["a_only"], fx["b_only"], fx["both_fail"],
        )
        self.assertAlmostEqual(res.chi2, fx["chi2"], places=12)
        self.assertAlmostEqual(res.p_chi2, fx["p_chi2"], places=12)
        self.assertAlmostEqual(res.p_exact, fx["p_exact"], places=12)
        self.assertTrue(res.significant)
        self.assertEqual(res.n, 100)

    def test_zero_discordants_is_not_significant(self):
        res = mcnemar_from_counts(20, 0, 0, 5)
        self.assertEqual(res.chi2, 0.0)
        self.assertEqual(res.p_chi2, 1.0)
        self.assertEqual(res.p_exact, 1.0)
        self.assertFalse(res.significant)

    def test_paired_vectors_match_counts(self):
        a = [1, 1, 1, 0, 0]
        b = [1, 0, 1, 1, 0]
        res = mcnemar_paired(a, b)
        self.assertEqual(res.both_success, 2)
        self.assertEqual(res.a_only, 1)
        self.assertEqual(res.b_only, 1)
        self.assertEqual(res.both_fail, 1)
        self.assertAlmostEqual(res.p_exact, exact_mcnemar_p(1, 1))

    def test_exact_tail_is_stable_above_one_thousand_discordants(self):
        self.assertEqual(exact_mcnemar_p(700, 700), 1.0)
        value = exact_mcnemar_p(590, 611)
        # Independent reference from scipy.stats.binomtest(590, 1201, 0.5).
        self.assertAlmostEqual(value, 0.563883319454372, places=10)
        self.assertEqual(value, exact_mcnemar_p(611, 590))

    def test_exact_tail_has_a_documented_resource_limit(self):
        with self.assertRaisesRegex(EvalkitError, "discordant-pair limit"):
            exact_mcnemar_p(MAX_MCNEMAR_DISCORDANTS + 1, 0)
        with self.assertRaisesRegex(EvalkitError, "discordant-pair limit"):
            mcnemar_from_counts(0, MAX_MCNEMAR_DISCORDANTS, 1, 0)

    def test_exact_tail_accumulates_from_a_lazy_stream(self):
        real_fsum = math.fsum

        def consume(values):
            self.assertNotIsInstance(values, (list, tuple))
            return real_fsum(values)

        with patch("skillopt_sleep.evalkit.math.fsum", side_effect=consume):
            self.assertAlmostEqual(exact_mcnemar_p(590, 611), 0.563883319454372)

    def test_invalid_counts_and_binary_vectors_are_refused(self):
        for args in ((-1, 0), (True, 0), (1.5, 0)):
            with self.subTest(args=args), self.assertRaises(EvalkitError):
                exact_mcnemar_p(*args)
        with self.assertRaises(EvalkitError):
            mcnemar_from_counts(1, -1, 0, 1)
        with self.assertRaises(EvalkitError):
            mcnemar_paired([], [])
        with self.assertRaises(EvalkitError):
            mcnemar_paired([0, 2], [0, 1])


class TestBootstrapCoverage(unittest.TestCase):
    def test_identical_series_ci_collapses_to_zero(self):
        a = [1, 0, 1, 0, 1, 0, 1, 0]
        ci = bootstrap_delta_ci(a, a, n_boot=2000, seed=7)
        self.assertEqual(ci.low, 0.0)
        self.assertEqual(ci.high, 0.0)
        self.assertEqual(ci.mean, 0.0)

    def test_known_shift_ci_excludes_zero(self):
        # A always 0, B always 1: delta = 1 exactly, CI is [1, 1].
        a = [0] * 30
        b = [1] * 30
        ci = bootstrap_delta_ci(a, b, n_boot=1000, seed=1)
        self.assertEqual(ci.low, 1.0)
        self.assertEqual(ci.high, 1.0)

    def test_seed_is_deterministic(self):
        a = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
        b = [1, 1, 1, 0, 0, 1, 1, 0, 0, 1]
        x = bootstrap_delta_ci(a, b, n_boot=500, seed=99)
        y = bootstrap_delta_ci(a, b, n_boot=500, seed=99)
        self.assertEqual((x.low, x.high, x.mean), (y.low, y.high, y.mean))

    def test_invalid_alpha_and_bootstrap_counts_are_refused(self):
        for alpha in (0, 1, -0.1, 1.1, float("nan"), float("inf"), True, "0.05"):
            with self.subTest(alpha=alpha), self.assertRaises(EvalkitError):
                bootstrap_delta_ci([0], [1], alpha=alpha)
        for n_boot in (0, -1, 1.5, True, 1_000_001):
            with self.subTest(n_boot=n_boot), self.assertRaises(EvalkitError):
                bootstrap_delta_ci([0], [1], n_boot=n_boot)
        for seed in (True, 1.5, "7"):
            with self.subTest(seed=seed), self.assertRaises(EvalkitError):
                bootstrap_delta_ci([0], [1], n_boot=10, seed=seed)

    def test_total_bootstrap_draws_are_bounded(self):
        n_tasks = 100
        excessive_bootstraps = MAX_BOOTSTRAP_DRAWS // n_tasks + 1
        with self.assertRaisesRegex(EvalkitError, "paired-draw limit"):
            bootstrap_delta_ci(
                [0] * n_tasks,
                [1] * n_tasks,
                n_boot=excessive_bootstraps,
            )

    def test_malformed_direct_api_scores_are_contract_errors(self):
        for value in (
            "not-a-number", "1", True, None, object(), 10**1000,
            float("nan"), float("inf"),
        ):
            with self.subTest(value=value), self.assertRaises(EvalkitError):
                bootstrap_delta_ci([0], [value], n_boot=10)

    def test_direct_api_scores_must_be_in_the_unit_interval(self):
        for value in (-1, -0.00001, 1.00001, 2):
            with self.subTest(value=value), self.assertRaisesRegex(
                EvalkitError, "between 0 and 1"
            ):
                bootstrap_delta_ci([0], [value], n_boot=10)


class TestAACalibration(unittest.TestCase):
    def test_aa_does_not_reject(self):
        man = _load("aa_manifest.json")
        out = _load("aa_outcomes.json")
        report = compare_aa(man["ids"], out["outcomes"], n_boot=2000, seed=42)
        self.assertEqual(report.delta, 0.0)
        self.assertIsNotNone(report.mcnemar)
        self.assertFalse(report.mcnemar.significant)
        self.assertEqual(report.mcnemar.p_exact, 1.0)
        self.assertLessEqual(report.bootstrap.low, 0.0)
        self.assertGreaterEqual(report.bootstrap.high, 0.0)

    def test_exact_test_controls_type_one_error_under_a_seeded_null(self):
        # Unlike comparing an array with itself, this exercises non-zero,
        # symmetrically distributed discordance and can catch p-value inflation.
        rng = random.Random(20260824)
        trials = 500
        rejected = 0
        for _ in range(trials):
            a = []
            b = []
            for _task in range(80):
                left = int(rng.random() < 0.5)
                right = 1 - left if rng.random() < 0.30 else left
                a.append(left)
                b.append(right)
            rejected += int(mcnemar_paired(a, b, alpha=0.05).significant)
        rate = rejected / trials
        self.assertGreaterEqual(rate, 0.025)
        self.assertLessEqual(rate, 0.075)

    def test_paired_bootstrap_has_nominal_coverage_under_a_seeded_null(self):
        rng = random.Random(20260825)
        trials = 160
        covered = 0
        for trial in range(trials):
            a = []
            b = []
            for _task in range(80):
                left = int(rng.random() < 0.5)
                right = 1 - left if rng.random() < 0.30 else left
                a.append(left)
                b.append(right)
            ci = bootstrap_delta_ci(a, b, n_boot=300, seed=10_000 + trial)
            covered += int(ci.low <= 0.0 <= ci.high)
        coverage = covered / trials
        self.assertGreaterEqual(coverage, 0.90)
        self.assertLessEqual(coverage, 0.99)


class TestCompareContracts(unittest.TestCase):
    def test_mismatched_ids_are_refused(self):
        with self.assertRaises(EvalkitError) as ctx:
            compare(["t1", "t2"], {"t1": 1, "t2": 0}, {"t1": 1, "t3": 0})
        self.assertIn("must equal the manifest", str(ctx.exception))

    def test_duplicate_manifest_ids_refused(self):
        with self.assertRaises(EvalkitError):
            compare(["t1", "t1"], {"t1": 1}, {"t1": 0})

    def test_empty_count_table_refused(self):
        with self.assertRaisesRegex(EvalkitError, "at least one paired observation"):
            mcnemar_from_counts(0, 0, 0, 0)

    def test_empty_manifest_refused(self):
        with self.assertRaises(EvalkitError):
            compare([], {}, {})

    def test_task_ids_are_nonempty_strings_without_coercion(self):
        for task_id in (1, {}, [], None, True, "", "   "):
            with self.subTest(task_id=task_id), self.assertRaisesRegex(
                EvalkitError, "non-empty JSON strings"
            ):
                compare([task_id], {}, {})
        with self.assertRaisesRegex(EvalkitError, "non-empty JSON strings"):
            compare(["t1"], {"": 0}, {"t1": 1})

    def test_scores_are_json_numbers_without_coercion(self):
        for score in ("0", "1.0", True, False, None):
            with self.subTest(score=score), self.assertRaisesRegex(
                EvalkitError, "JSON numbers"
            ):
                compare(["t1"], {"t1": score}, {"t1": 1})
        with self.assertRaisesRegex(EvalkitError, "finite and between 0 and 1"):
            compare(["t1"], {"t1": 10**1000}, {"t1": 1})
        with self.assertRaisesRegex(EvalkitError, "contain only a seeds array"):
            compare(
                ["t1"],
                {"t1": {"seeds": [0, 1], "ignored": 1}},
                {"t1": [0, 1]},
            )

    def test_graded_refused_without_flag(self):
        with self.assertRaises(EvalkitError) as ctx:
            compare(["t1", "t2"], {"t1": 0.4, "t2": 0.9}, {"t1": 0.5, "t2": 0.8})
        self.assertIn("allow-graded", str(ctx.exception))

    def test_graded_bootstrap_only(self):
        report = compare(
            ["t1", "t2"],
            {"t1": 0.4, "t2": 0.9},
            {"t1": 0.5, "t2": 0.8},
            allow_graded=True,
            n_boot=500,
            seed=3,
        )
        self.assertIsNone(report.mcnemar)
        self.assertTrue(any("graded" in n for n in report.notes))
        self.assertAlmostEqual(report.delta, 0.0, places=12)

    def test_multi_seed_variance_band(self):
        report = compare(
            ["t1", "t2"],
            {"t1": [1, 0, 1], "t2": [0, 0, 1]},
            {"t1": [1, 1, 1], "t2": [1, 0, 1]},
            n_boot=400,
            seed=2,
        )
        self.assertEqual(len(report.per_seed), 3)
        self.assertEqual(
            [row["seed_index"] for row in report.per_seed],
            [0, 1, 2],
        )
        self.assertTrue(any("positional repeats" in note for note in report.notes))
        self.assertIsNotNone(report.seed_mean_delta)
        self.assertGreaterEqual(report.seed_sd_delta, 0.0)
        self.assertAlmostEqual(report.rate_a, (2 / 3 + 1 / 3) / 2)
        self.assertAlmostEqual(report.rate_b, (1.0 + 2 / 3) / 2)
        self.assertIsNone(report.mcnemar)
        self.assertTrue(any("task-cluster" in note for note in report.notes))

    def test_empty_nonfinite_and_out_of_range_seed_scores_are_refused(self):
        bad_values = ([], [float("nan")], [float("inf")], [-0.01], [1.01])
        for value in bad_values:
            with self.subTest(value=value), self.assertRaises(EvalkitError):
                compare(["t1"], {"t1": value}, {"t1": [1]}, allow_graded=True)

    def test_duplicating_seeds_within_tasks_does_not_inflate_inference(self):
        ids = ["t1", "t2", "t3", "t4"]
        a_two = {tid: [0, 0] for tid in ids}
        b_two = {tid: [1, 1] for tid in ids}
        a_many = {tid: [0] * 100 for tid in ids}
        b_many = {tid: [1] * 100 for tid in ids}
        two = compare(ids, a_two, b_two, n_boot=500, seed=8)
        many = compare(ids, a_many, b_many, n_boot=500, seed=8)
        self.assertIsNone(two.mcnemar)
        self.assertIsNone(many.mcnemar)
        self.assertEqual(two.delta, many.delta)
        self.assertEqual(two.bootstrap.to_dict(), many.bootstrap.to_dict())

    def test_heterogeneous_seed_duplication_does_not_change_cluster_inference(self):
        ids = ["t1", "t2", "t3", "t4"]
        a = {
            "t1": [0, 1],
            "t2": [1, 0],
            "t3": [0, 0],
            "t4": [1, 1],
        }
        b = {
            "t1": [1, 1],
            "t2": [0, 0],
            "t3": [0, 1],
            "t4": [1, 0],
        }
        duplicated_a = {task_id: values * 50 for task_id, values in a.items()}
        duplicated_b = {task_id: values * 50 for task_id, values in b.items()}
        original = compare(ids, a, b, n_boot=500, seed=8)
        duplicated = compare(ids, duplicated_a, duplicated_b, n_boot=500, seed=8)
        self.assertEqual(original.delta, duplicated.delta)
        self.assertEqual(original.bootstrap.to_dict(), duplicated.bootstrap.to_dict())
        self.assertIsNone(original.mcnemar)
        self.assertIsNone(duplicated.mcnemar)

    def test_repeats_cannot_be_inflated_for_only_one_task(self):
        with self.assertRaisesRegex(
            EvalkitError,
            "every task must have the same number of seed repeats",
        ):
            compare(
                ["t1", "t2"],
                {"t1": [0, 1] * 50, "t2": [0, 1]},
                {"t1": [1, 1] * 50, "t2": [1, 0]},
                n_boot=20,
            )

    def test_positional_repeat_sd_is_explicitly_noninferential(self):
        report = compare(
            ["t1", "t2"],
            {"t1": [0, 1], "t2": [1, 1]},
            {"t1": [1, 1], "t2": [0, 1]},
            n_boot=20,
        )
        self.assertTrue(any("not an uncertainty estimate" in note for note in report.notes))
        self.assertIn("descriptive sample sd", format_markdown(report))

    def test_invalid_compare_parameters_are_refused(self):
        for alpha in (0, 1, float("nan"), "0.05", 10**1000):
            with self.subTest(alpha=alpha), self.assertRaises(EvalkitError):
                compare(["t1"], {"t1": 0}, {"t1": 1}, alpha=alpha)
        for n_boot in (0, -4, True):
            with self.subTest(n_boot=n_boot), self.assertRaises(EvalkitError):
                compare(["t1"], {"t1": 0}, {"t1": 1}, n_boot=n_boot)


class TestResultsCellReplay(unittest.TestCase):
    def test_malformed_reconstruction_inputs_are_contract_errors(self):
        for rates in (
            ("bad", 0.5), ("0.5", 0.5), (None, 0.5),
            (10**1000, 0.5), (float("nan"), 0.5), (0.5, 1.1),
        ):
            with self.subTest(rates=rates), self.assertRaises(EvalkitError):
                reconstruct_paired_from_rates(10, *rates)
        for n in (True, 0, 1.5):
            with self.subTest(n=n), self.assertRaises(EvalkitError):
                reconstruct_paired_from_rates(n, 0.5, 0.5)

    def test_published_searchqa_nano_gated_point_delta(self):
        cell = _load("results_searchqa_nano_gated.json")
        a, b = reconstruct_paired_from_rates(cell["n"], cell["baseline"], cell["after"])
        self.assertEqual(len(a), cell["n"])
        rate_a = math.fsum(a) / cell["n"]
        rate_b = math.fsum(b) / cell["n"]
        self.assertAlmostEqual(rate_a, cell["baseline"], places=3)
        self.assertAlmostEqual(rate_b, cell["after"], places=3)
        self.assertAlmostEqual(rate_b - rate_a, cell["published_delta"], places=3)


class TestCLI(unittest.TestCase):
    def test_aa_cli_exit_zero(self):
        rc = evalkit_main([
            "--manifest", os.path.join(FIXTURE_DIR, "aa_manifest.json"),
            "--a", os.path.join(FIXTURE_DIR, "aa_outcomes.json"),
            "--aa",
            "--boot", "300",
            "--json",
        ])
        self.assertEqual(rc, 0)

    def test_mismatch_cli_exit_two(self):
        with tempfile.TemporaryDirectory() as td:
            man = os.path.join(td, "m.json")
            a = os.path.join(td, "a.json")
            b = os.path.join(td, "b.json")
            with open(man, "w", encoding="utf-8") as f:
                json.dump(["t1", "t2"], f)
            with open(a, "w", encoding="utf-8") as f:
                json.dump({"t1": 1, "t2": 0}, f)
            with open(b, "w", encoding="utf-8") as f:
                json.dump({"t1": 1, "t3": 0}, f)
            rc = evalkit_main(["--manifest", man, "--a", a, "--b", b])
            self.assertEqual(rc, 2)

    def test_exactly_one_of_b_or_aa_is_required(self):
        manifest = os.path.join(FIXTURE_DIR, "aa_manifest.json")
        outcomes = os.path.join(FIXTURE_DIR, "aa_outcomes.json")
        for extra in ([], ["--b", outcomes, "--aa"], ["--b", "", "--aa"]):
            with self.subTest(extra=extra):
                stdout = io.StringIO()
                stderr = io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    rc = evalkit_main([
                        "--manifest", manifest,
                        "--a", outcomes,
                        "--json",
                        *extra,
                    ])
                self.assertEqual(rc, 2)
                self.assertEqual(stdout.getvalue(), "")
                self.assertIn("exactly one of --b or --aa", stderr.getvalue())

    def test_umbrella_cli_enforces_b_or_aa_exclusivity(self):
        manifest = os.path.join(FIXTURE_DIR, "aa_manifest.json")
        outcomes = os.path.join(FIXTURE_DIR, "aa_outcomes.json")
        base = [
            sys.executable,
            "-m",
            "skillopt_sleep",
            "evalkit",
            "--manifest",
            manifest,
            "--a",
            outcomes,
            "--json",
        ]
        for extra in ([], ["--b", outcomes, "--aa"], ["--b", "", "--aa"]):
            with self.subTest(extra=extra):
                proc = subprocess.run(
                    [*base, *extra],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(proc.stdout, "")
                self.assertIn("exactly one of --b or --aa", proc.stderr)

    def test_umbrella_cli_accepts_explicit_aa(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "skillopt_sleep",
                "evalkit",
                "--manifest",
                os.path.join(FIXTURE_DIR, "aa_manifest.json"),
                "--a",
                os.path.join(FIXTURE_DIR, "aa_outcomes.json"),
                "--aa",
                "--boot",
                "20",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["delta"], 0.0)

    def test_duplicate_json_object_keys_are_refused(self):
        with tempfile.TemporaryDirectory() as td:
            man = os.path.join(td, "m.json")
            a = os.path.join(td, "a.json")
            b = os.path.join(td, "b.json")
            with open(man, "w", encoding="utf-8") as handle:
                handle.write('["t1"]')
            with open(a, "w", encoding="utf-8") as handle:
                handle.write('{"t1": 0, "t1": 1}')
            with open(b, "w", encoding="utf-8") as handle:
                handle.write('{"t1": 1}')
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = evalkit_main([
                    "--manifest", man,
                    "--a", a,
                    "--b", b,
                    "--json",
                ])
            self.assertEqual(rc, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("duplicate JSON object key: 't1'", stderr.getvalue())

    def test_task_named_outcomes_is_not_mistaken_for_wrapper(self):
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for name, content in (
                ("m.json", '["outcomes"]'),
                ("a.json", '{"outcomes": 0}'),
                ("b.json", '{"outcomes": 1}'),
            ):
                path = os.path.join(td, name)
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(content)
                paths.append(path)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = evalkit_main([
                    "--manifest", paths[0],
                    "--a", paths[1],
                    "--b", paths[2],
                    "--boot", "20",
                    "--json",
                ])
            self.assertEqual(rc, 0, stderr.getvalue())
            self.assertEqual(json.loads(stdout.getvalue())["delta"], 1.0)

    def test_task_named_outcomes_accepts_seeded_object_values(self):
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for name, content in (
                ("m.json", '["outcomes"]'),
                ("a.json", '{"outcomes": {"seeds": [0, 1]}}'),
                ("b.json", '{"outcomes": {"seeds": [1, 1]}}'),
            ):
                path = os.path.join(td, name)
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(content)
                paths.append(path)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = evalkit_main([
                    "--manifest", paths[0], "--a", paths[1], "--b", paths[2],
                    "--boot", "20", "--json",
                ])
            self.assertEqual(rc, 0, stderr.getvalue())
            self.assertEqual(json.loads(stdout.getvalue())["delta"], 0.5)

    def test_malformed_json_and_shapes_are_clean_contract_errors(self):
        cases = (
            ("{", '{"t1": 1}', '{"t1": 1}'),
            ('{"ids": "t1"}', '{"t1": 1}', '{"t1": 1}'),
            ('["t1"]', '{"outcomes": []}', '{"t1": 1}'),
        )
        for manifest_text, a_text, b_text in cases:
            with self.subTest(manifest=manifest_text), tempfile.TemporaryDirectory() as td:
                paths = []
                for name, content in (("m.json", manifest_text), ("a.json", a_text), ("b.json", b_text)):
                    path = os.path.join(td, name)
                    with open(path, "w", encoding="utf-8") as handle:
                        handle.write(content)
                    paths.append(path)
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    rc = evalkit_main(["--manifest", paths[0], "--a", paths[1], "--b", paths[2]])
                self.assertEqual(rc, 2)
                self.assertTrue(stderr.getvalue().startswith("ERR_EVALKIT "))
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_module_entrypoint(self):
        proc = subprocess.run(
            [
                sys.executable, "-m", "skillopt_sleep.evalkit",
                "--manifest", os.path.join(FIXTURE_DIR, "aa_manifest.json"),
                "--a", os.path.join(FIXTURE_DIR, "aa_outcomes.json"),
                "--aa",
                "--boot", "200",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("delta (B-A): +0.000000", proc.stdout)

    def test_nonfinite_input_is_refused_without_nonstandard_json_output(self):
        with tempfile.TemporaryDirectory() as td:
            man = os.path.join(td, "m.json")
            a = os.path.join(td, "a.json")
            b = os.path.join(td, "b.json")
            with open(man, "w", encoding="utf-8") as f:
                json.dump(["t1"], f)
            with open(a, "w", encoding="utf-8") as f:
                f.write('{"t1": NaN}')
            with open(b, "w", encoding="utf-8") as f:
                json.dump({"t1": 1}, f)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = evalkit_main([
                    "--manifest", man, "--a", a, "--b", b, "--json",
                ])
            self.assertEqual(rc, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("non-standard JSON constant", stderr.getvalue())
            self.assertNotIn("NaN", stdout.getvalue())

    def test_nonstandard_json_constants_are_rejected_even_in_ignored_metadata(self):
        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant), tempfile.TemporaryDirectory() as td:
                man = os.path.join(td, "m.json")
                a = os.path.join(td, "a.json")
                b = os.path.join(td, "b.json")
                with open(man, "w", encoding="utf-8") as handle:
                    handle.write('{"ids": ["t1"], "ignored": ' + constant + "}")
                with open(a, "w", encoding="utf-8") as handle:
                    handle.write('{"t1": 0}')
                with open(b, "w", encoding="utf-8") as handle:
                    handle.write('{"t1": 1}')
                stdout = io.StringIO()
                stderr = io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    rc = evalkit_main([
                        "--manifest", man, "--a", a, "--b", b, "--json",
                    ])
                self.assertEqual(rc, 2)
                self.assertEqual(stdout.getvalue(), "")
                self.assertIn("non-standard JSON constant", stderr.getvalue())

    def test_malformed_manifest_ids_are_rejected_by_the_cli(self):
        invalid_ids = (1, {}, [], None, True, "", "   ")
        for task_id in invalid_ids:
            with self.subTest(task_id=task_id), tempfile.TemporaryDirectory() as td:
                man = os.path.join(td, "m.json")
                a = os.path.join(td, "a.json")
                with open(man, "w", encoding="utf-8") as handle:
                    json.dump({"tasks": [{"id": task_id}]}, handle)
                with open(a, "w", encoding="utf-8") as handle:
                    json.dump({"t1": 0}, handle)
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    rc = evalkit_main(["--manifest", man, "--a", a, "--aa"])
                self.assertEqual(rc, 2)
                self.assertIn("non-empty JSON strings", stderr.getvalue())

    def test_conflicting_manifest_forms_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            man = os.path.join(td, "m.json")
            a = os.path.join(td, "a.json")
            with open(man, "w", encoding="utf-8") as handle:
                json.dump({"ids": ["t1"], "tasks": [{"id": "t1"}]}, handle)
            with open(a, "w", encoding="utf-8") as handle:
                json.dump({"t1": 0}, handle)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                rc = evalkit_main(["--manifest", man, "--a", a, "--aa"])
            self.assertEqual(rc, 2)
            self.assertIn("exactly one of ids, tasks, or outcomes", stderr.getvalue())

    def test_input_errors_do_not_disclose_paths(self):
        with tempfile.TemporaryDirectory() as td:
            secret_name = "secret-customer-path.json"
            path = os.path.join(td, secret_name)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                rc = evalkit_main(["--manifest", path, "--a", path, "--aa"])
            self.assertEqual(rc, 2)
            self.assertNotIn(td, stderr.getvalue())
            self.assertNotIn(secret_name, stderr.getvalue())
            self.assertIn("--manifest", stderr.getvalue())

    def test_confidence_label_is_not_truncated_by_float_roundoff(self):
        report = compare_aa(["t1"], {"t1": 1}, alpha=0.34, n_boot=20)
        rendered = format_markdown(report)
        self.assertIn("bootstrap 66% CI", rendered)
        self.assertNotIn("bootstrap 65% CI", rendered)


if __name__ == "__main__":
    unittest.main()
