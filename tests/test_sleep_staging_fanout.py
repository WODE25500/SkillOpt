"""Tests for per-skill staging fan-out (issue #120).

Pure-stdlib (unittest), hermetic (tmpdir only), no API key, no network.
Run:  python -m pytest tests/test_sleep_staging_fanout.py
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from skillopt_sleep.staging import (
    SkillProposal,
    StagingError,
    latest_staging,
    new_staging_dir,
    proposal_filename,
    skill_proposal_rows,
    write_skill_proposals,
    write_staging,
)
from skillopt_sleep.types import SleepReport


def _proposal(name="example-skill", body="# example\n", live=None, root="/tmp/live"):
    if live is None:
        live = os.path.join(root, name, "SKILL.md")
    return SkillProposal(name, body, live)


def _canonical(path):
    return os.path.realpath(os.path.abspath(os.path.normpath(path)))


def _report():
    return SleepReport(night=1, project="/repo/example", accepted=True,
                       gate_action="accept_new_best")


class TestSkillProposalRows(unittest.TestCase):
    def test_one_row_per_skill_in_order(self):
        rows = skill_proposal_rows([_proposal("alpha"), _proposal("beta")])
        self.assertEqual([r["skill_name"] for r in rows], ["alpha", "beta"])
        self.assertEqual([r["proposed_file"] for r in rows],
                         ["proposed_SKILL.alpha.md", "proposed_SKILL.beta.md"])
        self.assertEqual(rows[0]["live_skill_path"],
                         _canonical("/tmp/live/alpha/SKILL.md"))
        self.assertEqual(
            rows[0]["sha256"],
            hashlib.sha256(b"# example\n").hexdigest(),
        )

    def test_filenames_are_unique_per_skill(self):
        self.assertNotEqual(proposal_filename("alpha"), proposal_filename("beta"))

    def test_duplicate_skill_name_is_refused(self):
        with self.assertRaises(StagingError):
            skill_proposal_rows([_proposal("alpha"),
                                 _proposal("alpha", live="/tmp/other/SKILL.md")])

    def test_names_differing_only_by_case_are_refused(self):
        # Skill names are case-sensitive, but every proposal lands in one
        # staging directory — and on macOS and Windows that directory is
        # case-insensitive. Before this guard, staging "Research" then
        # "research" produced two manifest rows but one file on disk, named
        # after the first skill and containing the second one's document.
        with self.assertRaises(StagingError):
            skill_proposal_rows([_proposal("Research"),
                                 _proposal("research", live="/tmp/other/SKILL.md")])

    def test_case_differing_proposals_never_lose_a_staged_file(self):
        # Guards the guard: if the refusal above is ever relaxed, this asserts
        # the actual filesystem outcome rather than the intent.
        with tempfile.TemporaryDirectory() as out:
            try:
                rows = write_skill_proposals(
                    out,
                    [_proposal("Research"),
                     _proposal("research", live="/tmp/other/SKILL.md")],
                )
            except StagingError:
                return  # refused up front, which is the desired behaviour
            staged = [f for f in os.listdir(out) if f.startswith("proposed_")]
            self.assertEqual(len(staged), len(rows),
                             "a manifest row exists whose staged file was overwritten")

    def test_live_paths_differing_only_by_case_are_refused(self):
        # os.path.normcase would not catch this: it only folds case on Windows,
        # so it is a no-op on the macOS filesystem where /x/A.md and /x/a.md
        # are nevertheless the same file.
        with self.assertRaises(StagingError):
            skill_proposal_rows([_proposal("alpha", live="/tmp/live/A.md"),
                                 _proposal("beta", live="/tmp/live/a.md")])

    def test_unicode_equivalent_staged_names_are_refused(self):
        # HFS+/APFS commonly normalise filenames. NFC ``café`` and the
        # decomposed NFD spelling must not produce two manifest rows for one
        # physical staged file.
        with self.assertRaises(StagingError):
            skill_proposal_rows([
                _proposal("café", live="/tmp/live/cafe-a/SKILL.md"),
                _proposal("cafe\u0301", live="/tmp/live/cafe-b/SKILL.md"),
            ])

    def test_a_generator_of_proposals_still_writes_every_file(self):
        # The annotation says Sequence but nothing enforces it. Validation used
        # to drain a generator, leaving the write loop empty and returning a
        # full set of manifest rows for files that were never created.
        with tempfile.TemporaryDirectory() as out:
            gen = (_proposal(n, live=f"/tmp/live/{n}/SKILL.md") for n in ("alpha", "beta"))
            rows = write_skill_proposals(out, gen)
            staged = [f for f in os.listdir(out) if f.startswith("proposed_")]
            self.assertEqual(len(staged), len(rows))
            self.assertEqual(len(rows), 2)

    def test_names_windows_cannot_store_are_refused_cleanly(self):
        # These reach the filesystem as a filename. Without an explicit guard
        # they raise OSError from inside the write instead of a StagingError
        # naming the offending skill.
        for bad in ["a:b", "a*b", "a?b", 'a"b', "a<b", "a>b", "a|b", "trailing."]:
            with self.assertRaises(StagingError, msg=bad):
                skill_proposal_rows([_proposal(bad)])

    def test_absolute_paths_needing_normalisation_are_accepted(self):
        # Requiring the input to already equal normpath() rejected safe paths:
        # duplicate separators everywhere, and every forward-slash absolute
        # path on Windows. Normalising first keeps the traversal guard.
        rows = skill_proposal_rows([_proposal("alpha", live="/tmp/live//alpha/SKILL.md")])
        self.assertEqual(rows[0]["live_skill_path"], _canonical("/tmp/live/alpha/SKILL.md"))

    def test_current_directory_segments_are_normalised_not_refused(self):
        rows = skill_proposal_rows([
            _proposal("alpha", live="/tmp/live/./alpha/SKILL.md")
        ])
        self.assertEqual(rows[0]["live_skill_path"],
                         _canonical("/tmp/live/alpha/SKILL.md"))

    def test_two_skills_targeting_one_file_are_refused(self):
        shared = "/tmp/live/shared/SKILL.md"
        with self.assertRaises(StagingError):
            skill_proposal_rows([_proposal("alpha", live=shared),
                                 _proposal("beta", live=shared)])

    def test_unsafe_skill_names_are_refused(self):
        for bad in ["", "  ", ".", "..", "../escape", "a/b", "a\\b", "/abs",
                    "~home", "bad\nname"]:
            with self.assertRaises(StagingError, msg=bad):
                skill_proposal_rows([_proposal(bad)])

    def test_unsafe_live_paths_are_refused(self):
        for bad in ["", "relative/SKILL.md", "~/skills/a/SKILL.md",
                    "/tmp/live/../../etc/SKILL.md", "/tmp/live/a/SKILL.txt",
                    "/tmp/live/a/skill\x00/SKILL.md", "/tmp/live/a\n/SKILL.md"]:
            with self.assertRaises(StagingError, msg=bad):
                skill_proposal_rows([_proposal("alpha", live=bad)])


class TestWriteSkillProposals(unittest.TestCase):
    def test_writes_one_file_per_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = write_skill_proposals(tmp, [
                _proposal("alpha", "# alpha\n"),
                _proposal("beta", "# beta\n"),
            ])
            self.assertEqual(sorted(os.listdir(tmp)),
                             ["proposed_SKILL.alpha.md", "proposed_SKILL.beta.md"])
            with open(os.path.join(tmp, rows[0]["proposed_file"]), encoding="utf-8") as f:
                self.assertEqual(f.read(), "# alpha\n")

    def test_empty_proposal_body_is_refused_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(StagingError, "is empty"):
                write_skill_proposals(tmp, [_proposal("alpha", "   \n")])
            self.assertEqual(os.listdir(tmp), [])

    def test_no_proposals_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(write_skill_proposals(tmp, []), [])
            self.assertEqual(os.listdir(tmp), [])

    def test_rejected_fan_out_leaves_no_partial_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(StagingError):
                write_skill_proposals(tmp, [_proposal("alpha"), _proposal("../escape")])
            self.assertEqual(os.listdir(tmp), [])

    def test_non_text_proposal_leaves_no_partial_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(StagingError, "must be text"):
                write_skill_proposals(
                    tmp,
                    [_proposal("alpha", "# alpha\n"), _proposal("beta", None)],
                )
            self.assertEqual(os.listdir(tmp), [])

    def test_writes_leave_no_temporary_files_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_skill_proposals(tmp, [_proposal("alpha")])
            self.assertEqual([n for n in os.listdir(tmp) if n.startswith(".tmp-")], [])

    def test_rewrite_replaces_content_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_skill_proposals(tmp, [_proposal("alpha", "# first\n")])
            write_skill_proposals(tmp, [_proposal("alpha", "# second\n")])
            path = os.path.join(tmp, proposal_filename("alpha"))
            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), "# second\n")
            self.assertEqual(sorted(os.listdir(tmp)), [proposal_filename("alpha")])


class TestWriteStagingCompatibility(unittest.TestCase):
    def _manifest(self, out):
        with open(os.path.join(out, "manifest.json"), encoding="utf-8") as f:
            return json.load(f)

    def test_legacy_layout_when_multi_skill_is_unused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = write_staging(
                tmp, report=_report(), proposed_skill="# skill\n",
                proposed_memory="# memory\n",
                live_skill_path=os.path.join(tmp, "live", "SKILL.md"),
                live_memory_path=os.path.join(tmp, "live", "CLAUDE.md"),
                report_md="# report\n",
            )
            self.assertEqual(
                sorted(os.listdir(out)),
                ["manifest.json", "proposed_CLAUDE.md", "proposed_SKILL.md",
                 "report.json", "report.md"],
            )
            manifest = self._manifest(out)
            self.assertNotIn("skills", manifest)
            self.assertEqual(manifest["schema"], "skillopt-sleep-staging")
            self.assertEqual(manifest["schema_version"], 2)
            self.assertFalse(manifest["has_skill"])
            self.assertFalse(manifest["has_memory"])
            self.assertTrue(manifest["has_managed_skill"])
            self.assertTrue(manifest["has_managed_memory"])

    def test_v020_compatibility_flags_select_no_unpinned_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = write_staging(
                tmp, report=_report(), proposed_skill="# skill\n",
                proposed_memory="# memory\n",
                live_skill_path=os.path.join(tmp, "live", "SKILL.md"),
                live_memory_path=os.path.join(tmp, "live", "CLAUDE.md"),
                report_md="# report\n",
            )
            manifest = self._manifest(out)
            # This is the complete v0.2.0 mutation decision: it trusted only
            # these top-level booleans, without consulting integrity pins.
            selected = []
            if manifest.get("has_skill"):
                selected.append("proposed_SKILL.md")
            if manifest.get("has_memory"):
                selected.append("proposed_CLAUDE.md")
            self.assertEqual(selected, [])
            self.assertIn("skill", manifest["legacy"])
            self.assertIn("memory", manifest["legacy"])

    def test_unknown_manifest_schema_version_is_refused(self):
        from skillopt_sleep.staging import staged_skills

        with tempfile.TemporaryDirectory() as tmp:
            out = write_staging(
                tmp, report=_report(), proposed_skill=None, proposed_memory=None,
                live_skill_path=os.path.join(tmp, "live", "SKILL.md"),
                live_memory_path=os.path.join(tmp, "live", "CLAUDE.md"),
                report_md="# report\n",
                skill_proposals=[_proposal("alpha", root=os.path.join(tmp, "live"))],
            )
            path = os.path.join(out, "manifest.json")
            manifest = self._manifest(out)
            manifest["schema_version"] = 999
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            with self.assertRaisesRegex(StagingError, "unsupported schema"):
                staged_skills(out)

    def test_fan_out_adds_files_and_manifest_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            live_root = os.path.join(tmp, "live")
            out = write_staging(
                tmp, report=_report(), proposed_skill=None, proposed_memory=None,
                live_skill_path=os.path.join(live_root, "SKILL.md"),
                live_memory_path=os.path.join(live_root, "CLAUDE.md"),
                report_md="# report\n",
                skill_proposals=[
                    _proposal("alpha", "# alpha\n", root=live_root),
                    _proposal("beta", "# beta\n", root=live_root),
                ],
            )
            self.assertEqual(
                sorted(os.listdir(out)),
                ["manifest.json", "proposed_SKILL.alpha.md", "proposed_SKILL.beta.md",
                 "report.json", "report.md"],
            )
            rows = self._manifest(out)["skills"]
            self.assertEqual([r["skill_name"] for r in rows], ["alpha", "beta"])
            self.assertEqual(rows[1]["live_skill_path"],
                             _canonical(os.path.join(live_root, "beta", "SKILL.md")))
            self.assertEqual(
                rows[0]["sha256"],
                hashlib.sha256(b"# alpha\n").hexdigest(),
            )
            self.assertEqual(rows[0]["live_sha256"], "")
            self.assertEqual(
                rows[0]["live_realpath"],
                os.path.realpath(os.path.join(live_root, "alpha", "SKILL.md")),
            )

    def test_two_same_second_nights_get_distinct_reserved_directories(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "skillopt_sleep.staging._ts_dir", return_value="20260815-010203"
        ):
            first = write_staging(
                tmp, report=_report(), proposed_skill="# first\n",
                proposed_memory=None,
                live_skill_path=os.path.join(tmp, "live", "SKILL.md"),
                live_memory_path=os.path.join(tmp, "live", "CLAUDE.md"),
                report_md="# first report\n",
            )
            second = write_staging(
                tmp, report=_report(), proposed_skill="# second\n",
                proposed_memory=None,
                live_skill_path=os.path.join(tmp, "live", "SKILL.md"),
                live_memory_path=os.path.join(tmp, "live", "CLAUDE.md"),
                report_md="# second report\n",
            )
            self.assertNotEqual(first, second)
            self.assertEqual(
                sorted((os.path.basename(first), os.path.basename(second))),
                ["20260815-010203", "20260815-010203-2"],
            )
            with open(os.path.join(first, "report.md"), encoding="utf-8") as f:
                self.assertEqual(f.read(), "# first report\n")
            with open(os.path.join(second, "report.md"), encoding="utf-8") as f:
                self.assertEqual(f.read(), "# second report\n")

    def test_concurrent_staging_reservations_are_unique_and_exist(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "skillopt_sleep.staging._ts_dir", return_value="20260815-010203"
        ):
            with ThreadPoolExecutor(max_workers=8) as pool:
                paths = list(pool.map(lambda _i: new_staging_dir(tmp), range(16)))
            self.assertEqual(len(set(paths)), len(paths))
            self.assertTrue(all(os.path.isdir(path) for path in paths))

    def test_concurrent_publications_leave_one_valid_atomic_latest_pointer(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "skillopt_sleep.staging._ts_dir", return_value="20260815-010203"
        ):
            def publish(index):
                return write_staging(
                    tmp, report=_report(), proposed_skill=f"# skill {index}\n",
                    proposed_memory=None,
                    live_skill_path=os.path.join(tmp, "live", "SKILL.md"),
                    live_memory_path=os.path.join(tmp, "live", "CLAUDE.md"),
                    report_md=f"# report {index}\n",
                )

            with ThreadPoolExecutor(max_workers=8) as pool:
                paths = list(pool.map(publish, range(16)))
            latest = latest_staging(tmp)
            self.assertIn(latest, paths)
            with open(
                os.path.join(tmp, ".skillopt-sleep", "staging", ".latest"),
                encoding="utf-8",
            ) as handle:
                self.assertEqual(handle.read().strip(), os.path.basename(latest))

    def test_latest_pointer_retries_a_transient_windows_sharing_violation(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, ".skillopt-sleep", "staging")
            night = os.path.join(root, "20260815-010203")
            os.makedirs(night)
            with open(os.path.join(night, "manifest.json"), "w", encoding="utf-8") as f:
                f.write("{}")
            real_replace = os.replace
            calls = []

            def transient_replace(source, destination):
                calls.append((source, destination))
                if len(calls) == 1:
                    raise PermissionError("simulated Windows sharing violation")
                return real_replace(source, destination)

            with mock.patch.object(staging_mod.os, "name", "nt"), mock.patch.object(
                staging_mod.os, "replace", side_effect=transient_replace
            ), mock.patch.object(staging_mod.time, "sleep") as sleep:
                staging_mod._publish_latest(root, night)

            self.assertEqual(len(calls), 2)
            sleep.assert_called_once_with(0.005)
            with open(os.path.join(root, ".latest"), encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "20260815-010203\n")

    def test_generic_atomic_write_never_retries_permission_error(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            destination = os.path.join(tmp, "live.md")
            with open(destination, "wb") as handle:
                handle.write(b"original")
            with mock.patch.object(staging_mod.os, "name", "nt"), mock.patch.object(
                staging_mod.os,
                "replace",
                side_effect=PermissionError("live file is busy"),
            ) as replace, self.assertRaises(PermissionError):
                staging_mod._write_atomic_bytes(destination, b"proposal")

            replace.assert_called_once()
            with open(destination, "rb") as handle:
                self.assertEqual(handle.read(), b"original")
            self.assertFalse(any(name.startswith(".tmp-") for name in os.listdir(tmp)))

    def test_latest_pointer_persistent_error_preserves_destination_and_cleans_temp(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, ".skillopt-sleep", "staging")
            night = os.path.join(root, "20260815-010203")
            os.makedirs(night)
            with open(os.path.join(night, "manifest.json"), "w", encoding="utf-8") as f:
                f.write("{}")
            pointer = os.path.join(root, ".latest")
            with open(pointer, "w", encoding="utf-8") as handle:
                handle.write("20260814-010203\n")

            with mock.patch.object(staging_mod.os, "name", "nt"), mock.patch.object(
                staging_mod.os,
                "replace",
                side_effect=PermissionError("pointer remains busy"),
            ) as replace, mock.patch.object(staging_mod.time, "sleep"), self.assertRaises(
                PermissionError
            ):
                staging_mod._publish_latest(root, night)

            self.assertEqual(replace.call_count, 21)
            with open(pointer, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "20260814-010203\n")
            self.assertFalse(any(name.startswith(".tmp-") for name in os.listdir(root)))

    def test_staging_descendant_accepts_root_alias_but_rejects_child_symlink(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            real_root = os.path.join(tmp, "real-staging")
            os.makedirs(real_root)
            alias_root = os.path.join(tmp, "staging-alias")
            outside = os.path.join(tmp, "outside")
            os.makedirs(outside)
            try:
                os.symlink(real_root, alias_root, target_is_directory=True)
                os.symlink(
                    outside,
                    os.path.join(real_root, "child-alias"),
                    target_is_directory=True,
                )
            except OSError:
                self.skipTest("directory symlinks unavailable")

            ordinary_lexical = os.path.join(real_root, "backup.md")
            with open(ordinary_lexical, "w", encoding="utf-8") as handle:
                handle.write("backup")
            # WAL/manifest paths are canonical, while callers can still supply
            # the same staging root through a lexical alias.
            ordinary = os.path.realpath(ordinary_lexical)
            escaped = os.path.join(alias_root, "child-alias", "outside.md")
            with open(os.path.join(outside, "outside.md"), "w", encoding="utf-8") as handle:
                handle.write("outside")

            self.assertTrue(
                staging_mod._existing_path_is_canonical_staging_descendant(
                    ordinary,
                    alias_root,
                )
            )
            self.assertFalse(
                staging_mod._existing_path_is_canonical_staging_descendant(
                    escaped,
                    alias_root,
                )
            )

    def test_latest_ignores_a_symlinked_night(self):
        with tempfile.TemporaryDirectory() as tmp:
            real_night = write_staging(
                tmp, report=_report(), proposed_skill="# real\n",
                proposed_memory=None,
                live_skill_path=os.path.join(tmp, "live", "SKILL.md"),
                live_memory_path=os.path.join(tmp, "live", "CLAUDE.md"),
                report_md="# real report\n",
            )
            outside = os.path.join(tmp, "outside-night")
            os.makedirs(outside)
            with open(
                os.path.join(outside, "manifest.json"), "w", encoding="utf-8"
            ) as handle:
                handle.write("{}")
            alias = os.path.join(
                os.path.dirname(real_night), "99991231-235959"
            )
            try:
                os.symlink(outside, alias)
            except OSError:
                self.skipTest("symlinks unavailable")

            self.assertEqual(latest_staging(tmp), real_night)

    def test_explicit_symlink_staging_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            outside = os.path.join(tmp, "outside-night")
            os.makedirs(outside)
            sentinel = os.path.join(outside, "keep.txt")
            with open(sentinel, "w", encoding="utf-8") as handle:
                handle.write("untouched\n")
            alias = os.path.join(tmp, "staging-alias")
            try:
                os.symlink(outside, alias)
            except OSError:
                self.skipTest("symlinks unavailable")

            with self.assertRaisesRegex(StagingError, "staging directory is unsafe"):
                write_staging(
                    tmp, report=_report(), proposed_skill="# proposal\n",
                    proposed_memory=None,
                    live_skill_path=os.path.join(tmp, "live", "SKILL.md"),
                    live_memory_path=os.path.join(tmp, "live", "CLAUDE.md"),
                    report_md="# report\n",
                    out_dir=alias,
                )
            with open(sentinel, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "untouched\n")
            self.assertEqual(sorted(os.listdir(outside)), ["keep.txt"])

    def test_publication_pointer_orders_equal_mtimes_across_clock_rollback(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "skillopt_sleep.staging._ts_dir",
            side_effect=["20261101-015959", "20261101-010001"],
        ):
            before_rollback = write_staging(
                tmp, report=_report(), proposed_skill="# before\n",
                proposed_memory=None,
                live_skill_path=os.path.join(tmp, "live", "SKILL.md"),
                live_memory_path=os.path.join(tmp, "live", "CLAUDE.md"),
                report_md="# before rollback\n",
            )
            after_rollback = write_staging(
                tmp, report=_report(), proposed_skill="# after\n",
                proposed_memory=None,
                live_skill_path=os.path.join(tmp, "live", "SKILL.md"),
                live_memory_path=os.path.join(tmp, "live", "CLAUDE.md"),
                report_md="# after rollback\n",
            )
            os.utime(
                os.path.join(before_rollback, "manifest.json"),
                ns=(1_000_000_000, 1_000_000_000),
            )
            os.utime(
                os.path.join(after_rollback, "manifest.json"),
                ns=(1_000_000_000, 1_000_000_000),
            )

            self.assertEqual(latest_staging(tmp), after_rollback)

    def test_tampered_symlink_pointer_is_ignored_for_contained_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = write_staging(
                tmp, report=_report(), proposed_skill="# first\n",
                proposed_memory=None,
                live_skill_path=os.path.join(tmp, "live", "SKILL.md"),
                live_memory_path=os.path.join(tmp, "live", "CLAUDE.md"),
                report_md="# first\n",
            )
            pointer = os.path.join(
                tmp, ".skillopt-sleep", "staging", ".latest"
            )
            outside = os.path.join(tmp, "outside-pointer")
            with open(outside, "w", encoding="utf-8") as handle:
                handle.write("../../outside-night\n")
            os.unlink(pointer)
            try:
                os.symlink(outside, pointer)
            except OSError:
                self.skipTest("symlinks unavailable")
            self.assertEqual(latest_staging(tmp), first)

    def test_failed_artifact_write_publishes_no_partial_night(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            real_write = staging_mod._write_atomic

            def fail_second_proposal(path, text, *, create_parents=True):
                if path.endswith("proposed_SKILL.beta.md"):
                    raise OSError("disk full")
                return real_write(path, text, create_parents=create_parents)

            with mock.patch.object(
                staging_mod, "_write_atomic", side_effect=fail_second_proposal
            ), self.assertRaises(OSError):
                write_staging(
                    tmp, report=_report(), proposed_skill=None,
                    proposed_memory=None,
                    live_skill_path=os.path.join(tmp, "live", "SKILL.md"),
                    live_memory_path=os.path.join(tmp, "live", "CLAUDE.md"),
                    report_md="# report\n",
                    skill_proposals=[
                        _proposal("alpha", "# alpha\n", root=os.path.join(tmp, "live")),
                        _proposal("beta", "# beta\n", root=os.path.join(tmp, "live")),
                    ],
                )

            self.assertIsNone(latest_staging(tmp))
            for root, _dirs, files in os.walk(tmp):
                self.assertNotIn("manifest.json", files, root)
                self.assertFalse(
                    any(name.startswith("proposed_SKILL") for name in files),
                    root,
                )

    def test_artifact_rollback_attempts_every_restore_and_preserves_primary(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "first.md")
            second = os.path.join(tmp, "second.md")
            for path, body in ((first, "old first"), (second, "old second")):
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(body)
            restore_attempts = []
            real_restore = staging_mod._write_atomic_bytes

            def fail_publish(path, text, *, create_parents=True):
                if path == second:
                    raise OSError("primary publish failure")

            def restore(path, data, *, create_parents=True, mode=None):
                restore_attempts.append(path)
                if path == first:
                    raise OSError("first restore failed")
                return real_restore(
                    path,
                    data,
                    create_parents=create_parents,
                    mode=mode,
                )

            with mock.patch.object(
                staging_mod, "_write_atomic", side_effect=fail_publish
            ), mock.patch.object(
                staging_mod, "_write_atomic_bytes", side_effect=restore
            ):
                with self.assertRaises(staging_mod.StagingRecoveryError) as caught:
                    staging_mod._write_artifact_batch([
                        (first, "new first"),
                        (second, "new second"),
                    ])
            self.assertIsInstance(caught.exception.primary, OSError)
            self.assertIn("primary publish failure", str(caught.exception.primary))
            self.assertEqual(restore_attempts, [second, first])

    def test_post_commit_artifact_error_rolls_back_the_whole_night(self):
        from skillopt_sleep import staging as staging_mod

        with tempfile.TemporaryDirectory() as tmp:
            real_write = staging_mod._write_atomic

            def commit_then_fail(path, text, *, create_parents=True):
                result = real_write(path, text, create_parents=create_parents)
                if path.endswith("proposed_SKILL.beta.md"):
                    raise OSError("late close failure")
                return result

            with mock.patch.object(
                staging_mod, "_write_atomic", side_effect=commit_then_fail
            ), self.assertRaisesRegex(OSError, "late close failure"):
                write_staging(
                    tmp, report=_report(), proposed_skill=None,
                    proposed_memory=None,
                    live_skill_path=os.path.join(tmp, "live", "SKILL.md"),
                    live_memory_path=os.path.join(tmp, "live", "CLAUDE.md"),
                    report_md="# report\n",
                    skill_proposals=[
                        _proposal("alpha", "# alpha\n", root=os.path.join(tmp, "live")),
                        _proposal("beta", "# beta\n", root=os.path.join(tmp, "live")),
                    ],
                )

            self.assertIsNone(latest_staging(tmp))
            for root, _dirs, files in os.walk(tmp):
                self.assertFalse(files, f"partial staging artifacts remain in {root}")

    def test_exact_cycle_baseline_change_before_staging_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            live = os.path.join(tmp, "live", "alpha", "SKILL.md")
            os.makedirs(os.path.dirname(live), exist_ok=True)
            with open(live, "w", encoding="utf-8") as handle:
                handle.write("# alpha baseline v1\n")
            proposal = SkillProposal(
                "alpha",
                "# alpha proposal derived from v1\n",
                live,
                live_sha256=hashlib.sha256(b"# alpha baseline v1\n").hexdigest(),
                live_realpath=os.path.realpath(live),
            )
            with open(live, "w", encoding="utf-8") as handle:
                handle.write("# alpha human v2\n")

            with self.assertRaisesRegex(StagingError, "changed during consolidation"):
                write_staging(
                    tmp, report=_report(), proposed_skill=None,
                    proposed_memory=None,
                    live_skill_path=live,
                    live_memory_path=os.path.join(tmp, "live", "CLAUDE.md"),
                    report_md="# report\n",
                    skill_proposals=[proposal],
                )
            self.assertIsNone(latest_staging(tmp))

    def test_unsafe_fan_out_writes_no_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(StagingError):
                write_staging(
                    tmp, report=_report(), proposed_skill=None, proposed_memory=None,
                    live_skill_path=os.path.join(tmp, "live", "SKILL.md"),
                    live_memory_path=os.path.join(tmp, "live", "CLAUDE.md"),
                    report_md="# report\n",
                    skill_proposals=[_proposal("alpha", live="relative/SKILL.md")],
                )
            for root, _dirs, files in os.walk(tmp):
                self.assertNotIn("manifest.json", files, root)


if __name__ == "__main__":
    unittest.main()
