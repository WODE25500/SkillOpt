"""Tests for workspace-prep symlink fail-closed behavior + Copilot trace redaction."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from skillopt.model.codex_harness import (
    _is_symlink_privilege_error,
    _redact_copilot_trace,
    prepare_workspace,
)


def _mk_src(tmp_path, name="srcdir", content="source-content") -> Path:
    src = tmp_path / name
    src.mkdir()
    (src / "data.txt").write_text(content, encoding="utf-8")
    return src


def _privilege_error() -> OSError:
    e = OSError()
    e.winerror = 1314  # ERROR_PRIVILEGE_NOT_HELD
    return e


def test_is_symlink_privilege_error():
    assert _is_symlink_privilege_error(_privilege_error()) is True
    e = OSError()
    e.errno = errno.EPERM
    assert _is_symlink_privilege_error(e) is True
    e2 = OSError()
    e2.errno = errno.EACCES
    assert _is_symlink_privilege_error(e2) is False
    assert _is_symlink_privilege_error(OSError()) is False


def test_symlink_privilege_fallback_copies_and_leaves_source(monkeypatch, tmp_path):
    src = _mk_src(tmp_path)
    work = tmp_path / "work"
    work.mkdir()

    def fake_symlink(a, b, **kw):
        raise _privilege_error()

    monkeypatch.setattr("skillopt.model.codex_harness.os.symlink", fake_symlink)
    prepare_workspace(work_dir=str(work), skill_md="x", link_dirs=[(str(src), "docs/data")])

    # Source is not modified.
    assert (src / "data.txt").read_text(encoding="utf-8") == "source-content"
    # The fallback copied into the private work dir.
    assert (work / "docs" / "data" / "data.txt").read_text(encoding="utf-8") == "source-content"


def test_existing_destination_fails_closed(monkeypatch, tmp_path):
    src = _mk_src(tmp_path)
    work = tmp_path / "work"
    work.mkdir()

    # extra_files creates the destination directory before link_dirs runs.
    with pytest.raises(FileExistsError):
        prepare_workspace(
            work_dir=str(work),
            skill_md="x",
            extra_files={"docs/data/file.txt": "stale"},
            link_dirs=[(str(src), "docs/data")],
        )
    # extra_files content was not overwritten by a merge.
    assert (work / "docs" / "data" / "file.txt").read_text(encoding="utf-8") == "stale"


def test_duplicate_destination_fails_closed(monkeypatch, tmp_path):
    src_a = _mk_src(tmp_path, "A")
    src_b = _mk_src(tmp_path, "B", content="B-content")
    work = tmp_path / "work"
    work.mkdir()

    # Force the first to succeed via a dir symlink stub, then assert the second
    # duplicate destination is refused rather than merged.
    def fake_symlink(a, b, **kw):
        Path(b).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("skillopt.model.codex_harness.os.symlink", fake_symlink)
    with pytest.raises(FileExistsError):
        prepare_workspace(
            work_dir=str(work),
            skill_md="x",
            link_dirs=[(str(src_a), "shared"), (str(src_b), "shared")],
        )
    # src_b (the second source) must not have been copied over the first.
    assert not (work / "shared" / "data.txt").exists() or (
        src_b / "data.txt"
    ).read_text(encoding="utf-8") == "B-content"


def test_non_privilege_symlink_error_reraises(monkeypatch, tmp_path):
    src = _mk_src(tmp_path)
    work = tmp_path / "work"
    work.mkdir()

    def fake_symlink(a, b, **kw):
        e = OSError()
        e.errno = errno.EACCES
        raise e

    monkeypatch.setattr("skillopt.model.codex_harness.os.symlink", fake_symlink)
    with pytest.raises(OSError):
        prepare_workspace(work_dir=str(work), skill_md="x", link_dirs=[(str(src), "docs/data")])


def test_redact_copilot_trace_redacts_json_secret_fields():
    line = '{"type":"message","token":"plain-secret","content":"hi"}'
    out = _redact_copilot_trace(line)
    assert "plain-secret" not in out
    # Non-secret fields survive (content is an omitted trace field by design).
    assert "message" in out


def test_redact_copilot_trace_preserves_normal_strings():
    out = _redact_copilot_trace('{"type":"assistant","message":"hello"}')
    # Normal field names are preserved.
    assert "hello" in out


def test_redact_copilot_trace_redacts_string_key_forms():
    out = _redact_copilot_trace("token = plain-secret")
    assert "plain-secret" not in out
