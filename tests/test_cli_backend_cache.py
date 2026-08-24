"""Tests for CliBackend response caching + thread-safety on the parallel path."""

from __future__ import annotations

import threading

from skillopt_sleep.backend import CliBackend


class _EchoBackend(CliBackend):
    """Minimal backend: echoes the prompt, records call count."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def _call(self, prompt: str, *, max_tokens: int = 1024) -> str:
        self.calls += 1
        return f"resp:{prompt}"


def test_cached_call_caches_and_counts_tokens():
    b = _EchoBackend()
    out = b._cached_call("k:1", "hello")
    assert out == "resp:hello"
    assert b.calls == 1
    assert b._tokens > 0

    # A cache hit returns the value and adds no tokens.
    tokens_before = b._tokens
    out2 = b._cached_call("k:1", "hello")
    assert out2 == "resp:hello"
    assert b.calls == 1  # no new _call on the hit
    assert b._tokens == tokens_before


def test_cached_call_distinct_keys_are_distinct():
    b = _EchoBackend()
    assert b._cached_call("k:1", "a") == "resp:a"
    assert b._cached_call("k:2", "b") == "resp:b"
    assert b.calls == 2


def test_cached_call_concurrent_access_does_not_corrupt():
    """Concurrent workers over one backend must not lose cache/token updates."""
    b = _EchoBackend()
    results: list[str] = []
    errors: list[Exception] = []

    def worker():
        try:
            results.append(b._cached_call("k:1", "hello"))
        except Exception as exc:  # pragma: no cover - safety net
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(results) == 8
    # The cache may duplicate a call on a concurrent miss, but every result is
    # a valid cached value and the cache/token state stays consistent.
    assert all(r == "resp:hello" for r in results)
    assert b._cache["k:1"] == "resp:hello"
