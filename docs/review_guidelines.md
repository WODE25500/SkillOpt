# Code Review & Contribution Guidelines

> Distilled from maintainer review feedback on this repository, these are a
> **shared baseline for both contributors and reviewers**. They are especially
> useful early on — following the pre-submission self-check avoids the common
> blockers that typically surface in a first review round — and they give
> everyone a common vocabulary for what "ready to merge" means here.
> These are **advisory guidance, not a hard policy**: reviewers and maintainers
> apply them with judgment, and a PR that doesn't need a given check should not
> be blocked on it.

## Review approach

- **Recheck the current HEAD** — never review a stale version of the branch.
- **Switch to the target branch before probing it** — the same file path can
  differ across branches; probing from another branch reads an old copy and
  produces false positives/negatives.
- **Reproduce the failure against the real version and real behavior**, not an
  imagined input.
- **Check the real third-party contract** — verify the actual CLI/SDK flags and
  behavior a dependency supports, rather than assuming.
- **Run focused + full test suites** and report the counts as evidence
  (e.g. "99 passed / 1 skipped").
- **Report blockers as "N blockers" with exact file:line and one sentence on why
  it matters.**
- When acknowledging, **point to concrete fixable spots** and distinguish
  "useful but blocked" from "unusable".

## Pre-submission self-check

- [ ] Every supported version of a declared dependency range actually works
      (version-adapt APIs; add a real build/launch smoke test per version).
- [ ] Error fallbacks fire only for the *expected* failure; other errors
      re-raise; existing targets fail closed; no out-of-bounds mutation.
- [ ] Sensitive data is redacted **structurally** (mapping-key aware), on **every
      output boundary** (json/md/stdout/file), while keeping debuggable content.
      Secret-key detection uses **endswith** on the normalized key, not substring
      (so `token_count` / `token_budget` diagnostics are preserved).
- [ ] Shared mutable state (cache, accounting, errors) goes through a **locked**
      helper; one caller's failure does not clobber another's success; accounting
      is call-local; transient failures are not cached.
- [ ] A root-cause fix to a shared function/helper verifies **all callers and
      subclasses** (via `super()` inheritance) are covered, and gives new threads
      a safe default; the test asserts the **quantity itself** so it can fail on
      the old bug.
- [ ] Concurrency tests use a **Barrier** to force overlap; degraded inputs are
      covered (prefixed stderr, multiline, reentrancy), not just the happy path.
- [ ] Third-party contracts are validated against the real version; smoke tests
      exercise real calls, not just `--version`.
- [ ] PR hygiene: original author attribution preserved, rebased on latest main,
      no unrelated stacked commits, focused scope.
- [ ] Domain math (stats/research/numeric) is correct and the formula verified.
- [ ] Config/role routing is consistent (optimizer/target/judge are role-aware;
      matching model fields accompany backend changes).
- [ ] Data isolation: test/eval data never leaks back into training; holdout/test
      sets are explicitly excluded.
- [ ] Trust boundaries: external paths/ids are bound to a verified manifest/hash;
      fail-closed validation.
- [ ] Features wired end-to-end (not just primitives).
- [ ] Deliberate resilience/fallback contracts are respected, not "cleaned up";
      deprecated options retire cleanly rather than being re-purposed.
- [ ] Reuse existing infrastructure over reinventing it.
- [ ] Sensitive content is redacted **before** reaching downstream consumers;
      resource operations are bounded.
- [ ] Mutating endpoints are authenticated and CSRF-protected.
- [ ] Backward compatibility preserved; the same contract behaves identically
      across backends.
- [ ] Filesystem boundaries handled (cross-drive paths, empty home, path
      normalization, symlinks).
- [ ] First mergeable slice submitted; follow-ups as independent PRs — each with
      its own tests.

## Notes

- The most common review root cause is **validating the shape you imagined
  rather than the shape reality produces**. Prefer running the real entry point
  once with a realistic input before writing assertions.
- When a failure is reported, find and fix the **root cause in the shared
  function/helper** (one fix covers all callers) — but align the fix with the
  codebase's existing conventions to avoid over- or under-correction.
