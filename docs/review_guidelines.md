# Code Review & Contribution Guidelines

> This checklist summarizes common considerations for contributors and
> reviewers. It is advisory, not an exhaustive merge policy: apply the items
> relevant to the scope and risk of a change, and use maintainer judgment for
> merge decisions.

## Review approach

- **Review the current PR HEAD** — confirm the exact revision before probing or
  testing, and use the target branch as the comparison baseline.
- **Reproduce the failure against the real version and real behavior**, not an
  imagined input.
- **Check the real third-party contract** — verify the actual CLI/SDK flags and
  behavior a dependency supports, rather than assuming.
- **Run tests proportionate to the change** — include focused regression tests
  and, when practical, the relevant broader suite; report the commands and
  results.
- **Separate blockers from suggestions** and make feedback actionable by citing
  the relevant behavior or location and explaining the impact.

## Pre-submission self-check

- [ ] The declared dependency range matches supported behavior; test
      representative boundary versions when compatibility differs or the range
      changes.
- [ ] Error fallbacks fire only for the *expected* failure; other errors
      re-raise; existing targets fail closed; no out-of-bounds mutation.
- [ ] Sensitive data is redacted at every relevant output boundary while
      preserving useful diagnostics. Prefer structural redaction for structured
      data, and test both secret-key variants and non-secret lookalikes such as
      `token_count`.
- [ ] Shared mutable state is synchronized with a mechanism appropriate to the
      implementation; failures and accounting remain isolated per operation,
      and cache behavior is explicit.
- [ ] A bug fix considers all affected callers and subclasses and includes a
      regression test that fails before the fix and verifies externally
      observable behavior.
- [ ] Concurrency tests coordinate execution deterministically enough to
      exercise the race (for example with barriers, events, or controlled hooks)
      and cover relevant failure inputs.
- [ ] Third-party behavior is checked against supported versions. Use an
      integration smoke test when practical, or a faithful offline contract test
      when live calls are unsuitable.
- [ ] PR hygiene: focused scope, no unrelated commits, appropriate attribution,
      and a sufficiently current base to assess conflicts and integration.
- [ ] Domain math (stats/research/numeric) is correct and the formula verified.
- [ ] Config/role routing is consistent (optimizer/target/judge are role-aware;
      matching model fields accompany backend changes).
- [ ] Data isolation: test/eval data never leaks back into training; holdout/test
      sets are explicitly excluded.
- [ ] External paths and identifiers are validated at the appropriate trust
      boundary; invalid or untrusted inputs are rejected safely.
- [ ] Features wired end-to-end (not just primitives).
- [ ] Deliberate resilience/fallback contracts are respected, not "cleaned up";
      deprecated options retire cleanly rather than being re-purposed.
- [ ] Prefer existing infrastructure when it fits the requirement; introduce
      new abstractions when they provide a clear benefit.
- [ ] Sensitive content is redacted **before** reaching downstream consumers;
      resource operations are bounded.
- [ ] Mutating endpoints enforce appropriate authorization; browser endpoints
      that rely on ambient credentials include CSRF protection or an equivalent
      defense.
- [ ] Avoid unintended compatibility regressions across supported backends;
      document and test intentional backend-specific differences.
- [ ] Filesystem boundaries handled (cross-drive paths, empty home, path
      normalization, symlinks).
- [ ] Keep each PR a coherent, reviewable slice with tests appropriate to its
      behavior; split unrelated follow-up work into separate PRs.

## Notes

- A recurring review failure mode is validating an assumed shape rather than
  the behavior the real entry point produces. When practical, exercise the real
  entry point with representative input before writing assertions.
- Fix failures at the narrowest layer that correctly covers the affected
  callers; check neighboring callers and follow existing conventions to avoid
  over- or under-correction.
