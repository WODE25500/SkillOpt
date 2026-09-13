"""Paired A/B evaluation kit for SkillOpt-Sleep.

Sleep reports (and many PRs) quote single-run success rates with no
uncertainty and no guarantee that the two conditions saw the same tasks.
This module is the shared instrument for those comparisons:

  * one fixed task manifest, paired by task id
  * McNemar's test on per-task binary outcomes
  * percentile-bootstrap confidence intervals on the success-rate delta
  * optional multi-seed repeats with task-cluster inference

It does not change the nightly gate. It standardizes the evidence that
reports and PRs cite. Pure stdlib; no numpy / scipy.

Refuse comparisons whose task-id sets differ. Graded (non-binary) scores
are bootstrap-only: McNemar is not defined for them.

CLI::

    python -m skillopt_sleep.evalkit --manifest M.json --a A.json --b B.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# ── errors ────────────────────────────────────────────────────────────────────

class EvalkitError(ValueError):
    """User-facing contract failure (mismatched ids, empty, etc.)."""


MAX_BOOTSTRAPS = 1_000_000
MAX_BOOTSTRAP_DRAWS = 50_000_000
MAX_MCNEMAR_DISCORDANTS = 1_000_000


def _validate_alpha(alpha: float) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise EvalkitError("alpha must be a finite number strictly between 0 and 1")
    try:
        value = float(alpha)
    except OverflowError:
        raise EvalkitError("alpha must be a finite number strictly between 0 and 1") from None
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise EvalkitError("alpha must be a finite number strictly between 0 and 1")
    return value


def _validate_bootstraps(n_boot: int) -> int:
    if isinstance(n_boot, bool) or not isinstance(n_boot, int):
        raise EvalkitError("n_boot must be an integer")
    if not 1 <= n_boot <= MAX_BOOTSTRAPS:
        raise EvalkitError(f"n_boot must be between 1 and {MAX_BOOTSTRAPS}")
    return n_boot


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise EvalkitError("seed must be an integer")
    return seed


def _validate_count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvalkitError(f"{name} must be a non-negative integer")
    return value


# ── results ───────────────────────────────────────────────────────────────────

@dataclass
class McNemarResult:
    both_success: int
    a_only: int          # A success, B fail  (c in the usual 2x2)
    b_only: int          # A fail, B success  (b)
    both_fail: int
    n: int
    chi2: float          # uncorrected (b-c)^2 / (b+c); nan if no discordants
    p_chi2: float
    p_exact: float       # two-sided exact binomial on discordants
    significant: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BootstrapCI:
    n_boot: int
    seed: int
    alpha: float
    low: float
    high: float
    mean: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvalReport:
    n_tasks: int
    rate_a: float
    rate_b: float
    delta: float
    mcnemar: Optional[McNemarResult]
    bootstrap: BootstrapCI
    per_seed: List[Dict[str, float]] = field(default_factory=list)
    seed_mean_delta: Optional[float] = None
    seed_sd_delta: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    refused: bool = False
    refuse_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.mcnemar is not None:
            d["mcnemar"] = self.mcnemar.to_dict()
        d["bootstrap"] = self.bootstrap.to_dict()
        return d


# ── statistics ────────────────────────────────────────────────────────────────

def _chi2_sf_df1(x: float) -> float:
    """Survival function of chi-square with 1 df: P(X > x) = erfc(sqrt(x/2))."""
    if x < 0.0 or math.isnan(x):
        return float("nan")
    if x == 0.0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def _binom_pmf(k: int, n: int, p: float = 0.5) -> float:
    if k < 0 or k > n:
        return 0.0
    # Retained for callers/tests that need one PMF value. Use log-gamma so the
    # integer binomial coefficient is never coerced to an overflowing float.
    if p == 0.5:
        log_pmf = (
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
            - n * math.log(2.0)
        )
        return math.exp(log_pmf)
    if not 0.0 < p < 1.0:
        raise EvalkitError("binomial p must be strictly between 0 and 1")
    log_pmf = (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )
    return math.exp(log_pmf)


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial test of discordants, p=0.5)."""
    b = _validate_count("b", b)
    c = _validate_count("c", c)
    n = b + c
    if n == 0:
        return 1.0
    if n > MAX_MCNEMAR_DISCORDANTS:
        raise EvalkitError(
            "exact McNemar workload exceeds the "
            f"{MAX_MCNEMAR_DISCORDANTS} discordant-pair limit"
        )
    k = min(b, c)
    # Start at the largest term in the requested lower tail, then recur
    # downward. This avoids both the enormous int-to-float conversion in
    # comb(n, k) * 0.5**n and loss from starting at an underflowed 2**-n.
    def lower_tail_terms() -> Iterable[float]:
        # Feed fsum lazily so memory stays bounded independently of k.
        term = _binom_pmf(k, n, 0.5)
        yield term
        for i in range(k, 0, -1):
            term *= i / (n - i + 1)
            yield term

    tail = math.fsum(lower_tail_terms())
    return min(1.0, 2.0 * tail)


def mcnemar_from_counts(
    both_success: int,
    a_only: int,
    b_only: int,
    both_fail: int,
    *,
    alpha: float = 0.05,
) -> McNemarResult:
    alpha = _validate_alpha(alpha)
    both_success = _validate_count("both_success", both_success)
    a_only = _validate_count("a_only", a_only)
    b_only = _validate_count("b_only", b_only)
    both_fail = _validate_count("both_fail", both_fail)
    n = both_success + a_only + b_only + both_fail
    if n == 0:
        raise EvalkitError("McNemar requires at least one paired observation")
    disc = a_only + b_only
    if disc > MAX_MCNEMAR_DISCORDANTS:
        raise EvalkitError(
            "exact McNemar workload exceeds the "
            f"{MAX_MCNEMAR_DISCORDANTS} discordant-pair limit"
        )
    if disc == 0:
        chi2 = 0.0
        p_chi2 = 1.0
    else:
        chi2 = (b_only - a_only) ** 2 / float(disc)
        p_chi2 = _chi2_sf_df1(chi2)
    p_exact = exact_mcnemar_p(b_only, a_only)
    return McNemarResult(
        both_success=both_success,
        a_only=a_only,
        b_only=b_only,
        both_fail=both_fail,
        n=n,
        chi2=chi2,
        p_chi2=p_chi2,
        p_exact=p_exact,
        significant=p_exact < alpha,
    )


def mcnemar_paired(a: Sequence[int], b: Sequence[int], *, alpha: float = 0.05) -> McNemarResult:
    alpha = _validate_alpha(alpha)
    if len(a) != len(b) or not a:
        raise EvalkitError("McNemar requires a non-empty, equal-length paired sample")
    bs = ao = bo = bf = 0
    for x, y in zip(a, b):
        if _as_binary(x) is None or _as_binary(y) is None:
            raise EvalkitError("McNemar outcomes must be binary 0/1 values")
        if x and y:
            bs += 1
        elif x and not y:
            ao += 1
        elif (not x) and y:
            bo += 1
        else:
            bf += 1
    return mcnemar_from_counts(bs, ao, bo, bf, alpha=alpha)


def bootstrap_delta_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> BootstrapCI:
    alpha = _validate_alpha(alpha)
    n_boot = _validate_bootstraps(n_boot)
    seed = _validate_seed(seed)
    if len(a) != len(b) or not a:
        raise EvalkitError("bootstrap requires a non-empty paired sample")
    n = len(a)
    if n_boot > MAX_BOOTSTRAP_DRAWS // n:
        raise EvalkitError(
            "bootstrap workload exceeds the "
            f"{MAX_BOOTSTRAP_DRAWS} paired-draw limit (n_tasks * n_boot)"
        )
    if any(
        isinstance(x, bool) or not isinstance(x, (int, float))
        for values in (a, b)
        for x in values
    ):
        raise EvalkitError("bootstrap scores must be JSON numbers, not booleans or strings")
    try:
        numeric_a = [float(x) for x in a]
        numeric_b = [float(x) for x in b]
    except OverflowError:
        raise EvalkitError("bootstrap scores must be numeric and finite") from None
    if any(not math.isfinite(x) for x in numeric_a + numeric_b):
        raise EvalkitError("bootstrap scores must be numeric and finite")
    if any(not 0.0 <= x <= 1.0 for x in numeric_a + numeric_b):
        raise EvalkitError("bootstrap scores must be between 0 and 1")
    rng = random.Random(seed)
    paired_deltas = [right - left for left, right in zip(numeric_a, numeric_b)]
    deltas: List[float] = []
    for _ in range(n_boot):
        deltas.append(
            math.fsum(paired_deltas[rng.randrange(n)] for _ in range(n)) / n
        )
    deltas.sort()
    # Inclusive percentile on the sorted sample.
    lo_i = int(math.floor((alpha / 2.0) * (n_boot - 1)))
    hi_i = int(math.ceil((1.0 - alpha / 2.0) * (n_boot - 1)))
    lo_i = max(0, min(n_boot - 1, lo_i))
    hi_i = max(0, min(n_boot - 1, hi_i))
    return BootstrapCI(
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
        low=deltas[lo_i],
        high=deltas[hi_i],
        mean=sum(deltas) / n_boot,
    )


# ── pairing / loading ─────────────────────────────────────────────────────────

def _as_binary(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value == 1 or value == 1.0:
        return 1
    if value == 0 or value == 0.0:
        return 0
    return None


def _task_id(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvalkitError(f"{context} task ids must be non-empty JSON strings")
    return value


def _display_task_id(value: str) -> str:
    display = value if len(value) <= 80 else value[:77] + "..."
    return repr(display)


def _score(value: Any, *, task_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvalkitError(
            f"task {_display_task_id(task_id)} scores must be JSON numbers, "
            "not booleans or strings"
        )
    try:
        score = float(value)
    except OverflowError:
        raise EvalkitError(
            f"task {_display_task_id(task_id)} scores must be finite and between 0 and 1"
        ) from None
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise EvalkitError(
            f"task {_display_task_id(task_id)} scores must be finite and between 0 and 1"
        )
    return score


def _normalize_outcomes(raw: Mapping[str, Any]) -> Dict[str, List[float]]:
    """Map task id -> positional repeat scores (length 1 if unseeded)."""
    if not isinstance(raw, Mapping):
        raise EvalkitError("outcomes must be a JSON object keyed by task id")
    out: Dict[str, List[float]] = {}
    for tid, val in raw.items():
        key = _task_id(tid, context="outcome")
        if isinstance(val, Mapping):
            if set(val) != {"seeds"} or not isinstance(val["seeds"], list):
                raise EvalkitError(
                    f"task {_display_task_id(key)} must contain only a seeds array"
                )
            val = val["seeds"]
        if isinstance(val, (list, tuple)):
            values = list(val)
        else:
            values = [val]
        if not values:
            raise EvalkitError(f"task {_display_task_id(key)} has an empty seed list")
        out[key] = [_score(item, task_id=key) for item in values]
    return out


def align_pairs(
    manifest_ids: Sequence[str],
    outcomes_a: Mapping[str, Any],
    outcomes_b: Mapping[str, Any],
) -> Tuple[List[str], List[List[float]], List[List[float]]]:
    """Align A and B onto the manifest. Refuse any id-set mismatch."""
    if isinstance(manifest_ids, (str, bytes)) or not isinstance(manifest_ids, Sequence):
        raise EvalkitError("manifest task ids must be a JSON array")
    ids = [_task_id(item, context="manifest") for item in manifest_ids]
    if not ids:
        raise EvalkitError("manifest is empty")
    if len(ids) != len(set(ids)):
        raise EvalkitError("manifest has duplicate task ids")
    a = _normalize_outcomes(outcomes_a)
    b = _normalize_outcomes(outcomes_b)
    a_ids, b_ids = set(a), set(b)
    want = set(ids)
    if a_ids != want or b_ids != want:
        missing_a = sorted(want - a_ids)
        missing_b = sorted(want - b_ids)
        extra_a = sorted(a_ids - want)
        extra_b = sorted(b_ids - want)
        raise EvalkitError(
            "outcome task ids must equal the manifest "
            f"(missing_a={missing_a[:8]}, missing_b={missing_b[:8]}, "
            f"extra_a={extra_a[:8]}, extra_b={extra_b[:8]})"
        )
    n_seed_a = {len(a[i]) for i in ids}
    n_seed_b = {len(b[i]) for i in ids}
    if len(n_seed_a) != 1 or n_seed_a != n_seed_b:
        raise EvalkitError("every task must have the same number of seed repeats in A and B")
    return ids, [a[i] for i in ids], [b[i] for i in ids]


def _is_binary_matrix(rows: Sequence[Sequence[float]]) -> bool:
    for row in rows:
        for x in row:
            if _as_binary(x) is None:
                return False
    return True


def _mean(xs: Iterable[float]) -> float:
    seq = list(xs)
    return sum(seq) / len(seq) if seq else float("nan")


def _sd(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def reconstruct_paired_from_rates(n: int, rate_a: float, rate_b: float) -> Tuple[List[int], List[int]]:
    """Deterministic maximum-concordance reconstruction of paired binaries.

    First ``round(n * rate)`` tasks succeed in each condition, same id order.
    This is a published-rate replay convention, not original microdata.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise EvalkitError("n must be >= 1")
    if any(
        isinstance(rate, bool) or not isinstance(rate, (int, float))
        for rate in (rate_a, rate_b)
    ):
        raise EvalkitError("rates must be JSON numbers, finite, and between 0 and 1")
    try:
        numeric_a, numeric_b = float(rate_a), float(rate_b)
    except OverflowError:
        raise EvalkitError("rates must be finite and between 0 and 1") from None
    if not all(math.isfinite(rate) and 0.0 <= rate <= 1.0 for rate in (numeric_a, numeric_b)):
        raise EvalkitError("rates must be finite and between 0 and 1")
    ka = int(round(n * numeric_a))
    kb = int(round(n * numeric_b))
    a = [1 if i < ka else 0 for i in range(n)]
    b = [1 if i < kb else 0 for i in range(n)]
    return a, b


def compare(
    manifest_ids: Sequence[str],
    outcomes_a: Mapping[str, Any],
    outcomes_b: Mapping[str, Any],
    *,
    alpha: float = 0.05,
    n_boot: int = 10000,
    seed: int = 42,
    allow_graded: bool = False,
) -> EvalReport:
    alpha = _validate_alpha(alpha)
    n_boot = _validate_bootstraps(n_boot)
    if not isinstance(allow_graded, bool):
        raise EvalkitError("allow_graded must be a boolean")
    ids, a_rows, b_rows = align_pairs(manifest_ids, outcomes_a, outcomes_b)
    n_seed = len(a_rows[0])
    notes: List[str] = []

    # Per-task mean across seeds (the headline paired sample).
    a_mean = [_mean(row) for row in a_rows]
    b_mean = [_mean(row) for row in b_rows]
    rate_a = _mean(a_mean)
    rate_b = _mean(b_mean)
    delta = rate_b - rate_a
    boot = bootstrap_delta_ci(a_mean, b_mean, n_boot=n_boot, seed=seed, alpha=alpha)

    binary = _is_binary_matrix(a_rows) and _is_binary_matrix(b_rows)
    mcnemar: Optional[McNemarResult] = None
    if binary and n_seed == 1:
        # One independent binary observation per task: McNemar's intended unit.
        mcnemar = mcnemar_paired(
            [int(_as_binary(row[0]) or 0) for row in a_rows],
            [int(_as_binary(row[0]) or 0) for row in b_rows],
            alpha=alpha,
        )
    elif binary:
        # Repeated seeds within a task are clustered measurements, not
        # independent observations. The task-level bootstrap above is the
        # inferential result; pooling here would create pseudoreplication.
        notes.append(
            "multi-seed binary scores: McNemar omitted; task-cluster bootstrap CI is authoritative"
        )
    elif allow_graded:
        notes.append("graded scores: McNemar omitted; bootstrap CI only")
    else:
        raise EvalkitError(
            "non-binary scores require --allow-graded (McNemar is undefined)"
        )

    per_seed: List[Dict[str, float]] = []
    seed_mean = seed_sd = None
    if n_seed > 1:
        for s in range(n_seed):
            da = _mean(row[s] for row in a_rows)
            db = _mean(row[s] for row in b_rows)
            per_seed.append(
                {"seed_index": s, "rate_a": da, "rate_b": db, "delta": db - da}
            )
        deltas = [row["delta"] for row in per_seed]
        seed_mean = _mean(deltas)
        seed_sd = _sd(deltas)
        notes.append(
            f"multi-seed positional repeats: {n_seed}; mean delta={seed_mean:.6f} "
            f"sample sd={seed_sd:.6f}; descriptive only, not an uncertainty estimate"
        )
        notes.append(
            "seed lists are positional repeats; seed_index is not a verified RNG seed id"
        )

    return EvalReport(
        n_tasks=len(ids),
        rate_a=rate_a,
        rate_b=rate_b,
        delta=delta,
        mcnemar=mcnemar,
        bootstrap=boot,
        per_seed=per_seed,
        seed_mean_delta=seed_mean,
        seed_sd_delta=seed_sd,
        notes=notes,
    )


def compare_aa(
    manifest_ids: Sequence[str],
    outcomes: Mapping[str, Any],
    **kwargs: Any,
) -> EvalReport:
    """A/A identity smoke check: identical conditions must not reject."""
    report = compare(manifest_ids, outcomes, outcomes, **kwargs)
    report.notes.append("A/A identity smoke check (identical conditions)")
    return report


# ── I/O ───────────────────────────────────────────────────────────────────────

def _reject_duplicate_object_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    obj: Dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            display_key = key if len(key) <= 80 else key[:77] + "..."
            raise EvalkitError(f"duplicate JSON object key: {display_key!r}")
        obj[key] = value
    return obj


def _reject_json_constant(value: str) -> Any:
    raise EvalkitError(f"non-standard JSON constant {value} is not allowed")


def _load_json(path: str, *, label: str) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(
                f,
                object_pairs_hook=_reject_duplicate_object_keys,
                parse_constant=_reject_json_constant,
            )
    except EvalkitError as exc:
        raise EvalkitError(f"invalid JSON in {label}: {exc}") from None
    except (json.JSONDecodeError, UnicodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, json.JSONDecodeError):
            detail = exc.msg
        elif isinstance(exc, UnicodeError):
            detail = "invalid text encoding"
        elif isinstance(exc, RecursionError):
            detail = "nesting is too deep"
        else:
            detail = "invalid numeric literal"
        raise EvalkitError(f"invalid JSON in {label}: {detail}") from None
    except OSError as exc:
        detail = exc.strerror or type(exc).__name__
        raise EvalkitError(f"could not read {label}: {detail}") from None


def _manifest_ids(obj: Any) -> List[str]:
    if isinstance(obj, list):
        return [_task_id(item, context="manifest") for item in obj]
    if isinstance(obj, Mapping):
        forms = [name for name in ("ids", "tasks", "outcomes") if name in obj]
        if len(forms) > 1:
            raise EvalkitError("manifest must use exactly one of ids, tasks, or outcomes")
        if "ids" in obj:
            values = obj["ids"]
            if not isinstance(values, list):
                raise EvalkitError("manifest ids must be a JSON array")
            return [_task_id(item, context="manifest") for item in values]
        if "tasks" in obj:
            tasks = obj["tasks"]
            if not isinstance(tasks, list):
                raise EvalkitError("manifest tasks must be a JSON array")
            ids: List[str] = []
            for task in tasks:
                if not isinstance(task, Mapping) or "id" not in task:
                    raise EvalkitError("every manifest task must be an object containing id")
                ids.append(_task_id(task["id"], context="manifest"))
            return ids
        if "outcomes" in obj:
            outcomes = obj["outcomes"]
            if not isinstance(outcomes, Mapping):
                raise EvalkitError("manifest outcomes must be a JSON object")
            return [_task_id(key, context="manifest") for key in outcomes]
    raise EvalkitError("manifest must be a list of ids or an object with ids/tasks")


def _outcomes(obj: Any, manifest_ids: Sequence[str]) -> Dict[str, Any]:
    if not isinstance(obj, Mapping):
        raise EvalkitError("outcomes file must be an object mapping task id to score")
    # A direct mapping wins when its keys exactly match the manifest. This makes
    # a task literally named ``outcomes`` unambiguous even when its score is a
    # seeded object. Otherwise the one-key wrapper is recognized.
    if set(obj) == set(manifest_ids):
        return dict(obj)
    if set(obj) == {"outcomes"} and isinstance(obj["outcomes"], Mapping):
        values = obj["outcomes"]
        return dict(values)
    return dict(obj)


def format_markdown(report: EvalReport) -> str:
    lines = [
        "# Paired A/B evalkit report",
        "",
        f"- n_tasks: {report.n_tasks}",
        f"- rate_a: {report.rate_a:.6f}",
        f"- rate_b: {report.rate_b:.6f}",
        f"- delta (B-A): {report.delta:+.6f}",
        (
            f"- bootstrap {100 * (1 - report.bootstrap.alpha):g}% CI: "
            f"[{report.bootstrap.low:+.6f}, {report.bootstrap.high:+.6f}] "
            f"(n_boot={report.bootstrap.n_boot}, seed={report.bootstrap.seed})"
        ),
    ]
    if report.mcnemar is not None:
        m = report.mcnemar
        lines.append(
            f"- McNemar 2x2: both+={m.both_success} a_only={m.a_only} "
            f"b_only={m.b_only} both-={m.both_fail}"
        )
        lines.append(
            f"- McNemar chi2={m.chi2:.4f} p_chi2={m.p_chi2:.6g} "
            f"p_exact={m.p_exact:.6g} significant={m.significant}"
        )
    if report.seed_mean_delta is not None:
        lines.append(
            f"- positional-repeat mean delta: {report.seed_mean_delta:+.6f} "
            f"(descriptive sample sd {report.seed_sd_delta:.6f}, "
            f"k={len(report.per_seed)}; not an uncertainty estimate)"
        )
    for note in report.notes:
        lines.append(f"- note: {note}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="skillopt_sleep.evalkit",
        description="Paired A/B comparison with McNemar and bootstrap CIs",
    )
    p.add_argument("--manifest", required=True, help="JSON list of task ids (or {ids,tasks})")
    p.add_argument("--a", required=True, help="JSON outcomes for condition A")
    p.add_argument("--b", default=None, help="JSON outcomes for condition B (required unless --aa)")
    p.add_argument("--aa", action="store_true", help="A/A identity smoke check (mutually exclusive with --b)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-graded", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    try:
        if (args.b is not None) == args.aa:
            raise EvalkitError("exactly one of --b or --aa is required")
        ids = _manifest_ids(_load_json(args.manifest, label="--manifest"))
        a = _outcomes(_load_json(args.a, label="--a"), ids)
        if args.aa:
            report = compare_aa(
                ids, a, alpha=args.alpha, n_boot=args.boot,
                seed=args.seed, allow_graded=args.allow_graded,
            )
        else:
            b = _outcomes(_load_json(args.b, label="--b"), ids)
            report = compare(
                ids, a, b, alpha=args.alpha, n_boot=args.boot,
                seed=args.seed, allow_graded=args.allow_graded,
            )
    except EvalkitError as exc:
        print(f"ERR_EVALKIT {exc}", file=sys.stderr)
        return 2

    if args.json:
        try:
            print(json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False))
        except (TypeError, ValueError) as exc:
            print(f"ERR_EVALKIT report is not strict JSON: {exc}", file=sys.stderr)
            return 2
    else:
        print(format_markdown(report), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
