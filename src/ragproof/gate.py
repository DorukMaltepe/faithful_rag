"""Decide whether a run passes: absolute quality bar plus regression against a baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScorerDelta:
    scorer: str
    baseline_mean: float
    current_mean: float

    @property
    def delta(self) -> float:
        return round(self.current_mean - self.baseline_mean, 4)


@dataclass
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    deltas: dict[str, ScorerDelta] = field(default_factory=dict)
    regressed_cases: list[str] = field(default_factory=list)


def check(
    current: dict[str, Any],
    *,
    min_pass_rate: float = 1.0,
    baseline: dict[str, Any] | None = None,
    max_drop: float = 0.02,
) -> GateResult:
    """Evaluate a report (as produced by ``Report.to_dict``) against the gate rules.

    * every scorer's pass rate must be at least ``min_pass_rate``;
    * with a baseline, no scorer's mean may drop by more than ``max_drop``.

    Cases that passed in the baseline but fail now are listed as regressions; they
    don't fail the gate on their own, but they are what a reviewer wants to see first.
    """
    result = GateResult(passed=True)

    for name, summary in current["summaries"].items():
        if summary["pass_rate"] < min_pass_rate:
            result.reasons.append(
                f"{name}: pass rate {summary['pass_rate']:.1%} is below the required "
                f"{min_pass_rate:.1%} ({len(summary['failing'])} failing case(s))"
            )

    if baseline is not None:
        for name, summary in current["summaries"].items():
            base = baseline.get("summaries", {}).get(name)
            if base is None:
                continue
            d = ScorerDelta(name, base["mean"], summary["mean"])
            result.deltas[name] = d
            if d.delta < -max_drop:
                result.reasons.append(
                    f"{name}: mean score dropped {abs(d.delta):.4f} "
                    f"({d.baseline_mean:.4f} -> {d.current_mean:.4f}), "
                    f"more than the allowed {max_drop:.4f}"
                )
        result.regressed_cases = _regressed_cases(current, baseline)

    result.passed = not result.reasons
    return result


def _regressed_cases(current: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    threshold = current["threshold"]

    def passing(report: dict[str, Any], scorers: set[str]) -> dict[str, bool]:
        return {
            r["id"]: all(r["scores"][s]["score"] >= threshold for s in scorers if s in r["scores"])
            for r in report["results"]
        }

    shared = set(current["summaries"]) & set(baseline.get("summaries", {}))
    if not shared:
        return []
    before = passing(baseline, shared)
    after = passing(current, shared)
    return [cid for cid, ok in after.items() if not ok and before.get(cid, False)]
