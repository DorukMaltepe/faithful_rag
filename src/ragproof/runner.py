"""Run scorers over a dataset and summarise the results."""

from __future__ import annotations

from collections.abc import Sequence

from ragproof.models import Case, CaseResult, Report, ScoreDetail, ScorerSummary
from ragproof.scorers import Scorer


def evaluate(cases: Sequence[Case], scorers: Sequence[Scorer], threshold: float) -> Report:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if not scorers:
        raise ValueError("at least one scorer is required")

    per_scorer: dict[str, list[ScoreDetail]] = {}
    for scorer in scorers:
        details = scorer.score_batch(cases)
        if len(details) != len(cases):
            raise RuntimeError(
                f"scorer {scorer.name!r} returned {len(details)} scores for {len(cases)} cases"
            )
        per_scorer[scorer.name] = details

    results = [
        CaseResult(case=case, scores={name: details[i] for name, details in per_scorer.items()})
        for i, case in enumerate(cases)
    ]
    summaries = {
        name: summarise(name, cases, details, threshold) for name, details in per_scorer.items()
    }
    return Report(threshold=threshold, summaries=summaries, results=results)


def summarise(
    name: str, cases: Sequence[Case], details: Sequence[ScoreDetail], threshold: float
) -> ScorerSummary:
    scores = [d.score for d in details]
    failing = [c.id for c, s in zip(cases, scores, strict=True) if s < threshold]
    return ScorerSummary(
        scorer=name,
        mean=round(sum(scores) / len(scores), 4),
        min=round(min(scores), 4),
        pass_rate=round(1 - len(failing) / len(scores), 4),
        failing=failing,
    )
