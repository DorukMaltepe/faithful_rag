"""Core data structures shared by the dataset loader, scorers, runner and reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Case:
    """One RAG interaction to check: the retrieved contexts and the generated answer."""

    id: str
    contexts: list[str]
    answer: str
    question: str | None = None


@dataclass
class ScoreDetail:
    """A scorer's verdict for one case.

    ``score`` is always in ``[0, 1]`` where 1.0 means "fully supported by the contexts".
    ``details`` carries scorer-specific evidence, e.g. hallucinated spans or per-sentence scores.
    """

    score: float
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")


@dataclass
class CaseResult:
    case: Case
    scores: dict[str, ScoreDetail]

    def passed(self, threshold: float) -> bool:
        return all(s.score >= threshold for s in self.scores.values())


@dataclass
class ScorerSummary:
    scorer: str
    mean: float
    min: float
    pass_rate: float
    failing: list[str]


@dataclass
class Report:
    """The outcome of one evaluation run. Serialised to JSON so it can act as a baseline."""

    threshold: float
    summaries: dict[str, ScorerSummary]
    results: list[CaseResult]

    def failing_cases(self) -> list[CaseResult]:
        return [r for r in self.results if not r.passed(self.threshold)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "threshold": self.threshold,
            "summaries": {name: asdict(s) for name, s in self.summaries.items()},
            "results": [
                {
                    "id": r.case.id,
                    "question": r.case.question,
                    "answer": r.case.answer,
                    "scores": {name: asdict(s) for name, s in r.scores.items()},
                }
                for r in self.results
            ],
        }
