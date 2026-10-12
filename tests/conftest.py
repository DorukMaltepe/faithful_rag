from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from ragproof.models import Case, ScoreDetail
from ragproof.scorers.base import Scorer

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


class FixedScorer(Scorer):
    """Returns preset scores by case id, so runner and gate tests don't depend on a model."""

    def __init__(self, name: str, scores: dict[str, float]) -> None:
        self.name = name
        self._scores = scores

    def score_batch(self, cases: Sequence[Case]) -> list[ScoreDetail]:
        return [ScoreDetail(self._scores[c.id]) for c in cases]


@pytest.fixture
def cases() -> list[Case]:
    return [
        Case(id="a", contexts=["ctx a"], answer="answer a"),
        Case(id="b", contexts=["ctx b"], answer="answer b"),
        Case(id="c", contexts=["ctx c"], answer="answer c"),
    ]


@pytest.fixture
def examples_dir() -> Path:
    return EXAMPLES
