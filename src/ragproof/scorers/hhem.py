"""Vectara's HHEM-2.1-Open hallucination evaluation model.

HHEM is a cross-encoder that takes a (premise, hypothesis) pair and returns the
probability that the hypothesis is consistent with the premise.
"""

from __future__ import annotations

from collections.abc import Sequence

from ragproof.models import Case, ScoreDetail
from ragproof.scorers.base import MissingDependencyError, Scorer, join_contexts

DEFAULT_MODEL = "vectara/hallucination_evaluation_model"


class HHEMScorer(Scorer):
    name = "hhem"

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 8) -> None:
        try:
            from transformers import AutoModelForSequenceClassification
        except ImportError as exc:
            raise MissingDependencyError(self.name, "pip install 'ragproof[hhem]'") from exc

        # HHEM ships its own modelling code (and its predict() helper) on the Hub.
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._model.eval()
        self._batch_size = batch_size

    def score_batch(self, cases: Sequence[Case]) -> list[ScoreDetail]:
        pairs = [(join_contexts(c), c.answer) for c in cases]
        scores: list[float] = []
        for i in range(0, len(pairs), self._batch_size):
            out = self._model.predict(pairs[i : i + self._batch_size])
            scores.extend(float(s) for s in out)
        return [ScoreDetail(round(min(max(s, 0.0), 1.0), 4)) for s in scores]
