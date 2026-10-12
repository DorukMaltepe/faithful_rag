"""MiniCheck: sentence-level fact checking against a grounding document.

MiniCheck is trained on single-sentence claims, so the answer is split into
sentences, each is checked against the joined contexts, and the case score is the
weakest sentence's score: one unsupported sentence is enough to fail the answer.
"""

from __future__ import annotations

from collections.abc import Sequence

from ragproof.models import Case, ScoreDetail
from ragproof.scorers.base import MissingDependencyError, Scorer, join_contexts, split_sentences

DEFAULT_MODEL = "flan-t5-large"


class MiniCheckScorer(Scorer):
    name = "minicheck"

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: str = "./.ragproof-cache"):
        try:
            from minicheck.minicheck import MiniCheck
        except ImportError as exc:
            raise MissingDependencyError(
                self.name,
                'pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"',
            ) from exc

        self._model = MiniCheck(model_name=model_name, cache_dir=cache_dir)

    def score_batch(self, cases: Sequence[Case]) -> list[ScoreDetail]:
        docs: list[str] = []
        claims: list[str] = []
        owners: list[int] = []
        for idx, case in enumerate(cases):
            doc = join_contexts(case)
            for sentence in split_sentences(case.answer) or [case.answer]:
                docs.append(doc)
                claims.append(sentence)
                owners.append(idx)

        _, probs, _, _ = self._model.score(docs=docs, claims=claims)
        return aggregate_sentence_scores(len(cases), owners, claims, [float(p) for p in probs])


def aggregate_sentence_scores(
    n_cases: int, owners: list[int], claims: list[str], probs: list[float]
) -> list[ScoreDetail]:
    per_case: list[list[dict]] = [[] for _ in range(n_cases)]
    for owner, claim, prob in zip(owners, claims, probs, strict=True):
        per_case[owner].append({"sentence": claim, "score": round(prob, 4)})
    return [
        ScoreDetail(min(s["score"] for s in sentences), {"sentences": sentences})
        for sentences in per_case
    ]
