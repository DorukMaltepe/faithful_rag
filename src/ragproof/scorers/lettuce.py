"""LettuceDetect: token-level hallucination detection with span output.

The score is the share of answer characters NOT inside a predicted hallucinated
span, so a short fabricated number in a long answer costs little, while an answer
that is mostly invented scores close to zero. The spans themselves go into the
report so reviewers can see exactly which phrase was flagged.
"""

from __future__ import annotations

from collections.abc import Sequence

from ragproof.models import Case, ScoreDetail
from ragproof.scorers.base import MissingDependencyError, Scorer

DEFAULT_MODEL = "KRLabsOrg/lettucedect-base-modernbert-en-v1"


class LettuceDetectScorer(Scorer):
    name = "lettucedetect"

    def __init__(self, model_name: str = DEFAULT_MODEL, min_confidence: float = 0.5) -> None:
        try:
            from lettucedetect.models.inference import HallucinationDetector
        except ImportError as exc:
            raise MissingDependencyError(
                self.name, "pip install 'ragproof[lettucedetect]' (Python >= 3.11)"
            ) from exc

        self._detector = HallucinationDetector(method="transformer", model_path=model_name)
        self._min_confidence = min_confidence

    def score_batch(self, cases: Sequence[Case]) -> list[ScoreDetail]:
        return [self._score(c) for c in cases]

    def _score(self, case: Case) -> ScoreDetail:
        spans = self._detector.predict(
            context=case.contexts,
            question=case.question,
            answer=case.answer,
            output_format="spans",
            min_confidence=self._min_confidence,
        )
        return score_from_spans(case.answer, spans)


def score_from_spans(answer: str, spans: list[dict]) -> ScoreDetail:
    """Turn character spans into a support score. Overlapping spans are merged first."""
    covered = 0
    last_end = 0
    for span in sorted(spans, key=lambda s: s["start"]):
        start = max(span["start"], last_end)
        end = min(span["end"], len(answer))
        if end > start:
            covered += end - start
            last_end = max(last_end, end)
    score = 1.0 - covered / len(answer) if answer else 1.0
    evidence = [
        {
            "text": s.get("text", answer[s["start"] : s["end"]]),
            "confidence": round(float(s.get("confidence", 0.0)), 4),
        }
        for s in spans
    ]
    return ScoreDetail(round(score, 4), {"hallucinated_spans": evidence})
