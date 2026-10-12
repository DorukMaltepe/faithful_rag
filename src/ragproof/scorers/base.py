"""The scorer interface every faithfulness backend implements."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence

from ragproof.models import Case, ScoreDetail


class Scorer(ABC):
    """Scores how well each answer is supported by its retrieved contexts.

    Implementations receive the whole batch so model-backed scorers can run
    batched inference, and must return one ``ScoreDetail`` per case, in order.
    """

    name: str = "base"

    @abstractmethod
    def score_batch(self, cases: Sequence[Case]) -> list[ScoreDetail]: ...


class MissingDependencyError(ImportError):
    """Raised when a scorer's optional backend is not installed."""

    def __init__(self, scorer: str, install_hint: str) -> None:
        super().__init__(f"The '{scorer}' scorer needs extra dependencies: {install_hint}")


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter, good enough for claim-level checking of RAG answers."""
    parts = [p.strip() for p in _SENTENCE_RE.split(text.strip())]
    return [p for p in parts if p]


def join_contexts(case: Case) -> str:
    return "\n\n".join(case.contexts)
