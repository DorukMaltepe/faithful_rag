"""A dependency-free lexical baseline.

This is NOT a faithfulness model: it only measures how many of the answer's content
words (and every number) also appear in the contexts. It is useful as a fast smoke
test, for wiring up CI before downloading models, and as a deterministic scorer
in ragproof's own test suite.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ragproof.models import Case, ScoreDetail
from ragproof.scorers.base import Scorer, join_contexts

_TOKEN_RE = re.compile(r"\w+(?:[.,]\d+)*", re.UNICODE)

_STOPWORDS_TEXT = """
a an the and or but if then else of to in on at by for with from as is are was were be
been being it its this that these those there here which who whom whose what when where why
how not no do does did has have had can could should would may might will shall so than too
very also into about over under between i you he she we they them his her our your their
"""
_STOPWORDS = frozenset(_STOPWORDS_TEXT.split())


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class LexicalScorer(Scorer):
    name = "lexical"

    def score_batch(self, cases: Sequence[Case]) -> list[ScoreDetail]:
        return [self._score(c) for c in cases]

    def _score(self, case: Case) -> ScoreDetail:
        context_vocab = set(_tokens(join_contexts(case)))
        content = [
            t for t in _tokens(case.answer) if t not in _STOPWORDS or any(ch.isdigit() for ch in t)
        ]
        if not content:
            return ScoreDetail(1.0, {"unsupported_tokens": []})
        unsupported = [t for t in content if t not in context_vocab]
        score = 1.0 - len(unsupported) / len(content)
        return ScoreDetail(round(score, 4), {"unsupported_tokens": sorted(set(unsupported))})
