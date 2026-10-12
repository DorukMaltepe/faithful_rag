"""Scorer registry. Backends are imported lazily so unused extras are never required."""

from __future__ import annotations

from collections.abc import Callable

from ragproof.scorers.base import MissingDependencyError, Scorer


def _lexical() -> Scorer:
    from ragproof.scorers.lexical import LexicalScorer

    return LexicalScorer()


def _hhem() -> Scorer:
    from ragproof.scorers.hhem import HHEMScorer

    return HHEMScorer()


def _lettucedetect() -> Scorer:
    from ragproof.scorers.lettuce import LettuceDetectScorer

    return LettuceDetectScorer()


def _minicheck() -> Scorer:
    from ragproof.scorers.minicheck import MiniCheckScorer

    return MiniCheckScorer()


REGISTRY: dict[str, Callable[[], Scorer]] = {
    "lexical": _lexical,
    "hhem": _hhem,
    "lettucedetect": _lettucedetect,
    "minicheck": _minicheck,
}


def get_scorer(name: str) -> Scorer:
    try:
        factory = REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown scorer {name!r}. Available: {', '.join(sorted(REGISTRY))}"
        ) from None
    return factory()


__all__ = ["REGISTRY", "MissingDependencyError", "Scorer", "get_scorer"]
