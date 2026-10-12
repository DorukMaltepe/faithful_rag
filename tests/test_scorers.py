import pytest

from ragproof.models import Case, ScoreDetail
from ragproof.scorers import MissingDependencyError, get_scorer
from ragproof.scorers.base import split_sentences
from ragproof.scorers.lettuce import score_from_spans
from ragproof.scorers.lexical import LexicalScorer
from ragproof.scorers.minicheck import aggregate_sentence_scores


def test_lexical_scores_supported_answer_high():
    case = Case(
        id="1", contexts=["Munich is the capital of Bavaria."], answer="Munich is the capital."
    )
    [detail] = LexicalScorer().score_batch([case])
    assert detail.score == 1.0
    assert detail.details["unsupported_tokens"] == []


def test_lexical_flags_wrong_number():
    case = Case(id="1", contexts=["Founded in 1868."], answer="Founded in 1869.")
    [detail] = LexicalScorer().score_batch([case])
    assert detail.score == 0.5
    assert detail.details["unsupported_tokens"] == ["1869"]


def test_lexical_only_stopwords_counts_as_supported():
    [detail] = LexicalScorer().score_batch([Case(id="1", contexts=["x"], answer="It is.")])
    assert detail.score == 1.0


def test_score_detail_rejects_out_of_range():
    with pytest.raises(ValueError):
        ScoreDetail(1.5)


def test_get_scorer_unknown_name():
    with pytest.raises(ValueError, match="Unknown scorer"):
        get_scorer("nope")


def test_model_scorers_explain_missing_extras(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.split(".")[0] in {"transformers", "lettucedetect", "minicheck"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    for name in ("hhem", "lettucedetect", "minicheck"):
        with pytest.raises(MissingDependencyError, match=name):
            get_scorer(name)


def test_spans_to_score_merges_overlaps():
    answer = "0123456789"
    spans = [
        {"start": 2, "end": 5, "confidence": 0.9, "text": "234"},
        {"start": 4, "end": 6, "confidence": 0.7, "text": "45"},
    ]
    detail = score_from_spans(answer, spans)
    assert detail.score == 0.6
    assert [s["text"] for s in detail.details["hallucinated_spans"]] == ["234", "45"]


def test_spans_to_score_no_spans():
    assert score_from_spans("all good", []).score == 1.0


def test_minicheck_uses_weakest_sentence():
    details = aggregate_sentence_scores(
        2, owners=[0, 0, 1], claims=["s1", "s2", "t1"], probs=[0.9, 0.2, 0.8]
    )
    assert [d.score for d in details] == [0.2, 0.8]
    assert details[0].details["sentences"][1] == {"sentence": "s2", "score": 0.2}


def test_split_sentences():
    text = "TUM was founded in 1868. It has 50,000 students! Is it big? Yes."
    assert split_sentences(text) == [
        "TUM was founded in 1868.",
        "It has 50,000 students!",
        "Is it big?",
        "Yes.",
    ]


def test_spans_to_score_handles_nested_spans():
    spans = [
        {"start": 0, "end": 8, "confidence": 0.9},
        {"start": 2, "end": 4, "confidence": 0.8},
        {"start": 6, "end": 9, "confidence": 0.7},
    ]
    assert score_from_spans("0123456789", spans).score == 0.1
