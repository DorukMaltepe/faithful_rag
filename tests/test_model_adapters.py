"""Exercise the model adapters against stand-ins that mimic each library's public API.

Real model runs need network access to Hugging Face; these tests make
sure the glue code (batching, argument names, score conversion) stays correct.
"""

import sys
import types

from ragproof.models import Case

CASES = [
    Case(
        id="1", contexts=["ctx one", "ctx two"], answer="First claim. Second claim.", question="q"
    ),
    Case(id="2", contexts=["ctx"], answer="Only claim."),
]


def _install(monkeypatch, name, **attrs):
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        mod_name = ".".join(parts[:i])
        mod = sys.modules.get(mod_name)
        if mod is None or not getattr(mod, "_fake", False):
            mod = types.ModuleType(mod_name)
            mod._fake = True
            monkeypatch.setitem(sys.modules, mod_name, mod)
    for key, value in attrs.items():
        setattr(sys.modules[name], key, value)


def test_hhem_adapter_batches_premise_hypothesis_pairs(monkeypatch):
    calls = []

    class FakeModel:
        def eval(self):
            return self

        def predict(self, pairs):
            calls.append(pairs)
            return [0.25 for _ in pairs]

    class AutoModel:
        @staticmethod
        def from_pretrained(name, trust_remote_code):
            assert trust_remote_code is True
            return FakeModel()

    _install(monkeypatch, "transformers", AutoModelForSequenceClassification=AutoModel)
    from ragproof.scorers.hhem import HHEMScorer

    details = HHEMScorer(batch_size=1).score_batch(CASES)
    assert [d.score for d in details] == [0.25, 0.25]
    assert calls[0] == [("ctx one\n\nctx two", "First claim. Second claim.")]
    assert len(calls) == 2


def test_lettucedetect_adapter_requests_spans(monkeypatch):
    seen = {}

    class FakeDetector:
        def __init__(self, method, model_path):
            assert method == "transformer"

        def predict(self, context, question, answer, output_format, min_confidence):
            seen.update(context=context, question=question, output_format=output_format)
            return [{"start": 0, "end": 5, "confidence": 0.9, "text": answer[:5]}]

    _install(monkeypatch, "lettucedetect.models.inference", HallucinationDetector=FakeDetector)
    from ragproof.scorers.lettuce import LettuceDetectScorer

    [first, _] = LettuceDetectScorer().score_batch(CASES)
    assert seen["output_format"] == "spans"
    assert seen["context"] == ["ctx"]
    assert first.score == round(1 - 5 / len(CASES[0].answer), 4)
    assert first.details["hallucinated_spans"] == [{"text": "First", "confidence": 0.9}]


def test_minicheck_adapter_scores_each_sentence(monkeypatch):
    class FakeMiniCheck:
        def __init__(self, model_name, cache_dir):
            pass

        def score(self, docs, claims):
            probs = [0.1 if c.startswith("Second") else 0.9 for c in claims]
            return [int(p > 0.5) for p in probs], probs, None, None

    _install(monkeypatch, "minicheck.minicheck", MiniCheck=FakeMiniCheck)
    from ragproof.scorers.minicheck import MiniCheckScorer

    first, second = MiniCheckScorer().score_batch(CASES)
    assert first.score == 0.1
    assert [s["sentence"] for s in first.details["sentences"]] == ["First claim.", "Second claim."]
    assert second.score == 0.9
