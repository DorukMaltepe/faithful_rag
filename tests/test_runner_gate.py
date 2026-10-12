import pytest

from ragproof import gate
from ragproof.markdown import render
from ragproof.runner import evaluate

from .conftest import FixedScorer


def _report(cases, scores, threshold=0.5):
    scorers = [FixedScorer(name, s) for name, s in scores.items()]
    return evaluate(cases, scorers, threshold=threshold).to_dict()


def test_summary_statistics(cases):
    report = _report(cases, {"m": {"a": 0.9, "b": 0.3, "c": 0.6}})
    s = report["summaries"]["m"]
    assert s == {"scorer": "m", "mean": 0.6, "min": 0.3, "pass_rate": 0.6667, "failing": ["b"]}


def test_evaluate_validates_inputs(cases):
    with pytest.raises(ValueError):
        evaluate(cases, [], threshold=0.5)
    with pytest.raises(ValueError):
        evaluate(cases, [FixedScorer("m", {})], threshold=2)


def test_gate_passes_when_all_cases_pass(cases):
    report = _report(cases, {"m": {"a": 0.9, "b": 0.8, "c": 0.7}})
    assert gate.check(report).passed


def test_gate_fails_on_pass_rate(cases):
    report = _report(cases, {"m": {"a": 0.9, "b": 0.3, "c": 0.7}})
    result = gate.check(report, min_pass_rate=0.9)
    assert not result.passed
    assert "pass rate" in result.reasons[0]
    assert gate.check(report, min_pass_rate=0.6).passed


def test_gate_fails_on_regression_vs_baseline(cases):
    baseline = _report(cases, {"m": {"a": 0.9, "b": 0.9, "c": 0.9}})
    current = _report(cases, {"m": {"a": 0.9, "b": 0.4, "c": 0.9}})
    result = gate.check(current, min_pass_rate=0.0, baseline=baseline, max_drop=0.05)
    assert not result.passed
    assert result.deltas["m"].delta == pytest.approx(-0.1667, abs=1e-4)
    assert result.regressed_cases == ["b"]
    assert "dropped" in result.reasons[0]


def test_gate_tolerates_small_drop(cases):
    baseline = _report(cases, {"m": {"a": 0.9, "b": 0.9, "c": 0.9}})
    current = _report(cases, {"m": {"a": 0.88, "b": 0.9, "c": 0.9}})
    assert gate.check(current, baseline=baseline, max_drop=0.02).passed


def test_gate_ignores_scorers_missing_from_baseline(cases):
    baseline = _report(cases, {"m": {"a": 0.9, "b": 0.9, "c": 0.9}})
    current = _report(cases, {"other": {"a": 0.1, "b": 0.9, "c": 0.9}})
    result = gate.check(current, min_pass_rate=0.0, baseline=baseline)
    assert result.passed
    assert result.deltas == {}
    assert result.regressed_cases == []


def test_markdown_lists_flagged_cases(cases):
    baseline = _report(cases, {"m": {"a": 0.9, "b": 0.9, "c": 0.9}})
    current = _report(cases, {"m": {"a": 0.9, "b": 0.2, "c": 0.9}})
    result = gate.check(current, baseline=baseline)
    md = render(current, result)
    assert md.startswith("## ❌ ragproof failed")
    assert "| `m` | 0.6667 | -0.2333 |" in md
    assert "#### `b`" in md
    assert "#### `a`" not in md
    assert "**Regressed since baseline:** `b`" in md
