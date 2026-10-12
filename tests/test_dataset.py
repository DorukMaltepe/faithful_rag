import json

import pytest

from ragproof.dataset import DatasetError, load_cases, write_cases


def _write(tmp_path, lines):
    p = tmp_path / "cases.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_loads_examples(examples_dir):
    cases = load_cases(examples_dir / "cases.jsonl")
    assert len(cases) == 5
    assert cases[0].id == "faithful-capital"
    assert cases[2].contexts[1].startswith("Embeddings")


def test_single_string_context_and_default_id(tmp_path):
    p = _write(tmp_path, ["# comment", "", json.dumps({"context": "c", "answer": "a"})])
    [case] = load_cases(p)
    assert case.id == "3"
    assert case.contexts == ["c"]
    assert case.question is None


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"contexts": ["c"]}, "'answer'"),
        ({"contexts": [], "answer": "a"}, "'contexts'"),
        ({"contexts": [1], "answer": "a"}, "'contexts'"),
        ({"contexts": ["c"], "answer": "a", "question": 3}, "'question'"),
    ],
)
def test_rejects_invalid_records(tmp_path, record, message):
    p = _write(tmp_path, [json.dumps(record)])
    with pytest.raises(DatasetError, match=message):
        load_cases(p)


def test_reports_line_number_for_bad_json(tmp_path):
    p = _write(tmp_path, [json.dumps({"contexts": ["c"], "answer": "a"}), "{not json"])
    with pytest.raises(DatasetError, match=r"cases.jsonl:2"):
        load_cases(p)


def test_rejects_duplicate_ids(tmp_path):
    rec = json.dumps({"id": "x", "contexts": ["c"], "answer": "a"})
    with pytest.raises(DatasetError, match="duplicate"):
        load_cases(_write(tmp_path, [rec, rec]))


def test_rejects_empty_file(tmp_path):
    with pytest.raises(DatasetError, match="no cases"):
        load_cases(_write(tmp_path, ["# nothing here"]))


def test_round_trip(tmp_path, examples_dir):
    cases = load_cases(examples_dir / "cases.jsonl")
    out = tmp_path / "out.jsonl"
    write_cases(cases, out)
    assert load_cases(out) == cases
