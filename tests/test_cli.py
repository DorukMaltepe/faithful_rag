import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ragproof.cli import main


def test_eval_on_examples_fails_gate_and_writes_reports(tmp_path, examples_dir, monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    out, md = tmp_path / "report.json", tmp_path / "report.md"

    code = main(
        [
            "eval",
            str(examples_dir / "cases.jsonl"),
            "-s",
            "lexical",
            "--threshold",
            "0.8",
            "-o",
            str(out),
            "--markdown",
            str(md),
        ]
    )

    assert code == 1
    report = json.loads(out.read_text())
    failing = report["summaries"]["lexical"]["failing"]
    assert set(failing) == {"hallucinated-number", "hallucinated-entity"}
    assert "#### `hallucinated-entity`" in md.read_text()
    assert summary.read_text() == md.read_text()


def test_eval_passes_with_lenient_pass_rate(examples_dir):
    code = main(
        [
            "eval",
            str(examples_dir / "cases.jsonl"),
            "-s",
            "lexical",
            "--threshold",
            "0.8",
            "--min-pass-rate",
            "0.6",
        ]
    )
    assert code == 0


def test_compare_against_baseline(tmp_path, examples_dir):
    report = tmp_path / "r.json"
    main(
        [
            "eval",
            str(examples_dir / "cases.jsonl"),
            "-s",
            "lexical",
            "-o",
            str(report),
            "--min-pass-rate",
            "0",
        ]
    )
    baseline = json.loads(report.read_text())
    baseline["summaries"]["lexical"]["mean"] += 0.5
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(baseline))

    assert main(["compare", str(report), "--min-pass-rate", "0"]) == 0
    assert main(["compare", str(report), "--min-pass-rate", "0", "--baseline", str(base_path)]) == 1


def test_bad_dataset_exit_code(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{oops\n")
    assert main(["eval", str(bad), "-s", "lexical"]) == 2
    assert "invalid JSON" in capsys.readouterr().err


def test_unknown_scorer_is_rejected_by_argparse(examples_dir):
    with pytest.raises(SystemExit) as exc:
        main(["eval", str(examples_dir / "cases.jsonl"), "-s", "nope"])
    assert exc.value.code == 2


class _RagHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        payload = {
            "data": {
                "answer": f"echo: {body['query']}",
                "sources": [{"text": "ctx one"}, "ctx two"],
            }
        }
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


@pytest.fixture
def rag_server():
    server = HTTPServer(("127.0.0.1", 0), _RagHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/ask"
    server.shutdown()


def test_collect_from_http_service(tmp_path, rag_server):
    questions = tmp_path / "q.jsonl"
    questions.write_text('{"id": "q1", "question": "hello?"}\n')
    out = tmp_path / "cases.jsonl"

    code = main(
        [
            "collect",
            str(questions),
            "--endpoint",
            rag_server,
            "-o",
            str(out),
            "--question-field",
            "query",
            "--answer-field",
            "data.answer",
            "--contexts-field",
            "data.sources",
        ]
    )

    assert code == 0
    record = json.loads(out.read_text())
    assert record == {
        "id": "q1",
        "question": "hello?",
        "contexts": ["ctx one", "ctx two"],
        "answer": "echo: hello?",
    }


def test_collect_reports_missing_field(tmp_path, rag_server, capsys):
    questions = tmp_path / "q.jsonl"
    questions.write_text('{"question": "hello?"}\n')
    code = main(
        [
            "collect",
            str(questions),
            "--endpoint",
            rag_server,
            "-o",
            str(tmp_path / "o.jsonl"),
            "--question-field",
            "query",
        ]
    )
    assert code == 2
    assert "no field 'answer'" in capsys.readouterr().err
