"""Build a dataset by calling a running RAG service over HTTP.

This is how ragproof tests the pipeline itself rather than a frozen snapshot: CI
starts the service (e.g. ``docker compose up``), ``ragproof collect`` sends each
question to it, and ``ragproof eval`` scores what came back.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

from ragproof.models import Case


class CollectError(RuntimeError):
    pass


def load_questions(path: str | Path) -> list[dict[str, str]]:
    questions = []
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            if not isinstance(obj.get("question"), str):
                raise CollectError(f"{path}:{lineno}: 'question' must be a string")
            questions.append({"id": str(obj.get("id", lineno)), "question": obj["question"]})
    return questions


def collect(
    endpoint: str,
    questions: list[dict[str, str]],
    *,
    question_field: str = "question",
    answer_field: str = "answer",
    contexts_field: str = "contexts",
    timeout: float = 60.0,
) -> list[Case]:
    """POST ``{question_field: question}`` to the endpoint for each question.

    Field names may be dotted paths (``data.answer``) for nested responses. Each
    context may be a string or an object with a ``text`` field.
    """
    cases = []
    for q in questions:
        body = _post_json(endpoint, {question_field: q["question"]}, timeout)
        answer = _get_path(body, answer_field)
        contexts = _get_path(body, contexts_field)
        if not isinstance(answer, str):
            raise CollectError(f"{q['id']}: response field {answer_field!r} is not a string")
        if not isinstance(contexts, list):
            raise CollectError(f"{q['id']}: response field {contexts_field!r} is not a list")
        texts = [c["text"] if isinstance(c, dict) else c for c in contexts]
        cases.append(
            Case(
                id=q["id"], question=q["question"], answer=answer, contexts=[str(t) for t in texts]
            )
        )
    return cases


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> Any:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        raise CollectError(f"request to {url} failed: {exc}") from exc


def _get_path(obj: Any, dotted: str) -> Any:
    for key in dotted.split("."):
        if not isinstance(obj, dict) or key not in obj:
            raise CollectError(f"response has no field {dotted!r}")
        obj = obj[key]
    return obj
