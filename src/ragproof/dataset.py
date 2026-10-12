"""Load and validate evaluation cases from JSON Lines files."""

from __future__ import annotations

import json
from pathlib import Path

from ragproof.models import Case


class DatasetError(ValueError):
    """Raised when a dataset file is malformed. The message points at the offending line."""


def load_cases(path: str | Path) -> list[Case]:
    """Read one case per line. Blank lines and lines starting with ``#`` are ignored.

    Expected fields: ``answer`` (str), ``contexts`` (list of str, or a single str),
    and optionally ``id`` and ``question``. Missing ids default to the line number.
    """
    path = Path(path)
    cases: list[Case] = []
    seen: set[str] = set()

    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            where = f"{path}:{lineno}"
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{where}: invalid JSON ({exc.msg})") from exc
            if not isinstance(obj, dict):
                raise DatasetError(f"{where}: expected a JSON object")

            case = _parse_case(obj, default_id=str(lineno), where=where)
            if case.id in seen:
                raise DatasetError(f"{where}: duplicate case id {case.id!r}")
            seen.add(case.id)
            cases.append(case)

    if not cases:
        raise DatasetError(f"{path}: no cases found")
    return cases


def _parse_case(obj: dict, default_id: str, where: str) -> Case:
    answer = obj.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise DatasetError(f"{where}: 'answer' must be a non-empty string")

    contexts = obj.get("contexts", obj.get("context"))
    if isinstance(contexts, str):
        contexts = [contexts]
    if (
        not isinstance(contexts, list)
        or not contexts
        or not all(isinstance(c, str) for c in contexts)
    ):
        raise DatasetError(f"{where}: 'contexts' must be a non-empty list of strings")

    question = obj.get("question")
    if question is not None and not isinstance(question, str):
        raise DatasetError(f"{where}: 'question' must be a string")

    return Case(
        id=str(obj.get("id", default_id)), contexts=contexts, answer=answer, question=question
    )


def write_cases(cases: list[Case], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for c in cases:
            record = {
                "id": c.id,
                "question": c.question,
                "contexts": c.contexts,
                "answer": c.answer,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
