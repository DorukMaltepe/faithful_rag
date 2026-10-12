"""Command-line interface.

Exit codes: 0 = gate passed, 1 = gate failed, 2 = bad input, 3 = missing scorer dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from ragproof import __version__, gate, markdown
from ragproof.collect import CollectError, collect, load_questions
from ragproof.dataset import DatasetError, load_cases, write_cases
from ragproof.runner import evaluate
from ragproof.scorers import REGISTRY, MissingDependencyError, get_scorer

EXIT_OK, EXIT_FAILED, EXIT_BAD_INPUT, EXIT_MISSING_DEP = 0, 1, 2, 3


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.func(args)
    except (DatasetError, CollectError, ValueError, OSError) as exc:
        print(f"ragproof: error: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT
    except MissingDependencyError as exc:
        print(f"ragproof: error: {exc}", file=sys.stderr)
        return EXIT_MISSING_DEP


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ragproof", description="Faithfulness regression testing for RAG pipelines."
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(required=True, metavar="command")

    ev = sub.add_parser("eval", help="score a dataset and apply the quality gate")
    ev.add_argument("dataset", type=Path, help="JSONL file of cases (contexts + answer)")
    ev.add_argument(
        "-s",
        "--scorer",
        action="append",
        required=True,
        choices=sorted(REGISTRY),
        help="scorer to run; repeat for several",
    )
    ev.add_argument("-o", "--out", type=Path, help="write the JSON report here")
    _add_gate_args(ev)
    ev.set_defaults(func=_cmd_eval)

    cmp_ = sub.add_parser("compare", help="apply the gate to an existing JSON report")
    cmp_.add_argument("report", type=Path, help="JSON report produced by 'ragproof eval'")
    _add_gate_args(cmp_)
    cmp_.set_defaults(func=_cmd_compare)

    col = sub.add_parser("collect", help="query a running RAG service to build a dataset")
    col.add_argument("questions", type=Path, help="JSONL file with 'question' (and 'id')")
    col.add_argument("--endpoint", required=True, help="URL that accepts a POSTed question")
    col.add_argument("-o", "--out", type=Path, required=True, help="output dataset (JSONL)")
    col.add_argument("--question-field", default="question")
    col.add_argument("--answer-field", default="answer", help="dotted path in the response")
    col.add_argument("--contexts-field", default="contexts", help="dotted path in the response")
    col.add_argument("--timeout", type=float, default=60.0)
    col.set_defaults(func=_cmd_collect)

    ls = sub.add_parser("scorers", help="list available scorers")
    ls.set_defaults(func=_cmd_scorers)
    return p


def _add_gate_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("quality gate")
    g.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="a case passes when every scorer gives it at least this (default 0.5)",
    )
    g.add_argument(
        "--min-pass-rate",
        type=float,
        default=1.0,
        help="fail if any scorer's pass rate is below this (default 1.0)",
    )
    g.add_argument("--baseline", type=Path, help="JSON report from the base branch")
    g.add_argument(
        "--max-drop",
        type=float,
        default=0.02,
        help="fail if a scorer's mean drops more than this vs. baseline (default 0.02)",
    )
    g.add_argument("--markdown", type=Path, help="write a Markdown summary here")


def _cmd_eval(args: argparse.Namespace) -> int:
    cases = load_cases(args.dataset)
    scorers = [get_scorer(name) for name in dict.fromkeys(args.scorer)]
    report = evaluate(cases, scorers, threshold=args.threshold).to_dict()
    if args.out:
        args.out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return _apply_gate(report, args)


def _cmd_compare(args: argparse.Namespace) -> int:
    report = _read_json(args.report)
    return _apply_gate(report, args)


def _cmd_collect(args: argparse.Namespace) -> int:
    cases = collect(
        args.endpoint,
        load_questions(args.questions),
        question_field=args.question_field,
        answer_field=args.answer_field,
        contexts_field=args.contexts_field,
        timeout=args.timeout,
    )
    write_cases(cases, args.out)
    print(f"collected {len(cases)} case(s) into {args.out}")
    return EXIT_OK


def _cmd_scorers(_: argparse.Namespace) -> int:
    for name in sorted(REGISTRY):
        print(name)
    return EXIT_OK


def _apply_gate(report: dict[str, Any], args: argparse.Namespace) -> int:
    baseline = _read_json(args.baseline) if args.baseline else None
    result = gate.check(
        report, min_pass_rate=args.min_pass_rate, baseline=baseline, max_drop=args.max_drop
    )
    md = markdown.render(report, result)

    if args.markdown:
        args.markdown.write_text(md, encoding="utf-8")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(md)

    for name, s in report["summaries"].items():
        delta = result.deltas.get(name)
        extra = f"  delta={delta.delta:+.4f}" if delta else ""
        print(
            f"{name:<14} mean={s['mean']:.4f}  min={s['min']:.4f}  "
            f"pass_rate={s['pass_rate']:.1%}{extra}"
        )
    for reason in result.reasons:
        print(f"FAIL: {reason}", file=sys.stderr)
    print("ragproof: PASSED" if result.passed else "ragproof: FAILED")
    return EXIT_OK if result.passed else EXIT_FAILED


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON report ({exc.msg})") from exc


if __name__ == "__main__":
    raise SystemExit(main())
