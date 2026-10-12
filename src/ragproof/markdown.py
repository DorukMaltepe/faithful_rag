"""Render a run as Markdown for GitHub job summaries and PR comments."""

from __future__ import annotations

from typing import Any

from ragproof.gate import GateResult

MAX_FAILING_SHOWN = 10


def render(report: dict[str, Any], gate: GateResult) -> str:
    status = "✅ ragproof passed" if gate.passed else "❌ ragproof failed"
    n_cases = len(report["results"])
    lines = [
        f"## {status}",
        "",
        f"{n_cases} case(s) evaluated, pass threshold `{report['threshold']}`.",
        "",
        "| Scorer | Mean | Δ vs baseline | Min | Pass rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, s in report["summaries"].items():
        d = gate.deltas.get(name)
        delta = f"{d.delta:+.4f}" if d else "n/a"
        lines.append(
            f"| `{name}` | {s['mean']:.4f} | {delta} | {s['min']:.4f} | {s['pass_rate']:.1%} |"
        )

    if gate.reasons:
        lines += ["", "**Why it failed**", ""]
        lines += [f"- {r}" for r in gate.reasons]

    if gate.regressed_cases:
        ids = ", ".join(f"`{c}`" for c in gate.regressed_cases)
        lines += ["", f"**Regressed since baseline:** {ids}"]

    threshold = report["threshold"]
    failing = [
        r for r in report["results"] if any(v["score"] < threshold for v in r["scores"].values())
    ]
    if failing:
        lines += ["", f"<details><summary>{len(failing)} flagged case(s)</summary>", ""]
        for r in failing[:MAX_FAILING_SHOWN]:
            lines += _render_case(r, threshold)
        if len(failing) > MAX_FAILING_SHOWN:
            lines.append(f"_…and {len(failing) - MAX_FAILING_SHOWN} more in the JSON report._")
        lines += ["", "</details>"]

    return "\n".join(lines) + "\n"


def _render_case(result: dict[str, Any], threshold: float) -> list[str]:
    out = ["", f"#### `{result['id']}`"]
    if result.get("question"):
        out.append(f"**Q:** {_inline(result['question'])}")
    out.append(f"**A:** {_inline(result['answer'])}")
    for name, s in result["scores"].items():
        mark = "⚠️" if s["score"] < threshold else "✓"
        out.append(f"- {mark} `{name}`: {s['score']:.4f}{_evidence(s.get('details', {}))}")
    return out


def _evidence(details: dict[str, Any]) -> str:
    if spans := details.get("hallucinated_spans"):
        return " — flagged: " + "; ".join(f"“{_inline(s['text'])}”" for s in spans)
    if sentences := details.get("sentences"):
        weakest = min(sentences, key=lambda s: s["score"])
        return f" — weakest sentence: “{_inline(weakest['sentence'])}”"
    if tokens := details.get("unsupported_tokens"):
        return " — not in context: " + ", ".join(f"`{t}`" for t in tokens[:12])
    return ""


def _inline(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
