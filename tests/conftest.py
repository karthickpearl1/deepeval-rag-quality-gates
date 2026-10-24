"""Shared pytest fixtures, plus a small recorder that saves every metric score to results/.

assert_test() stops at the first failing case's error message, so the recorder keeps the
full picture (every case, every metric, pass or fail) for the CI artifact and the README.
"""

import json
import os
from pathlib import Path

# Do not send usage telemetry to DeepEval's servers from tests (also set in CI).
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

import pytest  # noqa: E402

from chatbot.bot import get_variant_name  # noqa: E402
from evals.judges import get_judge, get_judge_name  # noqa: E402
from evals.metrics import load_thresholds  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


@pytest.fixture(scope="session")
def judge():
    return get_judge()


@pytest.fixture(scope="session")
def thresholds():
    return load_thresholds()


class ResultsRecorder:
    def __init__(self):
        self.rows = []

    def record(self, case, reply, metrics):
        self.rows.append(
            {
                "id": case["id"],
                "risk": case["risk"],
                "input": case["input"],
                "answer": reply["answer"],
                "passed": all(metric.success for metric in metrics),
                "metrics": [
                    {
                        "name": metric.__name__,
                        "score": metric.score,
                        "threshold": metric.threshold,
                        "passed": bool(metric.success),
                        "reason": metric.reason,
                    }
                    for metric in metrics
                ],
            }
        )

    def save(self):
        if not self.rows:
            return
        out_dir = RESULTS_DIR / get_variant_name()
        out_dir.mkdir(parents=True, exist_ok=True)
        passed = sum(row["passed"] for row in self.rows)
        summary = {
            "variant": get_variant_name(),
            "judge": get_judge_name(),
            "cases": len(self.rows),
            "cases_passed": passed,
            "results": self.rows,
        }
        (out_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (out_dir / "summary.md").write_text(render_summary(summary), encoding="utf-8")


def render_summary(summary):
    """A Markdown table: one row per case, one column per metric (score, or score and a cross)."""
    names = []
    for row in summary["results"]:
        for metric in row["metrics"]:
            if metric["name"] not in names:
                names.append(metric["name"])
    lines = [
        f"### Quality gate: {summary['variant']} bot, {summary['judge']} judge",
        "",
        f"**{summary['cases_passed']} of {summary['cases']} golden cases passed**",
        "",
        "| Case | Risk | " + " | ".join(names) + " |",
        "|---|---|" + "---:|" * len(names),
    ]
    for row in summary["results"]:
        scores = {m["name"]: m for m in row["metrics"]}
        cells = []
        for name in names:
            metric = scores.get(name)
            if metric is None:
                cells.append("")
            else:
                cells.append(f"{metric['score']:.2f}" + ("" if metric["passed"] else " ❌"))
        lines.append(f"| {row['id']} | {row['risk']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="session")
def results():
    recorder = ResultsRecorder()
    yield recorder
    recorder.save()
