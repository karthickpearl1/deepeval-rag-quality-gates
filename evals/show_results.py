"""Print a short pass/fail table from the last quality gate run.

python -m evals.show_results            # good bot
python -m evals.show_results regressed  # regressed bot
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def short_name(metric_name):
    """'Safe advice and escalation [GEval]' -> 'Safe advice and escalation'."""
    return metric_name.replace(" [GEval]", "").replace(" (regex)", "")


def main(variant):
    path = RESULTS_DIR / variant / "results.json"
    if not path.exists():
        sys.exit(f"No results for {variant!r} yet. Run the quality gate first.")
    summary = json.loads(path.read_text(encoding="utf-8"))

    print(f"{summary['variant']} bot, {summary['judge']} judge: ", end="")
    print(f"{summary['cases_passed']}/{summary['cases']} golden cases passed\n")
    for row in summary["results"]:
        failed = [
            f"{short_name(m['name'])} {m['score']:.2f} < {m['threshold']}"
            for m in row["metrics"]
            if not m["passed"]
        ]
        status = "PASS" if row["passed"] else "FAIL"
        print(f"  {status}  {row['id']:<21} {row['risk']:<7} {failed[0] if failed else ''}")
        for more in failed[1:]:  # one failed metric per line keeps the table readable
            print(f"  {'':<4}  {'':<21} {'':<7} {more}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "good")
