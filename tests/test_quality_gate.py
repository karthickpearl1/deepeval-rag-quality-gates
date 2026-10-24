"""The quality gate: every golden case must pass every metric for its risk tier.

Run:  pytest tests/test_quality_gate.py                       (good bot, offline judge)
      BOT_VARIANT=regressed pytest tests/test_quality_gate.py  (must FAIL)
      JUDGE=openai pytest tests/test_quality_gate.py           (real judge, needs a key)
"""

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from chatbot.bot import ask
from evals.metrics import build_metrics, load_golden_cases

GOLDEN_CASES = load_golden_cases()


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=[case["id"] for case in GOLDEN_CASES])
def test_golden_case(case, judge, thresholds, results):
    # 1. Ask the chatbot the golden question.
    reply = ask(case["input"])

    # 2. Describe what happened as a DeepEval test case.
    test_case = LLMTestCase(
        input=case["input"],
        actual_output=reply["answer"],
        expected_output=case["expected_output"],
        retrieval_context=reply["retrieval_context"],  # what the bot actually retrieved
        context=case.get("context"),  # what it should have used (ground truth)
    )

    # 3. Score it. assert_test raises an AssertionError if any metric is below its threshold.
    metrics = build_metrics(case, judge, thresholds)
    try:
        assert_test(test_case, metrics, run_async=False)
    finally:
        results.record(case, reply, metrics)  # keep the scores even when the case fails
