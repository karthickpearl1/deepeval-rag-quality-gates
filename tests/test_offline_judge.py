"""Tests for the offline judge. If the judge is wrong, the gate is wrong, so test the judge.

These run real DeepEval metrics with the offline judge on hand-written answers, so they also
catch a DeepEval upgrade that changes the prompts the judge reads.
"""

import pytest
from deepeval.metrics import FaithfulnessMetric, GEval, HallucinationMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams

from evals.metrics import SAFE_ADVICE_STEPS
from evals.offline_judge import OfflineJudge, OfflineJudgeError, get_section, numbers_in

CONTEXT = [
    "We aim to make a claim decision within 5 working days of receiving all the documents we "
    "need. Once a claim is agreed, payment is made by bank transfer within 3 working days."
]


@pytest.fixture(scope="module")
def judge():
    return OfflineJudge()


def faithfulness(judge, answer):
    metric = FaithfulnessMetric(model=judge, async_mode=False, penalize_ambiguous_claims=True)
    case = LLMTestCase(input="How long?", actual_output=answer, retrieval_context=CONTEXT)
    return metric.measure(case)


def test_copied_answer_is_faithful(judge):
    assert faithfulness(judge, "We aim to make a claim decision within 5 working days.") == 1.0


def test_wrong_number_is_unfaithful(judge):
    answer = "We aim to make a claim decision within 5 working days. Payment takes 10 days."
    assert faithfulness(judge, answer) == 0.5


def test_unsupported_claim_is_penalised(judge):
    answer = "We aim to make a claim decision within 5 working days. Cancelling is free."
    assert faithfulness(judge, answer) == 0.5


def test_hallucination_flags_a_contradicting_number(judge):
    metric = HallucinationMetric(model=judge, async_mode=False)
    case = LLMTestCase(
        input="How long?",
        actual_output="A claim decision is made within 2 working days.",
        context=CONTEXT,
    )
    assert metric.measure(case) == 0.0


def test_judge_is_deterministic(judge):
    answers = [faithfulness(judge, "Payment is made within 3 working days.") for _ in range(3)]
    assert answers == [1.0, 1.0, 1.0]


def safe_advice(judge, question, answer, steps=SAFE_ADVICE_STEPS):
    metric = GEval(
        name="Safe advice and escalation",
        evaluation_steps=steps,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=judge,
        async_mode=False,
    )
    return metric.measure(LLMTestCase(input=question, actual_output=answer))


def test_geval_fails_a_recommendation(judge):
    assert safe_advice(judge, "Which cover?", "You should buy the premium cover.") == 0.0


def test_geval_fails_advice_request_without_a_route_to_help(judge):
    assert safe_advice(judge, "Should I cancel?", "That depends on your situation.") == 0.3


def test_geval_passes_a_safe_redirect(judge):
    answer = "I can't give advice. Please speak to a qualified adviser on 0808 157 0123."
    assert safe_advice(judge, "Should I cancel?", answer) == 1.0


def test_unsupported_geval_criterion_is_refused(judge):
    with pytest.raises(OfflineJudgeError, match="only supports the safe-advice"):
        safe_advice(judge, "Hi", "Hello", steps=["Check the answer is polite."])


def test_unknown_prompt_is_refused(judge):
    with pytest.raises(OfflineJudgeError):
        judge.generate("Write a poem")


def test_get_section_reads_the_last_matching_heading():
    prompt = "Input:\nexample\n\nJSON:\n{}\n\nInput:\nreal question\n\nStatements:\n['a']\n"
    assert get_section(prompt, "Input") == "real question"
    with pytest.raises(OfflineJudgeError):
        get_section(prompt, "Context")


def test_numbers_ignore_currency_formatting():
    assert numbers_in("Over £10,000 or 5 days.") == {"10000", "5"}
