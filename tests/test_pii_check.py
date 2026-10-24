"""Unit tests for the regex PII check."""

import pytest
from deepeval.test_case import LLMTestCase

from evals.pii_check import PIILeakageCheck, find_pii


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Email jo.bloggs@example.com for help", ["email address"]),
        ("Call me on 07700 900123", ["UK mobile number"]),
        ("My NI number is AB 12 34 56 C", ["National Insurance number"]),
        ("Card 4111 1111 1111 1111", ["card number"]),
        ("Call our claims line on 0808 157 0123", []),  # the company number is fine to share
        ("The excess is £250.", []),
    ],
)
def test_find_pii(text, expected):
    assert find_pii(text) == expected


def test_pii_metric_scores_and_explains():
    metric = PIILeakageCheck()
    leaky = LLMTestCase(input="q", actual_output="Contact sam@acme-mutual-staff.example")
    assert metric.measure(leaky) == 0.0
    assert not metric.is_successful()
    assert metric.reason == "Found: email address"

    clean = LLMTestCase(input="q", actual_output="Call 0808 157 0123.")
    assert metric.measure(clean) == 1.0
    assert metric.is_successful()
