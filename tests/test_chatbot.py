"""Unit tests for the chatbot: fast, no judge, no DeepEval. The bottom of the test pyramid."""

import pytest

from chatbot.bot import (
    SAFE_ADVICE_REPLY,
    ask,
    get_variant_name,
    is_advice_request,
    load_chunks,
    retrieve,
    words,
)


def test_words_drops_stopwords_and_plurals():
    assert words("How do I make my claims?") == {"make", "claim"}


def test_retrieval_puts_the_right_paragraph_first():
    chunks = load_chunks()
    top = retrieve("What is the excess for a burst pipe?", chunks, top_k=2)
    assert top[0]["source"] == "home_cover.md"
    assert "£500" in top[0]["text"]


def test_internal_documents_are_not_indexed_for_customers():
    sources = {chunk["source"] for chunk in load_chunks()}
    assert "claims_team_notes.md" not in sources


@pytest.mark.parametrize(
    "question",
    ["Should I add accidental damage cover?", "Which policy is best for me?", "Is it worth it?"],
)
def test_advice_questions_are_detected(question):
    assert is_advice_request(question)


def test_factual_questions_are_not_treated_as_advice():
    assert not is_advice_request("What is the excess on a home claim?")


def test_advice_question_gets_the_safe_reply():
    reply = ask("Should I add accidental damage cover?", variant="good")
    assert reply == {"answer": SAFE_ADVICE_REPLY, "retrieval_context": []}


def test_unknown_topic_does_not_guess():
    reply = ask("Do you sell pet insurance?", variant="good")
    assert "don't have information" in reply["answer"]


def test_answer_is_built_only_from_retrieved_text():
    reply = ask("How long does a claim decision take?", variant="good")
    context = " ".join(reply["retrieval_context"])
    for sentence in reply["answer"].split(". "):
        assert sentence.rstrip(".") in context


def test_variant_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("BOT_VARIANT", "regressed")
    assert get_variant_name() == "regressed"
    monkeypatch.setenv("BOT_VARIANT", "typo")
    with pytest.raises(ValueError, match="Unknown BOT_VARIANT"):
        get_variant_name()
