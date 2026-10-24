"""An offline, deterministic stand-in for the LLM judge that DeepEval metrics normally call.

Why this exists
    DeepEval's RAG metrics are "LLM-as-a-judge": each metric sends prompts to a judge model
    (for example: "split this answer into statements", then "is each statement relevant to
    the question?") and calculates the score from the judge's answers. That needs an API key,
    costs money and gives slightly different answers on each run, which is not what you want
    for a pull-request check.

What it does
    It receives exactly the same prompts a real judge would, reads the parts it needs (the
    question, the answer, the contexts) and answers with simple, explainable word-overlap
    rules. DeepEval then does its normal scoring on those answers. So the real DeepEval
    metric code, thresholds and assert_test run end to end, just with a very simple judge.

What it does NOT do
    It does not understand language. Paraphrases, negation ("is" vs "is not") and subtle
    contradictions without numbers are invisible to it. Passing with this judge proves the
    pipeline and the gate work and catches crude regressions (copied text missing, numbers
    that are not in the source, irrelevant chunks). It does not prove answer quality: use a
    real judge model for that (see judges.py and the README).

It is written against the prompt wording of the DeepEval version pinned in requirements.txt.
If DeepEval changes its prompts, this judge raises OfflineJudgeError instead of guessing.
"""

import ast
import re

from deepeval.models import DeepEvalBaseLLM

from chatbot.bot import split_sentences, words

# A line such as "Input:" or "Retrieval Context  (2 documents):" starts a new prompt section.
SECTION_HEADING = re.compile(r"^[A-Z][A-Za-z ()0-9]*:$")
NUMBER = re.compile(r"\d[\d,.]*")

# Phrases used by the "safe advice and escalation" GEval criterion (see metrics.py).
ADVICE_PHRASES = ["you should", "i recommend", "we recommend", "best option", "definitely"]
REDIRECT_PHRASES = ["adviser", "advisor", "0808", "call us", "speak to"]
ADVICE_QUESTION_PHRASES = ["should i", "best for me", "recommend", "worth it"]
PROMISE_PHRASES = ["always", "guarantee", "never"]


class OfflineJudgeError(RuntimeError):
    """Raised when the judge is asked something it has no rule for."""


def numbers_in(text):
    """All numbers in a text, ignoring formatting: "£10,000" -> "10000"."""
    return {n.replace(",", "").rstrip(".") for n in NUMBER.findall(text)}


def support_ratio(claim, source):
    """Share of the claim's content words that also appear in the source text (0 to 1)."""
    claim_words = words(claim)
    if not claim_words:
        return 1.0
    return len(claim_words & words(source)) / len(claim_words)


def get_section(prompt, heading):
    """Return the text under the LAST heading that matches `heading` (a regex), e.g. "Input"."""
    lines = prompt.splitlines()
    starts = [i for i, line in enumerate(lines) if re.fullmatch(heading + r".*:", line.strip())]
    if not starts:
        raise OfflineJudgeError(f"Could not find a '{heading}:' section in the judge prompt")
    body = []
    for line in lines[starts[-1] + 1 :]:
        if SECTION_HEADING.match(line.strip()) or line.strip() == "---":
            break
        body.append(line)
    return "\n".join(body).strip()


def as_list(text):
    """DeepEval writes lists into prompts as Python lists, e.g. "['a', 'b']"."""
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [text]
    return [str(item) for item in value] if isinstance(value, list) else [str(value)]


# ---------------------------------------------------------------------------------------
# One small function per judge question. Each returns a plain dict in the shape the
# DeepEval schema expects; the judge turns it into the schema object.
# ---------------------------------------------------------------------------------------


def extract_sentences(prompt, heading, key):
    """Used for "list the statements / claims / truths in this text"."""
    return {key: split_sentences(get_section(prompt, heading))}


def answer_relevancy_verdicts(prompt):
    # A statement is relevant if it shares at least one content word with the question.
    question = get_section(prompt, "Input")
    verdicts = []
    for statement in as_list(get_section(prompt, "Statements")):
        if words(statement) & words(question):
            verdicts.append({"verdict": "yes"})
        else:
            verdicts.append({"verdict": "no", "reason": "Shares no words with the question."})
    return {"verdicts": verdicts}


def faithfulness_verdicts(prompt):
    # "no" when a claim has a number that is not in the retrieved text (the most common
    # and most harmful hallucination in an FAQ bot); "yes" when at least half of its words
    # are in the retrieved text; otherwise "borderline" (not supported).
    truths = get_section(prompt, "Retrieval Contexts")
    verdicts = []
    for claim in as_list(get_section(prompt, "Claims")):
        if numbers_in(claim) - numbers_in(truths):
            verdicts.append({"verdict": "no", "reason": "Contains a number not in the context."})
        elif support_ratio(claim, truths) >= 0.5:
            verdicts.append({"verdict": "yes"})
        else:
            verdicts.append({"verdict": "borderline", "reason": "Not supported by the context."})
    return {"verdicts": verdicts}


def hallucination_verdicts(prompt):
    # One verdict per ground-truth context. The answer contradicts a context when one of its
    # sentences is about the same thing (30%+ of its words are in the context) but uses a
    # number that appears in none of the contexts.
    contexts = as_list(get_section(prompt, "Contexts"))
    output = get_section(prompt, "Actual Output")
    all_numbers = set().union(*(numbers_in(c) for c in contexts)) if contexts else set()
    verdicts = []
    for context in contexts:
        contradicted = any(
            support_ratio(sentence, context) >= 0.3 and numbers_in(sentence) - all_numbers
            for sentence in split_sentences(output)
        )
        if contradicted:
            verdicts.append({"verdict": "no", "reason": "Uses a number the context does not."})
        else:
            verdicts.append({"verdict": "yes", "reason": "No contradiction found."})
    return {"verdicts": verdicts}


def contextual_relevancy_verdicts(prompt):
    # Split the retrieved chunk into statements; each is relevant if it shares a word with
    # the question.
    question = get_section(prompt, "Input")
    verdicts = []
    for statement in split_sentences(get_section(prompt, "Context")):
        relevant = bool(words(statement) & words(question))
        verdict = {"statement": statement, "verdict": "yes" if relevant else "no"}
        if not relevant:
            verdict["reason"] = "Shares no words with the question."
        verdicts.append(verdict)
    return {"verdicts": verdicts}


def contextual_precision_verdicts(prompt):
    # One verdict per retrieved chunk, in ranked order: was it useful for the expected answer?
    expected = get_section(prompt, "Expected output")
    verdicts = []
    for chunk in as_list(get_section(prompt, "Retrieval Context")):
        useful = support_ratio(expected, chunk) >= 0.3
        verdicts.append(
            {
                "verdict": "yes" if useful else "no",
                "reason": "Overlaps with the expected answer." if useful else "Not needed.",
            }
        )
    return {"verdicts": verdicts}


def contextual_recall_verdicts(prompt):
    # One verdict per sentence of the expected answer: can it be found in what was retrieved?
    retrieved = " ".join(as_list(get_section(prompt, "Retrieval Context")))
    verdicts = []
    for sentence in split_sentences(get_section(prompt, "Expected Output")):
        found = support_ratio(sentence, retrieved) >= 0.6 and not (
            numbers_in(sentence) - numbers_in(retrieved)
        )
        verdicts.append(
            {
                "verdict": "yes" if found else "no",
                "reason": "Found in the retrieval context." if found else "Not retrieved.",
            }
        )
    return {"verdicts": verdicts}


def safe_advice_score(prompt):
    # Only the "safe advice and escalation" criterion from metrics.py is supported offline.
    # Any other GEval criterion needs a real judge, so we refuse rather than invent a score.
    if "qualified adviser" not in prompt:
        raise OfflineJudgeError("The offline judge only supports the safe-advice GEval criterion")
    question = get_section(prompt, "Input").lower()
    answer = get_section(prompt, "Actual Output").lower()
    if any(phrase in answer for phrase in ADVICE_PHRASES):
        return {"score": 0, "reason": "The answer recommends a product or decision."}
    if any(phrase in answer for phrase in PROMISE_PHRASES):
        return {"score": 4, "reason": "The answer makes a promise the customer could rely on."}
    asked_for_advice = any(phrase in question for phrase in ADVICE_QUESTION_PHRASES)
    if asked_for_advice and not any(phrase in answer for phrase in REDIRECT_PHRASES):
        return {"score": 3, "reason": "Advice was requested but the answer gives no route to help."}
    return {"score": 10, "reason": "No advice given, and a route to help where needed."}


# Which function answers which question. DeepEval passes a pydantic schema with each
# prompt; its module tells us the metric and its class name tells us the step.
HANDLERS = {
    ("answer_relevancy", "Statements"): lambda p: extract_sentences(p, "Text", "statements"),
    ("answer_relevancy", "Verdicts"): answer_relevancy_verdicts,
    ("faithfulness", "Truths"): lambda p: extract_sentences(p, "Text", "truths"),
    ("faithfulness", "Claims"): lambda p: extract_sentences(p, "AI Output", "claims"),
    ("faithfulness", "Verdicts"): faithfulness_verdicts,
    ("hallucination", "Verdicts"): hallucination_verdicts,
    ("contextual_relevancy", "ContextualRelevancyVerdicts"): contextual_relevancy_verdicts,
    ("contextual_precision", "Verdicts"): contextual_precision_verdicts,
    ("contextual_recall", "Verdicts"): contextual_recall_verdicts,
    ("g_eval", "ReasonScore"): safe_advice_score,
}


class OfflineJudge(DeepEvalBaseLLM):
    """A DeepEval judge model that answers with the rules above instead of calling an LLM."""

    def __init__(self):
        super().__init__(model="offline-rules-judge")

    def load_model(self):
        return None  # nothing to load: there is no model

    def get_model_name(self):
        return "offline-rules-judge"

    def generate(self, prompt, schema=None, **kwargs):
        if schema is None:
            raise OfflineJudgeError("The offline judge only answers structured (schema) prompts")
        # Every metric's final "explain the score" step just needs some text back.
        if set(schema.model_fields) == {"reason"}:
            return schema(reason="Scored by the offline rules judge; see evals/offline_judge.py.")
        metric = schema.__module__.split(".")[2]  # deepeval.metrics.<metric>.schema
        handler = HANDLERS.get((metric, schema.__name__))
        if handler is None:
            raise OfflineJudgeError(f"No offline rule for {metric}.{schema.__name__}")
        return schema.model_validate(handler(prompt))

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)
