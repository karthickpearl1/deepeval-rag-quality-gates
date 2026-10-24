"""A deliberately small RAG chatbot: the system under test.

How it answers a question:
  1. Retrieve: score every FAQ paragraph by how many words it shares with the question and
     keep the best few (the "retrieval context").
  2. Guardrails: questions asking for financial advice get a fixed, safe reply that points to
     a qualified adviser instead of an answer.
  3. Generate: build the answer from the retrieved sentences that best match the question.

There is no LLM inside the bot. Answers are built from retrieved text, which keeps the bot
deterministic so the tests show the evaluation, not the randomness of a model. In a real
project, generate_answer() is where the LLM call would go and the tests would not change.

Two variants exist so the quality gate has something to catch:
  "good"       the version we want to ship
  "regressed"  the same bot with four realistic mistakes switched on (see VARIANTS)
"""

import os
import re
import sys
from pathlib import Path

from chatbot.tracing import span

CORPUS_DIR = Path(__file__).parent / "corpus"
INTERNAL_DOCS_DIR = Path(__file__).parent / "internal_docs"

# Each flag in a variant simulates a regression that really happens in RAG projects.
VARIANTS = {
    "good": {
        "include_internal_docs": False,  # only customer-facing documents are indexed
        "top_k": 2,  # how many paragraphs are passed to the generator
        "advice_guardrail": True,  # refuse to give financial advice
        "answer_only_from_context": True,  # no extra "helpful" sentences
    },
    "regressed": {
        "include_internal_docs": True,  # an ingestion change indexed staff-only notes
        "top_k": 3,  # a config change passes more, less relevant context
        "advice_guardrail": False,  # the guardrail was lost in a prompt refactor
        "answer_only_from_context": False,  # the prompt no longer says "use only the context"
    },
}

# Sentences the regressed bot adds when it is no longer told to stick to the context.
# They sound plausible, which is exactly why ungrounded answers are dangerous.
UNGROUNDED_ADDITIONS = {
    "claim": "Most claims are paid within 48 hours.",
    "excess": "There is no excess to pay if you have been with us for over a year.",
    "cancel": "Cancelling is always free of charge.",
    "abroad": "Your cover is valid worldwide for up to 365 days.",
}

ADVICE_PATTERNS = [
    r"\bshould i\b",
    r"\bwhich (policy|cover|option|product) is (best|right)\b",
    r"\brecommend\b",
    r"\bworth (it|getting|buying)\b",
]

SAFE_ADVICE_REPLY = (
    "I can't give personal advice or recommend a product. "
    "To compare your options, please speak to one of our qualified advisers on 0808 157 0123."
)

REGRESSED_ADVICE_REPLY = (
    "You should definitely add accidental damage cover. It is the best option for everyone."
)

NO_ANSWER_REPLY = "I'm sorry, I don't have information about that. Please call us on 0808 157 0123."

# Common words that carry no meaning for matching a question to a paragraph.
STOPWORDS = set(
    "a an and are as at be by can do does for from get go have how i if in is it my of on "
    "or our so that the their them then there this to we what when where which who will "
    "with you your me am any much long".split()
)


def get_variant_name():
    """The variant is chosen with the BOT_VARIANT environment variable (default: good)."""
    name = os.environ.get("BOT_VARIANT", "good")
    if name not in VARIANTS:
        raise ValueError(f"Unknown BOT_VARIANT {name!r}. Use one of: {', '.join(VARIANTS)}")
    return name


def words(text):
    """Lower-case content words, with a very rough plural strip ("claims" -> "claim")."""
    result = set()
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word in STOPWORDS:
            continue
        if len(word) > 3 and word.endswith("s"):
            word = word[:-1]
        result.add(word)
    return result


def split_sentences(text):
    """Split a paragraph into sentences on full stops, question marks and exclamation marks."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def load_chunks(include_internal_docs=False):
    """Read the FAQ files and return one chunk (paragraph) per blank-line-separated block."""
    folders = [CORPUS_DIR] + ([INTERNAL_DOCS_DIR] if include_internal_docs else [])
    chunks = []
    for folder in folders:
        for path in sorted(folder.glob("*.md")):
            for block in path.read_text(encoding="utf-8").split("\n\n"):
                block = block.strip()
                if block and not block.startswith("#"):  # skip headings
                    chunks.append({"source": path.name, "text": block})
    return chunks


def retrieve(question, chunks, top_k):
    """Return the top_k chunks that share the most words with the question."""
    question_words = words(question)
    scored = []
    for position, chunk in enumerate(chunks):
        overlap = len(question_words & words(chunk["text"]))
        if overlap > 0:
            # Sort by overlap (highest first), then by position so ties are stable.
            scored.append((-overlap, position, chunk))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [chunk for _, _, chunk in scored[:top_k]]


def is_advice_request(question):
    return any(re.search(pattern, question.lower()) for pattern in ADVICE_PATTERNS)


def generate_answer(question, context_texts, answer_only_from_context):
    """Build the answer from the two context sentences that best match the question."""
    question_words = words(question)
    sentences = [s for text in context_texts for s in split_sentences(text)]
    ranked = sorted(sentences, key=lambda s: -len(question_words & words(s)))
    best = [s for s in ranked[:2] if question_words & words(s)]
    if not best:
        return NO_ANSWER_REPLY
    # Keep the original order of the chosen sentences so the answer reads naturally.
    best.sort(key=sentences.index)
    answer = " ".join(best)

    if not answer_only_from_context:
        for keyword, extra in UNGROUNDED_ADDITIONS.items():
            if keyword in question.lower():
                answer += " " + extra
                break
    return answer


def ask(question, variant=None):
    """Answer a question. Returns the answer and the retrieval context used to produce it."""
    settings = VARIANTS[variant or get_variant_name()]

    # span() records each step in Phoenix when tracing is switched on; otherwise it does nothing.
    with span("ask", "CHAIN", question) as record_ask:
        if is_advice_request(question):
            safe = settings["advice_guardrail"]
            answer = SAFE_ADVICE_REPLY if safe else REGRESSED_ADVICE_REPLY
            record_ask({"output.value": answer, "guardrail.triggered": safe})
            return {"answer": answer, "retrieval_context": []}

        with span("retrieve", "RETRIEVER", question) as record_retrieval:
            chunks = load_chunks(settings["include_internal_docs"])
            retrieved = retrieve(question, chunks, settings["top_k"])
            context_texts = [chunk["text"] for chunk in retrieved]
            for i, chunk in enumerate(retrieved):
                record_retrieval(
                    {
                        f"retrieval.documents.{i}.document.id": chunk["source"],
                        f"retrieval.documents.{i}.document.content": chunk["text"],
                    }
                )

        with span("generate", "LLM", question) as record_generation:
            answer = generate_answer(question, context_texts, settings["answer_only_from_context"])
            record_generation({"output.value": answer})

        record_ask({"output.value": answer})
        return {"answer": answer, "retrieval_context": context_texts}


if __name__ == "__main__":
    # Quick manual check: python -m chatbot.bot "How do I make a claim?"
    reply = ask(" ".join(sys.argv[1:]) or "How do I make a claim?")
    print(reply["answer"])
