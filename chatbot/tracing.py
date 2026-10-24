"""Optional tracing of each chatbot step to Arize Phoenix.

Off by default, so the tests need nothing extra. To turn it on:
    pip install -r requirements-tracing.txt
    phoenix serve                                          # or a hosted Phoenix
    export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
    python -m chatbot.bot "How do I make a claim?"

Each question becomes a trace with a "retrieve" span (the paragraphs found) and a "generate"
span (the answer), using the OpenInference attribute names Phoenix understands. That is the
production side of the same quality story: the tests check the golden questions before a
release, traces show what real questions are doing after it.
"""

import os
from contextlib import contextmanager

_tracer = None


def get_tracer():
    """Create the tracer the first time it is needed, only if Phoenix is configured."""
    global _tracer
    if _tracer is None and os.environ.get("PHOENIX_COLLECTOR_ENDPOINT"):
        from phoenix.otel import register  # imported here so Phoenix stays optional

        # batch=False sends each span straight away, which suits a short-lived script.
        # register() reads PHOENIX_COLLECTOR_ENDPOINT itself.
        provider = register(project_name="acme-faq-bot", batch=False, verbose=False)
        _tracer = provider.get_tracer("chatbot")
    return _tracer


@contextmanager
def span(name, kind, input_value):
    """Record one step. `kind` is an OpenInference span kind: CHAIN, RETRIEVER or LLM.

    Yields a function that sets more attributes, e.g. set_attributes({"output.value": ...}).
    When tracing is off it yields a function that does nothing, so callers never need an if.
    """
    tracer = get_tracer()
    if tracer is None:
        yield lambda attributes: None
        return
    attributes = {"openinference.span.kind": kind, "input.value": input_value}
    with tracer.start_as_current_span(name, attributes=attributes) as current:
        yield current.set_attributes
