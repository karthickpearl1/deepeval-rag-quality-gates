"""Tracing is optional: off by default, and when on it records one span per chatbot step."""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from chatbot import tracing
from chatbot.bot import ask


def test_tracing_is_off_without_phoenix(monkeypatch):
    monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)
    monkeypatch.setattr(tracing, "_tracer", None)
    assert tracing.get_tracer() is None
    assert ask("How do I make a claim?", variant="good")["answer"]  # still works


def test_each_step_is_recorded_as_a_span(monkeypatch):
    # Use an in-memory OpenTelemetry exporter instead of a Phoenix server.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer("test"))

    reply = ask("What is the excess for a burst pipe?", variant="good")

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert set(spans) == {"ask", "retrieve", "generate"}
    assert spans["retrieve"].attributes["openinference.span.kind"] == "RETRIEVER"
    assert spans["retrieve"].attributes["retrieval.documents.0.document.id"] == "home_cover.md"
    assert spans["ask"].attributes["output.value"] == reply["answer"]
