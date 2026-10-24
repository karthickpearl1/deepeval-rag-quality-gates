"""The judge is picked with the JUDGE environment variable."""

import pytest
from deepeval.models import OpenAIModel

from evals.judges import get_judge
from evals.offline_judge import OfflineJudge


def test_offline_is_the_default(monkeypatch):
    monkeypatch.delenv("JUDGE", raising=False)
    assert isinstance(get_judge(), OfflineJudge)


def test_openai_judge_is_built_from_env(monkeypatch):
    # A fake key is enough to build the client; nothing is sent until a metric runs.
    monkeypatch.setenv("JUDGE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setenv("JUDGE_MODEL", "gpt-4.1")
    judge = get_judge()
    assert isinstance(judge, OpenAIModel)
    assert judge.get_model_name() == "gpt-4.1"


def test_unknown_judge_is_an_error(monkeypatch):
    monkeypatch.setenv("JUDGE", "my-laptop")
    with pytest.raises(ValueError, match="Unknown JUDGE"):
        get_judge()
