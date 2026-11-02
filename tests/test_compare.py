from types import SimpleNamespace

import pytest
from google.genai import errors

from app.ask import AskMeasurement
from app.compare import format_table, run_comparison


def _response(prompt=3, output=4, thoughts=None, text="回答"):
    usage = SimpleNamespace(prompt_token_count=prompt, candidates_token_count=output)
    if thoughts is not None:
        usage.thoughts_token_count = thoughts
    return SimpleNamespace(text=text, usage_metadata=usage)


class _ScriptedModels:
    def __init__(self, actions):
        self._actions = list(actions)

    def generate_content(self, **kwargs):
        action = self._actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


class _ScriptedClient:
    def __init__(self, actions):
        self.models = _ScriptedModels(actions)


def test_measure_records_elapsed_time_and_thought_tokens():
    from app.ask import measure_question

    client = _ScriptedClient([_response(thoughts=9)])
    result = measure_question("質問", "gemini-3.5-flash-lite", client, "LoRA")

    assert result.status == "ok"
    assert result.elapsed_s >= 0
    assert result.prompt_tokens == 3
    assert result.output_tokens == 4
    assert result.thoughts_tokens == 9
    assert result.text == "回答"


def test_measure_marks_missing_token_counts_unknown():
    from app.ask import measure_question

    client = _ScriptedClient([SimpleNamespace(text="回答", usage_metadata=None)])
    result = measure_question("質問", "gemini-3.5-flash-lite", client, "LoRA")
    table = format_table([result])

    assert result.prompt_tokens is None
    assert result.output_tokens is None
    assert result.thoughts_tokens is None
    assert "unknown" in table


def test_comparison_records_rate_limit_and_continues():
    client = _ScriptedClient(
        [
            errors.APIError(429, {"message": "quota"}),
            _response(),
        ]
    )
    results = run_comparison(
        client,
        ("gemini-3.5-flash-lite", "gemini-3.8-flash"),
        (("LoRA", "LoRAとは何ですか？"),),
        interval_s=0,
        sleep=lambda seconds: pytest.fail("interval 0 では待たない"),
    )

    assert [result.status for result in results] == ["429", "ok"]
    assert results[0].prompt_tokens is None
    assert results[1].prompt_tokens == 3


def test_comparison_waits_between_calls():
    sleeps = []
    client = _ScriptedClient([_response(), _response(prompt=8, output=9)])
    results = run_comparison(
        client,
        ("gemini-3.5-flash-lite",),
        (("LoRA", "LoRAとは何ですか？"), ("宿題の3レベル", "3レベルとは？")),
        interval_s=15,
        sleep=sleeps.append,
    )

    assert sleeps == [15]
    assert [result.question_label for result in results] == ["LoRA", "宿題の3レベル"]
    assert results[1].output_tokens == 9


def test_format_table_shows_status_and_elapsed_seconds():
    result = AskMeasurement(
        model="gemini-3.8-flash",
        question_label="LoRA",
        question="LoRAとは何ですか？",
        text=None,
        elapsed_s=1.2,
        prompt_tokens=None,
        output_tokens=None,
        thoughts_tokens=None,
        status="503",
        error_detail="unavailable",
    )

    table = format_table([result])

    assert "| gemini-3.8-flash | LoRA | 1.20 | unknown | unknown | unknown | 503 |" in table


def test_ask_reraises_api_error(monkeypatch):
    from app import ask

    client = _ScriptedClient([errors.APIError(503, {"message": "busy"})])
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")

    with pytest.raises(errors.APIError) as exc_info:
        ask.ask("質問")

    assert exc_info.value.code == 503
    assert exc_info.value.message == "busy"
