from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import httpx
import pytest

from datamate.config import settings
from datamate.llm import LLMClient, LLMError, LLMUnavailable


def test_chat_json_request_body_and_disk_cache(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"answer": 42}'}}],
                "usage": {"prompt_tokens": 18, "completion_tokens": 5},
            },
        )

    transport = httpx.MockTransport(handler)
    client = LLMClient(
        base_url="http://mock-server:8000/v1",
        model="qwen3-4b",
        api_key="SECRET_TEST_KEY",
        seed=20261116,
        cache_dir=tmp_path,
        cache_enabled=True,
        max_output_tokens=256,
        transport=transport,
    )

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    messages = [{"role": "user", "content": "Return 42"}]

    res1 = client.chat_json(messages, schema=schema, schema_name="test_ans")
    assert res1 == {"answer": 42}
    assert len(calls) == 1

    sent = calls[0]
    assert sent["temperature"] == 0
    assert sent["seed"] == 20261116
    assert sent["max_tokens"] == 256
    assert sent["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "test_ans",
            "schema": schema,
            "strict": True,
        },
    }

    # Check cache file on disk and ensure API key is never written
    cache_files = list((tmp_path / "llm").rglob("*.json"))
    assert len(cache_files) == 1
    assert "SECRET_TEST_KEY" not in cache_files[0].read_text(encoding="utf-8")

    # Second identical call -> cache hit, zero new HTTP calls
    res2 = client.chat_json(messages, schema=schema, schema_name="test_ans")
    assert res2 == {"answer": 42}
    assert len(calls) == 1

    assert len(client.usage_log) == 2
    assert client.usage_log[0].cache_hit is False
    assert client.usage_log[0].prompt_tokens == 18
    assert client.usage_log[0].completion_tokens == 5
    assert client.usage_log[1].cache_hit is True
    client.close()


def test_chat_json_fallback_to_guided_json_on_400(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        if "response_format" in body:
            return httpx.Response(
                400,
                text='{"error": "Unsupported field response_format in request"}',
            )
        assert "guided_json" in body
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"status": "ok"}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )

    client = LLMClient(
        base_url="http://mock-server:8000/v1",
        cache_dir=tmp_path,
        transport=httpx.MockTransport(handler),
    )
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string"}},
        "required": ["status"],
    }
    res = client.chat_json(
        [{"role": "user", "content": "ping"}],
        schema=schema,
        schema_name="ping_schema",
    )
    assert res == {"status": "ok"}
    assert len(calls) == 2
    assert "response_format" in calls[0]
    assert "guided_json" in calls[1]
    assert "response_format" not in calls[1]
    client.close()


def test_chat_json_raises_llm_error_on_500(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Out of Memory")

    client = LLMClient(
        base_url="http://mock-server:8000/v1",
        cache_dir=tmp_path,
        transport=httpx.MockTransport(handler),
    )
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    with pytest.raises(LLMError, match="Internal Server Out of Memory"):
        client.chat_json([{"role": "user", "content": "hi"}], schema=schema, schema_name="err")
    client.close()


def test_unreachable_or_5xx_server_is_unavailable_not_a_bad_answer(tmp_path: Path) -> None:
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    messages = [{"role": "user", "content": "hi"}]

    def overloaded(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="busy")

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    def bad_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="prompt too long")

    def stopped_app(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="modal-http: invalid function call")

    def wrong_key(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    for handler in (overloaded, refused, stopped_app, wrong_key):
        client = LLMClient(
            base_url="http://mock-server:8000/v1",
            cache_dir=tmp_path,
            cache_enabled=False,
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(LLMUnavailable):
            client.chat_json(messages, schema=schema, schema_name="down")
        client.close()

    client = LLMClient(
        base_url="http://mock-server:8000/v1",
        cache_dir=tmp_path,
        cache_enabled=False,
        transport=httpx.MockTransport(bad_request),
    )
    with pytest.raises(LLMError) as caught:
        client.chat_json(messages, schema=schema, schema_name="bad")
    assert not isinstance(caught.value, LLMUnavailable)
    client.close()


def test_count_tokens_endpoint_and_fallback(tmp_path: Path) -> None:
    tokenize_calls = 0

    def ok_handler(request: httpx.Request) -> httpx.Response:
        nonlocal tokenize_calls
        assert str(request.url) == "http://mock-server:8000/tokenize"
        tokenize_calls += 1
        return httpx.Response(200, json={"count": 12, "tokens": [1] * 12})

    client_ok = LLMClient(
        base_url="http://mock-server:8000/v1",
        cache_dir=tmp_path,
        transport=httpx.MockTransport(ok_handler),
    )
    assert client_ok.count_tokens("hello world") == 12
    # In-process cache hit on second call
    assert client_ok.count_tokens("hello world") == 12
    assert tokenize_calls == 1
    client_ok.close()

    def fail_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="Not Found")

    client_fallback = LLMClient(
        base_url="http://mock-server:8000/v1",
        cache_dir=tmp_path,
        transport=httpx.MockTransport(fail_handler),
    )
    sample = "Xin chào cửa hàng DataMate!"
    assert client_fallback.count_tokens(sample) == math.ceil(len(sample) / 2.5)
    client_fallback.close()


def test_chat_json_salvages_trailing_whitespace_and_unclosed_object(tmp_path: Path) -> None:
    payloads = [
        '{"a": 1}' + "\n" * 500,
        '{"a": 1' + "\n" * 500,
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        idx = 0 if body["messages"][-1]["content"] == "closed" else 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": payloads[idx]},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 512},
            },
        )

    client = LLMClient(
        base_url="http://mock-server:8000/v1",
        cache_dir=tmp_path,
        cache_enabled=False,
        transport=httpx.MockTransport(handler),
    )
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
    }
    assert client.chat_json(
        [{"role": "user", "content": "closed"}],
        schema=schema,
        schema_name="ws",
    ) == {"a": 1}
    assert client.chat_json(
        [{"role": "user", "content": "open"}],
        schema=schema,
        schema_name="ws",
    ) == {"a": 1}

    def truncated(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"a": '}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 512},
            },
        )

    broken = LLMClient(
        base_url="http://mock-server:8000/v1",
        cache_dir=tmp_path,
        cache_enabled=False,
        transport=httpx.MockTransport(truncated),
    )
    with pytest.raises(LLMError, match="truncated: max_tokens"):
        broken.chat_json(
            [{"role": "user", "content": "cut"}],
            schema=schema,
            schema_name="cut",
        )
    client.close()
    broken.close()


def test_salvaged_answer_missing_optional_key_becomes_null(tmp_path: Path) -> None:
    content = '{"line_amount": "Tổng tiền"' + "\n" * 40

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 512},
            },
        )

    client = LLMClient(
        base_url="http://mock-server:8000/v1",
        cache_dir=tmp_path,
        cache_enabled=False,
        transport=httpx.MockTransport(handler),
    )
    schema = {
        "type": "object",
        "properties": {
            "line_amount": {"type": ["string", "null"]},
            "cost_per_unit": {"type": ["string", "null"]},
        },
        "required": ["line_amount", "cost_per_unit"],
        "additionalProperties": False,
    }
    out = client.chat_json(
        [{"role": "user", "content": "map"}],
        schema=schema,
        schema_name="step_b_order_lines",
        required_keys=["line_amount"],
    )
    assert out == {"line_amount": "Tổng tiền", "cost_per_unit": None}
    with pytest.raises(LLMError, match="missing required keys"):
        client.chat_json(
            [{"role": "user", "content": "map"}],
            schema=schema,
            schema_name="step_b_order_lines",
        )
    client.close()


def test_chat_json_retries_httpx_timeout_once(tmp_path: Path) -> None:
    calls: list[bytes] = []
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    messages = [{"role": "user", "content": "Return 42"}]

    def timeout_then_ok(request: httpx.Request) -> httpx.Response:
        calls.append(request.content)
        if len(calls) == 1:
            raise httpx.ReadTimeout("timed out")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"answer": 42}'}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        )

    client = LLMClient(
        base_url="http://mock-server:8000/v1",
        cache_dir=tmp_path,
        cache_enabled=False,
        transport=httpx.MockTransport(timeout_then_ok),
    )
    assert client.chat_json(messages, schema=schema, schema_name="retry_ok") == {"answer": 42}
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert len(client.usage_log) == 1
    assert client.usage_log[0].retried is True
    assert client.usage_log[0].cache_hit is False
    assert client.usage_log[0].prompt_tokens == 7
    client.close()

    both_failed: list[int] = []

    def always_timeout(request: httpx.Request) -> httpx.Response:
        both_failed.append(1)
        raise httpx.ReadTimeout("timed out")

    broken = LLMClient(
        base_url="http://mock-server:8000/v1",
        cache_dir=tmp_path,
        cache_enabled=False,
        transport=httpx.MockTransport(always_timeout),
    )
    with pytest.raises(LLMUnavailable, match="timed out"):
        broken.chat_json(messages, schema=schema, schema_name="retry_fail")
    assert len(both_failed) == 2
    assert len(broken.usage_log) == 1
    assert broken.usage_log[0].retried is True
    broken.close()


def test_dead_server_fails_fast_then_one_probe_finds_it_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "llm_breaker_failures", 3)
    monkeypatch.setattr(settings, "llm_breaker_cooldown_s", 3600.0)
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    messages = [{"role": "user", "content": "hi"}]
    hits: list[int] = []
    server_up = False

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(1)
        if not server_up:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"x": 1}'}}], "usage": {}},
        )

    client = LLMClient(
        base_url="http://mock-server:8000/v1",
        cache_dir=tmp_path,
        cache_enabled=False,
        transport=httpx.MockTransport(handler),
    )
    for _ in range(3):
        with pytest.raises(LLMUnavailable):
            client.chat_json(messages, schema=schema, schema_name="down")
    assert len(hits) == 3
    # Marked down: the next calls never reach the server.
    for _ in range(5):
        with pytest.raises(LLMUnavailable, match="marked down"):
            client.chat_json(messages, schema=schema, schema_name="down")
    assert len(hits) == 3

    # Cooldown over and the server is back: one probe goes through and closes the breaker.
    monkeypatch.setattr(settings, "llm_breaker_cooldown_s", 0.0)
    client._down_until = 0.0
    server_up = True
    assert client.chat_json(messages, schema=schema, schema_name="up") == {"x": 1}
    assert client.chat_json(messages, schema=schema, schema_name="up") == {"x": 1}
    assert len(hits) == 5
    client.close()


def test_error_status_counts_as_server_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_breaker_failures", 2)
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    messages = [{"role": "user", "content": "hi"}]
    hits: list[int] = []

    def overloaded(request: httpx.Request) -> httpx.Response:
        hits.append(1)
        return httpx.Response(503, text="busy")

    client = LLMClient(
        base_url="http://mock-server:8000/v1",
        cache_dir=tmp_path,
        cache_enabled=False,
        transport=httpx.MockTransport(overloaded),
    )
    for _ in range(4):
        with pytest.raises(LLMUnavailable, match="HTTP 503"):
            client.chat_json(messages, schema=schema, schema_name="busy")
    assert len(hits) == 4
    client.close()


# Hard-coded BEFORE llm_disable_thinking landed — must stay byte-identical for 4B cache hits.
_BASELINE_CACHE_KEY = "c4be0e8d6abeeac48a6f2d3f9e8c1a140eba5c8b244ff22f13243cf1b6727df8"


def test_chat_json_retries_once_when_server_drops_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("datamate.llm.client._BUSY_RETRY_DELAY_S", 0.0)
    statuses = [408, 200]

    def handler(request: httpx.Request) -> httpx.Response:
        status = statuses.pop(0)
        if status != 200:
            return httpx.Response(status, text="Missing request")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"answer": 7}'}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )

    client = LLMClient(
        base_url="http://mock-server:8000/v1",
        model="qwen3-4b",
        api_key="k",
        cache_dir=tmp_path,
        cache_enabled=False,
        transport=httpx.MockTransport(handler),
    )
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
    }
    assert client.chat_json([{"role": "user", "content": "x"}], schema, "t") == {"answer": 7}
    assert statuses == []


def test_cache_key_stable_without_disable_thinking(tmp_path: Path) -> None:
    client = LLMClient(
        base_url="http://mock-server:8000/v1",
        model="qwen3-4b",
        seed=20261116,
        cache_dir=tmp_path,
        cache_enabled=False,
        max_output_tokens=256,
        disable_thinking=False,
        transport=httpx.MockTransport(lambda r: httpx.Response(500)),
    )
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    messages = [{"role": "user", "content": "Return 42"}]
    key, canonical = client._compute_cache_key(messages, schema, 256)
    assert key == _BASELINE_CACHE_KEY
    assert "chat_template_kwargs" not in canonical
    client.close()


def test_disable_thinking_changes_body_and_cache_key(tmp_path: Path) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"answer": 1}'}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    messages = [{"role": "user", "content": "Return 42"}]

    off = LLMClient(
        base_url="http://mock-server:8000/v1",
        model="qwen3-4b",
        seed=20261116,
        cache_dir=tmp_path / "off",
        cache_enabled=False,
        max_output_tokens=256,
        disable_thinking=False,
        transport=httpx.MockTransport(handler),
    )
    on = LLMClient(
        base_url="http://mock-server:8000/v1",
        model="qwen3-4b",
        seed=20261116,
        cache_dir=tmp_path / "on",
        cache_enabled=False,
        max_output_tokens=256,
        disable_thinking=True,
        transport=httpx.MockTransport(handler),
    )
    key_off, _ = off._compute_cache_key(messages, schema, 256)
    key_on, canonical_on = on._compute_cache_key(messages, schema, 256)
    assert key_off == _BASELINE_CACHE_KEY
    assert key_on != key_off
    assert canonical_on["chat_template_kwargs"] == {"enable_thinking": False}

    assert on.chat_json(messages, schema=schema, schema_name="think_off") == {"answer": 1}
    assert bodies[-1]["chat_template_kwargs"] == {"enable_thinking": False}
    off.close()
    on.close()


def test_disable_thinking_on_guided_json_fallback(tmp_path: Path) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        bodies.append(body)
        if "response_format" in body:
            return httpx.Response(400, text='{"error": "response_format"}')
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"status": "ok"}'}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2},
            },
        )

    client = LLMClient(
        base_url="http://mock-server:8000/v1",
        cache_dir=tmp_path,
        cache_enabled=False,
        disable_thinking=True,
        transport=httpx.MockTransport(handler),
    )
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string"}},
        "required": ["status"],
    }
    assert client.chat_json(
        [{"role": "user", "content": "ping"}],
        schema=schema,
        schema_name="ping",
    ) == {"status": "ok"}
    assert len(bodies) == 2
    assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert bodies[1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "guided_json" in bodies[1]
    client.close()


def test_strips_think_block_before_json_parse(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '<think>x</think>{"answer": 7}',
                        }
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 6},
            },
        )

    client = LLMClient(
        base_url="http://mock-server:8000/v1",
        cache_dir=tmp_path,
        cache_enabled=False,
        transport=httpx.MockTransport(handler),
    )
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
    }
    assert client.chat_json(
        [{"role": "user", "content": "n"}],
        schema=schema,
        schema_name="think",
    ) == {"answer": 7}
    client.close()


def test_live_vllm_smoke(tmp_path: Path) -> None:
    models_url = f"{settings.llm_base_url.rstrip('/')}/models"
    try:
        r = httpx.get(models_url, timeout=2.0)
        if r.status_code != 200:
            pytest.skip(f"vLLM server returned HTTP {r.status_code} at {models_url}")
    except httpx.HTTPError as exc:
        pytest.skip(f"vLLM server not reachable at {models_url}: {exc}")

    client = LLMClient(cache_dir=tmp_path)
    try:
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        out = client.chat_json(
            messages=[
                {
                    "role": "user",
                    "content": "What is 2+3? Return JSON with integer field 'answer'.",
                }
            ],
            schema=schema,
            schema_name="math_smoke",
            max_tokens=128,
        )
        assert out["answer"] == 5
    finally:
        client.close()


def test_chat_choice_probs_renormalise_and_cache(tmp_path: Path) -> None:
    import math

    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content.decode("utf-8")))
        yes_lp = math.log(0.7)
        no_lp = math.log(0.2)
        other_lp = math.log(0.1)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "yes"},
                        "logprobs": {
                            "content": [
                                {
                                    "token": "yes",
                                    "logprob": yes_lp,
                                    "top_logprobs": [
                                        {"token": "yes", "logprob": yes_lp},
                                        {"token": "no", "logprob": no_lp},
                                        {"token": "maybe", "logprob": other_lp},
                                    ],
                                }
                            ]
                        },
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 1},
            },
        )

    transport = httpx.MockTransport(handler)
    client = LLMClient(
        base_url="http://mock-server:8000/v1",
        model="qwen3-4b",
        api_key="SECRET",
        seed=20261116,
        cache_dir=tmp_path,
        cache_enabled=True,
        transport=transport,
    )
    messages = [{"role": "user", "content": "yes or no?"}]
    probs = client.chat_choice_probs(messages, ["yes", "no"])
    assert abs(probs["yes"] - 0.7 / 0.9) < 1e-9
    assert abs(probs["no"] - 0.2 / 0.9) < 1e-9
    assert abs(probs["yes"] + probs["no"] - 1.0) < 1e-9
    assert calls[0]["guided_choice"] == ["yes", "no"]
    assert calls[0]["logprobs"] is True
    assert calls[0]["max_tokens"] == 1

    # Cache hit
    probs2 = client.chat_choice_probs(messages, ["yes", "no"])
    assert probs2 == probs
    assert len(calls) == 1
    assert client.usage_log[1].cache_hit is True
    client.close()


def test_chat_choice_probs_missing_choice_is_zero(tmp_path: Path) -> None:
    import math

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "yes"},
                        "logprobs": {
                            "content": [
                                {
                                    "token": "yes",
                                    "logprob": math.log(1.0),
                                    "top_logprobs": [
                                        {"token": "yes", "logprob": math.log(1.0)},
                                    ],
                                }
                            ]
                        },
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    client = LLMClient(
        base_url="http://mock:8000/v1",
        cache_dir=tmp_path,
        cache_enabled=False,
        transport=httpx.MockTransport(handler),
    )
    probs = client.chat_choice_probs([{"role": "user", "content": "x"}], ["yes", "no"])
    assert probs["yes"] == 1.0
    assert probs["no"] == 0.0
    client.close()
