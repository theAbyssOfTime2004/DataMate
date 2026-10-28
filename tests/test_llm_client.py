from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import httpx
import pytest

from datamate.config import settings
from datamate.llm import LLMClient, LLMError


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
                "choices": [
                    {"message": {"content": '{"a": '}, "finish_reason": "length"}
                ],
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
                "choices": [
                    {"message": {"content": content}, "finish_reason": "length"}
                ],
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
    with pytest.raises(LLMError, match="timed out"):
        broken.chat_json(messages, schema=schema, schema_name="retry_fail")
    assert len(both_failed) == 2
    assert len(broken.usage_log) == 1
    assert broken.usage_log[0].retried is True
    broken.close()


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
