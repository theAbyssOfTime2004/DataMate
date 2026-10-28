from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from datamate.config import settings


class LLMError(RuntimeError):
    """Raised when an LLM request fails or returns invalid structured JSON."""


@dataclass
class UsageEntry:
    cache_hit: bool
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    retried: bool = False


def _strip_v1_suffix(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        return trimmed[:-3]
    return trimmed


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_s: float | None = None,
        seed: int | None = None,
        cache_dir: Path | None = None,
        cache_enabled: bool | None = None,
        max_output_tokens: int | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url if base_url is not None else settings.llm_base_url).rstrip("/")
        self.model = model if model is not None else settings.llm_model
        self._api_key = api_key if api_key is not None else settings.llm_api_key
        self.timeout_s = timeout_s if timeout_s is not None else settings.llm_timeout_s
        self.seed = seed if seed is not None else settings.seed
        self.cache_dir = cache_dir if cache_dir is not None else settings.cache_dir
        self.cache_enabled = (
            cache_enabled if cache_enabled is not None else settings.llm_cache_enabled
        )
        self.max_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else settings.llm_max_output_tokens
        )
        self.usage_log: list[UsageEntry] = []
        self._token_count_cache: dict[str, int] = {}

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        self._http = httpx.Client(
            timeout=self.timeout_s,
            headers=headers,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def _compute_cache_key(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        max_tokens: int,
    ) -> tuple[str, dict[str, Any]]:
        canonical_req = {
            "model": self.model,
            "messages": messages,
            "schema": schema,
            "temperature": 0,
            "seed": self.seed,
            "max_tokens": max_tokens,
        }
        dumped = json.dumps(canonical_req, sort_keys=True, ensure_ascii=False)
        key = hashlib.sha256(dumped.encode("utf-8")).hexdigest()
        return key, canonical_req

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / "llm" / key[:2] / f"{key}.json"

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        if not self.cache_enabled:
            return None
        path = self._cache_path(key)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache_atomic(self, key: str, payload: dict[str, Any]) -> None:
        if not self.cache_enabled:
            return
        target = self._cache_path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(target.parent),
            prefix=f".{key[:8]}_",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            tmp_path.replace(target)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _coerce_answer(
        parsed: Any,
        schema: dict[str, Any],
        required_keys: list[str] | None,
    ) -> dict[str, Any]:
        if not isinstance(parsed, dict):
            raise LLMError(f"Expected JSON object from LLM, got {type(parsed).__name__}")
        props = schema.get("properties")
        prop_names = list(props) if isinstance(props, dict) else []
        if required_keys is None:
            raw_required = schema.get("required")
            if isinstance(raw_required, list):
                must = [k for k in raw_required if isinstance(k, str)]
            else:
                must = []
        else:
            must = list(required_keys)
        must_set = set(must)
        filled = dict(parsed)
        for name in prop_names:
            if name not in filled and name not in must_set:
                filled[name] = None
        missing = [k for k in must if k not in filled]
        if missing:
            raise LLMError(f"LLM JSON response missing required keys {missing}: {filled}")
        return filled

    def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int | None = None,
        required_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        resolved_max_tokens = max_tokens if max_tokens is not None else self.max_output_tokens
        key, canonical_req = self._compute_cache_key(messages, schema, resolved_max_tokens)

        t0 = time.perf_counter()
        cached = self._read_cache(key)
        if cached is not None and "response_json" in cached:
            latency_s = time.perf_counter() - t0
            usage = cached.get("usage", {})
            self.usage_log.append(
                UsageEntry(
                    cache_hit=True,
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    latency_s=latency_s,
                )
            )
            return self._coerce_answer(cached["response_json"], schema, required_keys)

        url = f"{self.base_url}/chat/completions"
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "seed": self.seed,
            "max_tokens": resolved_max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                },
            },
        }

        resp, retried = self._post_with_timeout_retry(
            url,
            body,
            started=t0,
            error_prefix=f"HTTP transport error calling {url}",
        )

        if resp.status_code == 400 and "response_format" in resp.text:
            fallback_body = {k: v for k, v in body.items() if k != "response_format"}
            fallback_body["guided_json"] = schema
            resp, fallback_retried = self._post_with_timeout_retry(
                url,
                fallback_body,
                started=t0,
                error_prefix="HTTP transport error on guided_json retry",
            )
            retried = retried or fallback_retried

        if resp.status_code != 200:
            raise LLMError(f"LLM server error (HTTP {resp.status_code}): {resp.text}")

        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected completion payload structure: {resp.text}") from exc

        if not isinstance(content, str):
            raise LLMError(f"Completion content is not a string: {content!r}")

        finish_reason = data["choices"][0].get("finish_reason")
        parsed_obj = _parse_llm_json(
            content,
            finish_reason if isinstance(finish_reason, str) else None,
        )

        validated = self._coerce_answer(parsed_obj, schema, required_keys)

        raw_usage = data.get("usage") or {}
        prompt_tokens = int(raw_usage.get("prompt_tokens", 0))
        completion_tokens = int(raw_usage.get("completion_tokens", 0))
        latency_s = time.perf_counter() - t0

        self.usage_log.append(
            UsageEntry(
                cache_hit=False,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_s=latency_s,
                retried=retried,
            )
        )

        cache_record = {
            "request": canonical_req,
            "schema_name": schema_name,
            "response_json": validated,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }
        self._write_cache_atomic(key, cache_record)
        return validated

    def _post_with_timeout_retry(
        self,
        url: str,
        body: dict[str, Any],
        *,
        started: float,
        error_prefix: str,
    ) -> tuple[httpx.Response, bool]:
        """POST `body` once. On timeout, repeat that same request a single time."""
        try:
            return self._http.post(url, json=body), False
        except httpx.TimeoutException:
            pass
        except httpx.HTTPError as exc:
            raise LLMError(f"{error_prefix}: {exc}") from exc

        try:
            return self._http.post(url, json=body), True
        except httpx.HTTPError as exc:
            self.usage_log.append(
                UsageEntry(
                    cache_hit=False,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_s=time.perf_counter() - started,
                    retried=True,
                )
            )
            raise LLMError(f"{error_prefix}: {exc}") from exc

    def count_tokens(self, text: str) -> int:
        if text in self._token_count_cache:
            return self._token_count_cache[text]

        root_url = _strip_v1_suffix(self.base_url)
        tokenize_url = f"{root_url}/tokenize"
        count: int | None = None
        try:
            resp = self._http.post(
                tokenize_url,
                json={"model": self.model, "prompt": text},
            )
            if resp.status_code == 200:
                payload = resp.json()
                if isinstance(payload.get("count"), int):
                    count = int(payload["count"])
                elif isinstance(payload.get("tokens"), list):
                    count = len(payload["tokens"])
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            count = None

        if count is None:
            count = math.ceil(len(text) / 2.5)

        self._token_count_cache[text] = count
        return count


def _parse_llm_json(content: str, finish_reason: str | None) -> Any:
    """Parse structured output, including a JSON object followed only by whitespace.

    Some models emit a finished object and then keep printing newlines until
    max_tokens, which can also cut off the closing brace.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    stripped = content.lstrip()
    try:
        parsed, end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        parsed = None
    else:
        if stripped[end:].strip() == "":
            return parsed

    if finish_reason == "length":
        closed = content.rstrip() + "}"
        try:
            repaired = json.loads(closed)
        except json.JSONDecodeError:
            repaired = None
        if isinstance(repaired, dict):
            return repaired
        raise LLMError("truncated: max_tokens")

    raise LLMError(f"LLM returned invalid JSON: {content!r}")


def get_client() -> LLMClient:
    return LLMClient()
