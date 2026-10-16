"""LLM 客户端响应解析与节流的测试。

这些测试锁住两个真实踩过的坑：
  1. 推理模型的思维链会耗尽 max_tokens，导致 content 为空；
     此时**不能**把 reasoning_content 当成答案，否则结构化输出会静默失败。
  2. 客户端必须节流，否则高频请求会被服务端判为滥用并封禁凭据。
"""

from __future__ import annotations

import time

import pytest

from finpit.config import Settings
from finpit.llm.client import LLMClient, LLMError


def _response(content: str = "", reasoning: str = "", finish: str = "stop") -> dict:
    message: dict[str, str] = {}
    if content:
        message["content"] = content
    if reasoning:
        message["reasoning_content"] = reasoning
    return {
        "model": "test-model",
        "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


# ------------------------------------------------------------------ 解析
def test_parse_prefers_content():
    result = LLMClient._parse(_response(content='[{"score": 0.5}]', reasoning="思考过程"), "m", 0.1)
    assert result.text == '[{"score": 0.5}]'
    assert result.finish_reason == "stop"
    assert not result.truncated


def test_parse_falls_back_to_reasoning_when_content_empty():
    """未被截断且 content 为空时，答案可能在思维链里 —— 可以退回。"""
    result = LLMClient._parse(_response(reasoning='答案是 [{"score": 0.1}]'), "m", 0.1)
    assert "score" in result.text
    assert not result.truncated


def test_parse_refuses_reasoning_when_truncated():
    """关键防线：被截断时思维链只是半截思考，绝不能当答案。"""
    result = LLMClient._parse(
        _response(reasoning="我们需要回答用户：对12条财经新闻逐条打分。需要输出 JSON…", finish="length"),
        "m",
        0.1,
    )
    assert result.text == ""
    assert result.truncated is True


def test_parse_empty_when_both_missing():
    result = LLMClient._parse(_response(), "m", 0.1)
    assert result.text == ""


def test_parse_raises_on_missing_choices():
    with pytest.raises(LLMError):
        LLMClient._parse({"choices": []}, "m", 0.1)


def test_parse_carries_usage():
    result = LLMClient._parse(_response(content="ok"), "m", 0.25)
    assert result.total_tokens == 15
    assert result.prompt_tokens == 10
    assert result.latency_s == pytest.approx(0.25)


def test_completion_as_dict_includes_finish_reason():
    result = LLMClient._parse(_response(content="ok"), "m", 0.1)
    assert result.as_dict()["finish_reason"] == "stop"


# ------------------------------------------------------------- 缓存与统计
def _settings(tmp_path) -> Settings:
    return Settings(
        api_key="fake",
        base_url="https://example.invalid/v1",
        model="test-model",
        cache_dir=tmp_path,
        request_interval_s=0.0,
    )


def test_cache_roundtrip(tmp_path):
    client = LLMClient(_settings(tmp_path))
    from finpit.llm import Completion

    completion = Completion(text="cached!", model="test-model", total_tokens=3)
    key = client._cache_key("test-model", [{"role": "user", "content": "hi"}], {"max_tokens": 10})
    client._cache_write(key, completion)
    hit = client._cache_read(key)
    assert hit is not None
    assert hit.text == "cached!"
    assert hit.cached is True


def test_cache_restores_finish_reason(tmp_path):
    """缓存回放必须恢复 finish_reason。

    否则被截断的缓存项会丢掉 "length" 标记，把"预算不足"误报成"解析失败"，
    让排查方向跑偏。
    """
    client = LLMClient(_settings(tmp_path))
    from finpit.llm import Completion

    completion = Completion(text="", model="test-model", finish_reason="length")
    key = client._cache_key("test-model", [{"role": "user", "content": "x"}], {"max_tokens": 5})
    client._cache_write(key, completion)
    hit = client._cache_read(key)
    assert hit is not None
    assert hit.finish_reason == "length"
    assert hit.truncated is True


def test_cache_key_differs_by_model(tmp_path):
    """缓存键必须含模型名，否则换模型会读到旧结果（真实踩过的坑）。"""
    client = LLMClient(_settings(tmp_path))
    msgs = [{"role": "user", "content": "hi"}]
    key_a = client._cache_key("model-a", msgs, {"max_tokens": 10})
    key_b = client._cache_key("model-b", msgs, {"max_tokens": 10})
    assert key_a != key_b


# ------------------------------------------------------------------ 节流
def test_throttle_enforces_minimum_interval(tmp_path):
    settings = _settings(tmp_path)
    settings.request_interval_s = 0.25
    client = LLMClient(settings)

    client._throttle()  # 首次调用不等待
    start = time.monotonic()
    client._mark_request()
    client._throttle()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.2


def test_throttle_disabled_when_zero(tmp_path):
    settings = _settings(tmp_path)
    settings.request_interval_s = 0.0
    client = LLMClient(settings)
    client._mark_request()
    start = time.monotonic()
    client._throttle()
    assert time.monotonic() - start < 0.05


# ------------------------------------------------------------ 客户端身份
def test_headers_use_stable_session_and_custom_agent(tmp_path):
    settings = _settings(tmp_path)
    settings.org_id = "org_test"
    settings.user_agent = "finpit-research/0.1"
    client = LLMClient(settings)

    headers = client._headers()
    assert headers["User-Agent"] == "finpit-research/0.1"
    assert headers["x-opencode-session"] == client.session_id
    assert headers["x-opencode-org-id"] == "org_test"

    # 同一客户端多次取头，session 必须稳定不变
    assert client._headers()["x-opencode-session"] == headers["x-opencode-session"]


def test_session_is_stable_across_calls_but_unique_per_client(tmp_path):
    a = LLMClient(_settings(tmp_path))
    b = LLMClient(_settings(tmp_path))
    assert a.session_id == a.session_id
    assert a.session_id != b.session_id


def test_require_llm_raises_when_unconfigured(tmp_path):
    settings = Settings(api_key="", base_url="", cache_dir=tmp_path)
    with pytest.raises(RuntimeError):
        settings.require_llm()
