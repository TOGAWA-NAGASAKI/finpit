"""LLM 客户端：面向 OpenCode Go 通道，严格遵守其客户端规范。

三条硬性约束（违反了会被判为滥用并封禁凭据）：
  1. 使用自己的 User-Agent，而不是 ``python-httpx/x.y`` 或 ``OpenAI/Python`` 这类通用库名；
  2. 每个会话使用**稳定**的 session ID（``x-opencode-session``），而不是每请求随机生成；
  3. 流量形态应像正常的 coding agent，不要高频无意义探测。

此外本模块提供磁盘缓存：同一 (model, prompt, params) 只真正调用一次，
既省钱又让评测可复现。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from finpit.config import Settings

log = logging.getLogger(__name__)

# 可重试的 HTTP 状态：限流与服务端临时故障
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    """LLM 调用失败。``retryable`` 标记是否值得重试。"""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass(slots=True)
class Completion:
    """一次补全的结构化结果。"""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached: bool = False
    latency_s: float = 0.0
    finish_reason: str = ""
    """``stop`` 为正常结束；``length`` 表示被 ``max_tokens`` 截断。

    推理模型（如 deepseek-v4-pro）会把大量 token 花在思维链上，
    ``length`` 截断时 ``content`` 往往为空 —— 这时**不能**把
    ``reasoning_content`` 当成答案，否则结构化输出解析会静默失败。
    """

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached": self.cached,
            "latency_s": round(self.latency_s, 3),
            "finish_reason": self.finish_reason,
        }


class LLMClient:
    """OpenCode Go（OpenAI 兼容）客户端，带磁盘缓存与指数退避。

    用法::

        client = LLMClient(Settings.load())
        out = client.complete("回答两个字：可用")
    """

    def __init__(self, settings: Settings, *, use_cache: bool = True) -> None:
        self.settings = settings
        # 稳定 session：整个客户端实例共用，而非每请求新建
        self.session_id = settings.session_id or f"finpit-{uuid.uuid4()}"
        self.use_cache = use_cache
        self._cache_dir = Path(settings.cache_dir) / "llm"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_at: float = 0.0
        # 统计
        self.calls_made = 0
        self.calls_cached = 0
        self.tokens_used = 0

    # ---------------------------------------------------------------- 节流
    def _throttle(self) -> None:
        """保证两次**真实**请求之间有最小间隔。

        OpenCode Go 会监控流量是否"降级他人体验"。实测教训：在 2 分钟内以
        全新随机 session 连发 26 个请求，会被直接封禁凭据。节流 + 稳定 session
        是避免触发该机制的两个必要条件。
        """
        interval = self.settings.request_interval_s
        if interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if self._last_request_at > 0 and elapsed < interval:
            time.sleep(interval - elapsed)

    def _mark_request(self) -> None:
        self._last_request_at = time.monotonic()

    # ---------------------------------------------------------------- headers
    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
            # 规范要求：标识自己，别用通用库名
            "User-Agent": self.settings.user_agent,
            # 规范要求：每个会话稳定的 session id
            "x-opencode-session": self.session_id,
        }
        if self.settings.org_id:
            headers["x-opencode-org-id"] = self.settings.org_id
        return headers

    # ------------------------------------------------------------------ cache
    def _cache_key(self, model: str, messages: list[dict[str, str]], params: dict[str, Any]) -> str:
        blob = json.dumps(
            {"model": model, "messages": messages, "params": params},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def _cache_read(self, key: str) -> Completion | None:
        path = self._cache_dir / f"{key}.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return Completion(
            text=data["text"],
            model=data["model"],
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            cached=True,
            # 必须一并恢复 finish_reason：否则被截断的缓存项在回放时会丢掉
            # "length" 标记，把"预算不足"误报成"解析失败"，诊断失真。
            finish_reason=data.get("finish_reason", ""),
        )

    def _cache_write(self, key: str, completion: Completion) -> None:
        path = self._cache_dir / f"{key}.json"
        try:
            path.write_text(
                json.dumps(completion.as_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:  # 缓存写失败不该让评测崩掉
            log.warning("缓存写入失败 %s: %s", path, exc)

    # -------------------------------------------------------------- completion
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        use_cache: bool | None = None,
    ) -> Completion:
        """单轮补全。``temperature`` 默认 0 以保证可复现。"""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(
            messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            use_cache=use_cache,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        use_cache: bool | None = None,
    ) -> Completion:
        """多轮补全，带缓存、重试与用量统计。"""
        self.settings.require_llm()
        model = model or self.settings.model
        params = {"max_tokens": max_tokens, "temperature": temperature}
        want_cache = self.use_cache if use_cache is None else use_cache
        key = self._cache_key(model, messages, params)

        if want_cache:
            hit = self._cache_read(key)
            if hit is not None:
                self.calls_cached += 1
                return hit

        url = self.settings.base_url.rstrip("/") + "/chat/completions"
        payload = {"model": model, "messages": messages, **params}

        last_error: LLMError | None = None
        for attempt in range(self.settings.max_retries):
            self._throttle()
            started = time.perf_counter()
            try:
                with httpx.Client(timeout=self.settings.request_timeout) as client:
                    resp = client.post(url, headers=self._headers(), json=payload)
                    self._mark_request()
            except httpx.HTTPError as exc:
                self._mark_request()
                last_error = LLMError(f"网络错误: {exc}", retryable=True)
            else:
                elapsed = time.perf_counter() - started
                if resp.status_code == 200:
                    completion = self._parse(resp.json(), model, elapsed)
                    self.calls_made += 1
                    self.tokens_used += completion.total_tokens
                    if want_cache:
                        self._cache_write(key, completion)
                    return completion

                body = resp.text[:300]
                retryable = resp.status_code in RETRYABLE_STATUS
                last_error = LLMError(
                    f"HTTP {resp.status_code}: {body}",
                    status=resp.status_code,
                    retryable=retryable,
                )
                if not retryable:
                    break

            if attempt < self.settings.max_retries - 1:
                backoff = min(2.0**attempt, 30.0)
                log.warning("LLM 调用失败，%.1fs 后重试：%s", backoff, last_error)
                time.sleep(backoff)

        raise last_error or LLMError("LLM 调用失败且无具体错误")

    @staticmethod
    def _parse(data: dict[str, Any], model: str, elapsed: float) -> Completion:
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"响应缺少 choices: {str(data)[:200]}")
        choice = choices[0]
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason") or ""
        content = (message.get("content") or "").strip()
        reasoning = (message.get("reasoning_content") or "").strip()
        # 优先 content；仅当 content 为空**且未被截断**时，才退回思维链
        # （部分推理模型把最终答案写在 reasoning_content 里）。
        # 被截断时 reasoning_content 只是半截思考过程，当成答案会污染下游。
        if content:
            text = content
        elif reasoning and finish_reason != "length":
            text = reasoning
        else:
            text = ""
        usage = data.get("usage") or {}
        return Completion(
            text=text,
            model=data.get("model", model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_s=elapsed,
            finish_reason=finish_reason,
        )

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict[str, int]:
        return {
            "calls_made": self.calls_made,
            "calls_cached": self.calls_cached,
            "tokens_used": self.tokens_used,
        }


__all__ = ["Completion", "LLMClient", "LLMError", "RETRYABLE_STATUS"]
