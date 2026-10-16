"""运行期配置：从环境变量与 .env 装载，绝不把密钥写进代码或仓库。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ENV_FILE = ".env"


def _parse_env_file(path: Path) -> dict[str, str]:
    """极简 .env 解析：支持 KEY=VALUE、# 注释、跳过空行。"""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass(slots=True)
class Settings:
    """全部运行期设置。真实环境变量优先于 .env 文件。"""

    api_key: str = ""
    base_url: str = "https://opencode.ai/inference/go/openai/v1"
    model: str = "deepseek-v4-pro"
    model_fast: str = "deepseek-v4.1-flash"
    org_id: str = ""

    # OpenCode Go 客户端规范要求：自定义 UA + 稳定 session，否则会被判为滥用。
    user_agent: str = "finpit-research/0.1"
    session_id: str = ""

    github_token: str = ""
    github_owner: str = ""

    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "finpit")
    request_timeout: float = 90.0
    max_retries: int = 4
    request_interval_s: float = 1.5
    """两次真实请求之间的最小间隔（秒）。

    OpenCode Go 的服务条款明确写着 "Traffic is monitored for abuse that
    degrades the experience for other users"。高频调用会被判为滥用并**封禁凭据**，
    因此节流不是可选优化，而是正确性要求。
    """

    @classmethod
    def load(cls, env_file: str | Path | None = DEFAULT_ENV_FILE) -> Settings:
        file_values: dict[str, str] = {}
        if env_file is not None:
            file_values = _parse_env_file(Path(env_file))

        def get(name: str, default: str = "") -> str:
            return os.environ.get(name) or file_values.get(name, default)

        settings = cls(
            api_key=get("OPENCODE_API_KEY") or get("LLM_API_KEY"),
            base_url=get("LLM_BASE_URL", cls.base_url),
            model=get("LLM_MODEL", cls.model),
            model_fast=get("LLM_MODEL_FAST", cls.model_fast),
            org_id=get("OPENCODE_ORG_ID"),
            github_token=get("GITHUB_TOKEN"),
            github_owner=get("GITHUB_OWNER"),
        )
        interval = get("LLM_REQUEST_INTERVAL_S")
        if interval:
            try:
                settings.request_interval_s = float(interval)
            except ValueError:
                pass
        cache_override = get("FINPIT_CACHE_DIR")
        if cache_override:
            settings.cache_dir = Path(cache_override)
        return settings

    @property
    def llm_configured(self) -> bool:
        return bool(self.api_key and self.base_url)

    def require_llm(self) -> None:
        if not self.llm_configured:
            raise RuntimeError(
                "未配置 LLM：请在 .env 中设置 OPENCODE_API_KEY 与 LLM_BASE_URL，"
                "或改用不依赖 LLM 的确定性基线路径。"
            )


__all__ = ["Settings", "DEFAULT_ENV_FILE"]
