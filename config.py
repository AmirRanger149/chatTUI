from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")
    default_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")
    sessions_path: Path = Path.home() / ".chat-tui" / "sessions.json"
    temperature: float = 0.7

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        config_path = path or Path.home() / ".chat-tui" / "config.json"
        values: dict[str, Any] = {}
        if config_path.is_file():
            try:
                values = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                values = {}
        return cls(**values)

    @property
    def has_api_key(self) -> bool:
        return bool(self.openai_api_key)
