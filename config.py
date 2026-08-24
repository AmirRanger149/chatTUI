from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    gemini_base_url: str | None = Field(default=None, validation_alias="GEMINI_BASE_URL")
    default_model: str = Field(default="gemini-3.5-flash", validation_alias="GEMINI_MODEL")
    sessions_path: Path = Path.home() / ".chat-tui" / "sessions.json"
    temperature: float = 0.7

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        config_path = path or Path.home() / ".chat-tui" / "config.json"
        values: dict[str, Any] = {}
        if config_path.is_file():
            try:
                values = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                values = {}
        if not values:
            for candidate in (Path.cwd() / "gemini.json", Path(__file__).resolve().parent / "gemini.json"):
                if candidate.is_file():
                    try:
                        discovered = cls.from_json(candidate)
                        discovered.save(config_path)
                        return discovered
                    except (OSError, ValueError, TypeError, json.JSONDecodeError):
                        continue
        return cls(**values)

    def save(self, path: Path | None = None) -> None:
        config_path = path or Path.home() / ".chat-tui" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        values = self.model_dump(exclude_none=True)
        values["sessions_path"] = str(self.sessions_path)
        config_path.write_text(json.dumps(values, indent=2), encoding="utf-8")

    @property
    def has_api_key(self) -> bool:
        return bool(self.active_api_key)

    @property
    def active_api_key(self) -> str | None:
        return self.gemini_api_key

    @classmethod
    def from_json(cls, path: Path) -> Settings:
        """Load API settings from a user-selected JSON file."""
        text = path.read_text(encoding="utf-8").strip()
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            raw = {"api_key": text}
        if isinstance(raw, str):
            raw = {"api_key": raw}
        if not isinstance(raw, dict):
            raise ValueError("The JSON root must be an object")
        values = dict(raw)
        if "api_key" in values and "gemini_api_key" not in values:
            values["gemini_api_key"] = values.pop("api_key")
        if "base_url" in values and "gemini_base_url" not in values:
            values["gemini_base_url"] = values.pop("base_url")
        if "model" in values and "default_model" not in values:
            values["default_model"] = values.pop("model")
        allowed = {"gemini_api_key", "gemini_base_url", "default_model", "temperature", "sessions_path", "system_prompt"}
        return cls(**{key: value for key, value in values.items() if key in allowed})
