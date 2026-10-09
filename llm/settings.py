from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    provider: str = ""
    api_key: str = ""
    model: str = "openai/gpt-oss-20b"
    groq_api_key: str = Field(
        default="",
        validation_alias="GROQ_API_KEY",
    )

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        extra="ignore",
    )
