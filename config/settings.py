"""
settings.py — Centralised, environment-driven configuration.
All secrets live in .env; this module validates and exposes them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Directories ──────────────────────────────────────────────────────────
    base_dir: Path = Path(__file__).resolve().parent.parent
    output_dir: Path = Field(default=None)  # resolved below
    logs_dir: Path = Field(default=None)

    # ── AI credentials ───────────────────────────────────────────────────────
    anthropic_api_key: str = Field(..., description="Claude API key")
    assemblyai_api_key: str = Field(..., description="AssemblyAI key")

    # ── LLM settings ─────────────────────────────────────────────────────────
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.2

    # ── Publisher credentials ────────────────────────────────────────────────
    youtube_client_secrets_path: Path = Path("config/yt_client_secrets.json")
    youtube_token_path: Path = Path("config/yt_token.json")
    tiktok_access_token: str = ""
    instagram_access_token: str = ""
    instagram_user_id: str = ""

    # ── B-Roll / Assets ──────────────────────────────────────────────────────
    pexels_api_key: str = ""

    # ── Celery / Redis ───────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Video encoding ───────────────────────────────────────────────────────
    output_resolution: tuple[int, int] = (1080, 1920)   # 9:16 Full-HD
    output_fps: int = 30
    output_crf: int = 18   # visually lossless H.264

    # ── Pipeline behaviour ───────────────────────────────────────────────────
    auto_publish: bool = False          # False → manual review mode
    max_clips_per_video: int = 5
    min_clip_virality_score: float = 7.0   # 0-10 scale

    # ── Caption styling ──────────────────────────────────────────────────────
    caption_font: str = "Impact"
    caption_font_size: int = 72
    caption_base_color: str = "#FFFFFF"
    caption_highlight_color: str = "#FFD700"  # gold for punchy words
    caption_number_color: str = "#00FF88"     # green for numbers
    caption_stroke_color: str = "#000000"
    caption_stroke_width: int = 4

    @field_validator("output_dir", mode="before")
    @classmethod
    def _default_output(cls, v, info):
        return v or (info.data.get("base_dir", Path(".")) / "output")

    @field_validator("logs_dir", mode="before")
    @classmethod
    def _default_logs(cls, v, info):
        return v or (info.data.get("base_dir", Path(".")) / "logs")

    def ensure_dirs(self) -> None:
        for d in (self.output_dir, self.logs_dir,
                  self.output_dir / "clips",
                  self.output_dir / "captions",
                  self.output_dir / "assets"):
            d.mkdir(parents=True, exist_ok=True)


# Singleton — import this everywhere
settings = Settings()
settings.ensure_dirs()
