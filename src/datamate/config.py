from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    data_root: Path = Path("D:/zaloaic")
    cache_dir: Path = Path("cache")
    runs_dir: Path = Path("runs")
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "qwen3-4b"
    llm_api_key: str = "EMPTY"
    llm_timeout_s: float = 120.0
    llm_max_context: int = 8192
    llm_max_output_tokens: int = 1024
    llm_prompt_budget_tokens: int = 6000
    llm_cache_enabled: bool = True
    seed: int = 20261116

    @field_validator("cache_dir", "runs_dir", mode="after")
    @classmethod
    def _resolve_project_dir(cls, value: Path) -> Path:
        if not value.is_absolute():
            return PROJECT_ROOT / value
        return value

    def split_dir(self, split: str) -> Path:
        path = self.data_root / split
        if not path.exists():
            raise FileNotFoundError(
                f"Split directory for {split!r} does not exist at {path} (data_root={self.data_root})"
            )
        return path


settings = Settings()
