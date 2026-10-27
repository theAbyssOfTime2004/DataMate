from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_root: Path = Path("D:/zaloaic")
    cache_dir: Path = Path("cache")
    runs_dir: Path = Path("runs")
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "qwen3-4b"
    llm_api_key: str = "EMPTY"
    llm_timeout_s: float = 120.0
    seed: int = 20261116

    def split_dir(self, split: str) -> Path:
        path = self.data_root / split
        if not path.exists():
            raise FileNotFoundError(
                f"Split directory for {split!r} does not exist at {path} (data_root={self.data_root})"
            )
        return path


settings = Settings()
