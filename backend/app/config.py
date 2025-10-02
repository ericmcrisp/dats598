from pydantic import BaseSettings
from pathlib import Path
import os


class Settings(BaseSettings):
    # env config
    ENV: str = "development"
    DEBUG: bool = False

    # api keys
    CLAIMBUSTER_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    WIKIDATA_API_KEY: str | None = None

    # NLP model
    SPACY_MODEL: str = "en_core_web_sm"

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent
    DATA_DIR: Path = BASE_DIR / "data"

    class Config:
        env_file = f".env.{os.getenv('ENV', 'development')}"


# Create a single settings instance to import anywhere
settings = Settings()
