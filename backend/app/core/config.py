from pydantic_settings import BaseSettings
from pathlib import Path
import os


class Settings(BaseSettings):
    # env config
    ENV: str = "development"
    DEBUG: bool = False

    # connections
    FRONTEND_URL: str = "http://localhost:5173" 
    BACKEND_PORT: int = 8000

    # api keys
    CLAIMBUSTER_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    WIKIDATA_API_KEY: str | None = None

    # NLP model
    SPACY_MODEL: str = "en_core_web_sm"

    # relevant paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    FAISS_INDEX_PATH: str = str(DATA_DIR / "vector_db/faiss_index")
    EMBEDDING_DB_PATH: str = str(DATA_DIR / "embeddings/embeddings.db")

    # embedding to use
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # claim threshold
    CLAIM_CONFIDENCE_THRESHOLD: float = 0.6

    # evidence retrieval
    EVIDENCE_TOP_K: int = 5
    EVIDENCE_MIN_SIMILARITY: float = 0.3

    # verification data
    SUPPORTS_THRESHOLD: float = 0.50

    class Config:
        env_file = f".env.{os.getenv('ENV', 'development')}"
        env_file_encoding = "utf-8"


settings = Settings()
