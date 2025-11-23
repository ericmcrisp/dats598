from pydantic_settings import BaseSettings
from pathlib import Path
import json
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
    HF_TOKEN: str | None = None

    # NLP model
    SPACY_MODEL: str = "en_core_web_sm"

    # embedding to use
    EMBEDDING_MODEL_COMMON_NAME: str = "mini_L6"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # relevant paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    VECTOR_DB_DIR: Path = DATA_DIR / "vector_db"
    EMBEDDING_DB_PATH: Path = DATA_DIR / "embeddings/embeddings.db"
    FAISS_INDEX_PATH: str = str(VECTOR_DB_DIR / EMBEDDING_MODEL_COMMON_NAME)

    # whether to use rules or llm
    MODE: str = "rules"
    CLAIM_MODE: str = 'advanced'

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


def get_faiss_index_path(embedding_model_common_name: str) -> str:
    # possible models
    models = {'mini_L12': 'all-MiniLM-L12-v2',
                'mini_L6': 'all-MiniLM-L6-v2',
                'paraphase_L6': 'paraphrase-MiniLM-L6-v2',
                'Gemma3': 'tencent/KaLM-Embedding-Gemma3-12B-2511',
                'e5small': 'intfloat/e5-small-v2'}
    base_vector_dir = Path(settings.VECTOR_DB_DIR)
    index_path = base_vector_dir / embedding_model_common_name
    try:
        model_name = models.get(embedding_model_common_name)
    except KeyError:
        raise ValueError(f"Invalid embedding model name: {embedding_model_common_name}")
    return str(index_path), model_name


def load_settings():
    # Path to your untracked secrets file
    keys = Path(__file__).resolve().parent.parent / "private" / "keys.json"

    # get keys.json if it exists
    data = {}
    if keys.exists():
        with open(keys, "r") as f:
            data = json.load(f)

    return Settings(**data)


settings = load_settings()
