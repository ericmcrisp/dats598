""" 
Created this class to handle multiple embedding options
"""

from dataclasses import dataclass
from typing import Optional, Literal
from abc import ABC, abstractmethod
import numpy as np

@dataclass
class EmbeddingConfig:
    provider: Literal["huggingface", "openai", "cohere", "voyage", "gemini", "nvidia"]
    model_name: str
    api_key: Optional[str] = None
    batch_size: int = 32

class EmbeddingModel(ABC):
    """Unified interface for all embedding models"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
    
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Returns embeddings as numpy array of shape (n_texts, embedding_dim)"""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query (some models have different query/doc embeddings)"""
        pass


class HuggingFaceEmbedding(EmbeddingModel):
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(config.model_name, trust_remote_code=True)
    
    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=self.config.batch_size)
    
    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode([text])[0]


class GeminiEmbedding(EmbeddingModel):
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        import google.generativeai as genai
        genai.configure(api_key=config.api_key)
        self.model = genai.GenerativeModel(config.model_name)
    
    def embed(self, texts: list[str]) -> np.ndarray:
        import google.generativeai as genai
        result = genai.embed_content(
            model=self.config.model_name,
            content=texts,
            task_type="retrieval_document"
        )
        return np.array(result['embedding'])
    
    def embed_query(self, text: str) -> np.ndarray:
        import google.generativeai as genai
        result = genai.embed_content(
            model=self.config.model_name,
            content=text,
            task_type="retrieval_query"
        )
        return np.array(result['embedding'])


class NvidiaEmbedding(EmbeddingModel):
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=config.api_key
        )
    
    def embed(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            input=texts,
            model=self.config.model_name,
            encoding_format="float"
        )
        return np.array([e.embedding for e in response.data])
    
    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class OpenAIEmbedding(EmbeddingModel):
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        from openai import OpenAI
        self.client = OpenAI(api_key=config.api_key)
    
    def embed(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            input=texts,
            model=self.config.model_name
        )
        return np.array([e.embedding for e in response.data])
    
    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


# Factory to tie it all together
class EmbeddingFactory:
    embeddings_dict = {
        "huggingface": HuggingFaceEmbedding,
        "openai": OpenAIEmbedding,
        "gemini": GeminiEmbedding,
        "nvidia": NvidiaEmbedding,
    }
    
    @classmethod
    def create(cls, config: EmbeddingConfig) -> EmbeddingModel:
        model_class = cls.embeddings_dict.get(config.provider)
        if not model_class:
            raise ValueError(f"Unknown provider: {config.provider}")
        return model_class(config)


if __name__ == "__main__":
    qwen_config = EmbeddingConfig(
        provider="huggingface",
        model_name="Alibaba-NLP/gte-Qwen2-7B-instruct"
    )
    
    gemini_config = EmbeddingConfig(
        provider="gemini",
        model_name="models/text-embedding-004",
        api_key="your-api-key"
    )
    
    nvidia_config = EmbeddingConfig(
        provider="nvidia",
        model_name="nvidia/nv-embedqa-e5-v5",
        api_key="your-nvidia-key"
    )
    model = EmbeddingFactory.create(qwen_config)
    embeddings = model.embed(["Hello world", "Another text"])
    query_embedding = model.embed_query("Search query")
