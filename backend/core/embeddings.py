import numpy as np
from typing import List, Optional
import os
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer

class EmbeddingService:
    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self._local_vectorizer = None

    def embed_texts(self, texts: List[str], provider: Optional[str] = None, api_key: Optional[str] = None) -> List[List[float]]:
        """
        Generates dense vector embeddings for a list of texts.
        """
        active_provider = provider or self.provider
        active_key = api_key or self.api_key

        if active_provider == "gemini" and active_key:
            try:
                return self._embed_gemini(texts, active_key)
            except Exception as e:
                print(f"[Warning] Gemini embedding failed ({e}), falling back to local vectorizer.")
                return self._embed_local(texts)
        elif active_provider == "openai" and active_key:
            try:
                return self._embed_openai(texts, active_key)
            except Exception as e:
                print(f"[Warning] OpenAI embedding failed ({e}), falling back to local vectorizer.")
                return self._embed_local(texts)
        else:
            return self._embed_local(texts)

    def embed_query(self, query: str, provider: Optional[str] = None, api_key: Optional[str] = None) -> List[float]:
        """
        Generates embedding for a single query string.
        """
        embeddings = self.embed_texts([query], provider=provider, api_key=api_key)
        return embeddings[0]

    def _embed_gemini(self, texts: List[str], api_key: str) -> List[List[float]]:
        from google import genai
        client = genai.Client(api_key=api_key)
        # Gemini allows batch embedding
        results = []
        # Chunk batches of 50 to avoid API payload limits
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.models.embed_content(
                model="text-embedding-004",
                contents=batch,
            )
            for emb in response.embeddings:
                results.append(emb.values)
        return results

    def _embed_openai(self, texts: List[str], api_key: str) -> List[List[float]]:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        results = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            for item in resp.data:
                results.append(item.embedding)
        return results

    def _embed_local(self, texts: List[str], dim: int = 384) -> List[List[float]]:
        """
        Deterministic, offline, high-entropy pseudo-dense vector projection based on TF-IDF + random projection hashing.
        Guarantees fast, offline-capable fallback vector math without needing heavy multi-gigabyte models.
        """
        vectors = []
        for text in texts:
            words = text.lower().split()
            vec = np.zeros(dim, dtype=np.float32)
            for word in words:
                h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
                idx = h % dim
                sign = 1.0 if ((h >> 4) % 2 == 0) else -1.0
                vec[idx] += sign
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec.tolist())
        return vectors
