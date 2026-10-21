import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import chromadb
from chromadb.config import Settings
from backend.config import VECTOR_DB_DIR, METADATA_FILE
from backend.core.embeddings import EmbeddingService

class VectorStoreManager:
    def __init__(self, persist_dir: Path = VECTOR_DB_DIR):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection_name = "research_papers"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedding_service = EmbeddingService()
        self._ensure_metadata_file()

    def _ensure_metadata_file(self):
        if not METADATA_FILE.exists():
            with open(METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def load_all_metadata(self) -> Dict[str, Any]:
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_doc_metadata(self, doc_id: str, metadata: Dict[str, Any]):
        all_meta = self.load_all_metadata()
        all_meta[doc_id] = metadata
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(all_meta, f, indent=2)

    def delete_doc_metadata(self, doc_id: str):
        all_meta = self.load_all_metadata()
        if doc_id in all_meta:
            del all_meta[doc_id]
            with open(METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump(all_meta, f, indent=2)

    def add_chunks(self, chunks: List[Dict[str, Any]], provider: str = "gemini", api_key: Optional[str] = None):
        """
        Computes embeddings and persists chunks into ChromaDB.
        """
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_service.embed_texts(texts, provider=provider, api_key=api_key)

        ids = [c["chunk_id"] for c in chunks]
        metadatas = [
            {
                "doc_id": c["doc_id"],
                "filename": c["filename"],
                "page_number": int(c["page_number"]),
                "chunk_index": int(c["chunk_index"]),
                "section_header": c.get("section_header", "General")
            }
            for c in chunks
        ]

        # Batch upsert to ChromaDB in chunks of 100
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            self.collection.upsert(
                ids=ids[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size],
                documents=texts[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size]
            )

    def search(
        self,
        query: str,
        doc_ids: Optional[List[str]] = None,
        top_k: int = 5,
        hybrid: bool = True,
        alpha: float = 0.7,
        provider: str = "gemini",
        api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid dense + sparse retrieval.
        - alpha: weight for dense vector similarity (1.0 = purely semantic, 0.0 = purely lexical).
        """
        # Step 1: Filter condition
        where_filter = None
        if doc_ids and len(doc_ids) == 1:
            where_filter = {"doc_id": doc_ids[0]}
        elif doc_ids and len(doc_ids) > 1:
            where_filter = {"doc_id": {"$in": doc_ids}}

        # Step 2: Dense retrieval
        query_embedding = self.embedding_service.embed_query(query, provider=provider, api_key=api_key)
        
        # Retrieve candidate pool (e.g. top_k * 3 for hybrid reranking)
        fetch_limit = min(max(top_k * 3, 15), 100)
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=fetch_limit,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"[Error in ChromaDB search]: {e}")
            return []

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        doc_results = []
        raw_ids = results["ids"][0]
        raw_texts = results["documents"][0]
        raw_metas = results["metadatas"][0]
        raw_distances = results["distances"][0]

        # Cosine distance to similarity: similarity = 1 - distance (in [0, 1])
        dense_scores = [max(0.0, 1.0 - dist) for dist in raw_distances]

        # Step 3: Sparse Lexical Scoring (BM25 approximation using term frequencies)
        query_tokens = set(query.lower().split())
        sparse_scores = []
        for text in raw_texts:
            text_lower = text.lower()
            matches = sum(1 for token in query_tokens if token in text_lower)
            # Normalized lexical match ratio
            sparse_score = matches / max(len(query_tokens), 1)
            sparse_scores.append(sparse_score)

        # Step 4: Hybrid Fusion Score
        for i in range(len(raw_ids)):
            d_score = dense_scores[i]
            s_score = sparse_scores[i]
            if hybrid:
                combined_score = (alpha * d_score) + ((1.0 - alpha) * s_score)
            else:
                combined_score = d_score

            meta = raw_metas[i]
            doc_results.append({
                "chunk_id": raw_ids[i],
                "doc_id": meta["doc_id"],
                "filename": meta["filename"],
                "page_number": meta["page_number"],
                "chunk_index": meta["chunk_index"],
                "section_header": meta.get("section_header", "General"),
                "text": raw_texts[i],
                "score": round(float(combined_score), 4),
                "dense_score": round(float(d_score), 4),
                "sparse_score": round(float(s_score), 4)
            })

        # Sort by combined score descending
        doc_results.sort(key=lambda x: x["score"], reverse=True)
        return doc_results[:top_k]

    def get_document_chunks(self, doc_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieves all chunks for a specific document ordered by page and chunk index.
        """
        res = self.collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"]
        )
        if not res or not res["ids"]:
            return []

        chunks = []
        for i in range(len(res["ids"])):
            meta = res["metadatas"][i]
            chunks.append({
                "chunk_id": res["ids"][i],
                "doc_id": meta["doc_id"],
                "filename": meta["filename"],
                "page_number": meta["page_number"],
                "chunk_index": meta["chunk_index"],
                "section_header": meta.get("section_header", "General"),
                "text": res["documents"][i]
            })

        chunks.sort(key=lambda x: (x["page_number"], x["chunk_index"]))
        return chunks[:limit]

    def delete_document(self, doc_id: str) -> bool:
        """
        Deletes all chunks and metadata associated with a document ID.
        """
        try:
            self.collection.delete(where={"doc_id": doc_id})
            self.delete_doc_metadata(doc_id)
            return True
        except Exception as e:
            print(f"[Error deleting doc {doc_id}]: {e}")
            return False

    def list_all_documents(self) -> List[Dict[str, Any]]:
        """
        Returns list of all documents with their metadata.
        """
        meta_dict = self.load_all_metadata()
        return list(meta_dict.values())
