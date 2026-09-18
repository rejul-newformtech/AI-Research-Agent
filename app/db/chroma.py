from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from app.core.config import settings
from app.service.chunking import DocumentChunk

DEFAULT_CHROMA_PATH = settings.chroma_persist_dir
DEFAULT_COLLECTION_NAME = settings.chroma_collection_name


class ChromaService:
    """Service wrapping ChromaDB persistent client for document indexing and retrieval."""

    def __init__(self, persist_directory: str | Path | None = None):
        self.persist_directory = Path(persist_directory or DEFAULT_CHROMA_PATH)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))

    def get_collection(self, name: str | None = None) -> Collection:
        """Get or create a Chroma collection configured with cosine similarity distance."""
        col_name = name or DEFAULT_COLLECTION_NAME
        return self.client.get_or_create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> int:
        """Add document chunks with their embeddings and metadata to ChromaDB.

        Args:
            chunks: List of DocumentChunk instances.
            collection_name: Target collection name.

        Returns:
            Number of chunks successfully added.
        """
        if not chunks:
            return 0

        collection = self.get_collection(collection_name)

        ids: list[str] = []
        documents: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            ids.append(chunk.id)
            documents.append(chunk.text)
            if chunk.embedding is not None:
                embeddings.append(chunk.embedding)

            # ChromaDB only permits str, int, float, bool in metadata values
            clean_meta: dict[str, Any] = {
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number if chunk.page_number is not None else -1,
                "char_count": chunk.char_count,
            }
            for k, v in chunk.metadata.items():
                if isinstance(v, str | int | float | bool):
                    clean_meta[k] = v
                elif v is not None:
                    clean_meta[k] = str(v)

            metadatas.append(clean_meta)

        add_kwargs: dict[str, Any] = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }
        # Only pass embeddings if all chunks have embeddings
        if len(embeddings) == len(chunks):
            add_kwargs["embeddings"] = embeddings

        collection.upsert(**add_kwargs)
        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform vector similarity search using a query embedding vector.

        Args:
            query_embedding: Vector representation of the search query.
            top_k: Number of nearest chunks to retrieve.
            collection_name: Target collection name.
            where: Optional metadata filter dict.

        Returns:
            List of result dicts with id, document, metadata, and similarity score.
        """
        collection = self.get_collection(collection_name)
        total_items = collection.count()
        if total_items == 0:
            return []

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, total_items),
        }
        if where:
            query_kwargs["where"] = where

        res = collection.query(**query_kwargs)

        results: list[dict[str, Any]] = []
        if res and res["ids"] and len(res["ids"]) > 0:
            for i in range(len(res["ids"][0])):
                doc_id = res["ids"][0][i]
                doc_text = res["documents"][0][i] if res.get("documents") else ""
                meta = res["metadatas"][0][i] if res.get("metadatas") else {}
                distance = res["distances"][0][i] if res.get("distances") else 0.0
                # Cosine distance in Chroma: 0 is exact match, 2 is opposite.
                similarity = 1.0 - distance

                results.append(
                    {
                        "id": doc_id,
                        "text": doc_text,
                        "metadata": meta,
                        "distance": distance,
                        "similarity": round(similarity, 4),
                    }
                )

        return results

    def count(self, collection_name: str = DEFAULT_COLLECTION_NAME) -> int:
        """Return total document count in the collection."""
        return self.get_collection(collection_name).count()


# Alias for backward compatibility
VectorStoreService = ChromaService
