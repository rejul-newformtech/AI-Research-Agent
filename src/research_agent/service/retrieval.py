"""Hybrid retrieval service combining BM25 sparse keyword search and ChromaDB dense vector search."""

import pickle
import re
from pathlib import Path
from typing import Any

from google import genai
from rank_bm25 import BM25Plus

from research_agent.core.config import settings
from research_agent.core.logger import get_logger
from research_agent.db.chroma import ChromaService
from research_agent.service.chunking import DocumentChunk

logger = get_logger("research_agent.service.retrieval")


def tokenize_text(text: str) -> list[str]:
    """Simple alphanumeric tokenizer suitable for lexical BM25 matching."""
    return [word.lower() for word in re.findall(r"\b\w+\b", text)]


class BM25Index:
    """In-memory BM25 index with disk persistence for sparse lexical document retrieval."""

    def __init__(self, index_path: Path | str | None = None):
        self.index_path = Path(index_path) if index_path else Path("data/bm25_index.pkl")
        self.doc_ids: list[str] = []
        self.documents: list[str] = []
        self.metadatas: list[dict[str, Any]] = []
        self.tokenized_corpus: list[list[str]] = []
        self.model: BM25Plus | None = None

        # Attempt to load existing index if present
        self._load()

    def is_empty(self) -> bool:
        """Check whether the BM25 index contains any documents."""
        return self.model is None or len(self.doc_ids) == 0

    def add_documents(
        self,
        chunks: list[DocumentChunk] | list[dict[str, Any]],
        persist: bool = True,
    ) -> int:
        """Add chunks to the BM25 index and rebuild the model.

        Args:
            chunks: List of DocumentChunk instances or dicts with id, text, and metadata.
            persist: Whether to save the updated index to disk.

        Returns:
            Number of newly added or updated chunks.
        """
        if not chunks:
            return 0

        existing_ids = set(self.doc_ids)
        added_count = 0

        for chunk in chunks:
            if isinstance(chunk, DocumentChunk):
                cid = chunk.id
                ctext = chunk.text
                cmeta = chunk.metadata
            else:
                cid = chunk.get("id", "")
                ctext = chunk.get("text", "")
                cmeta = chunk.get("metadata", {})

            if not cid or not ctext:
                continue

            tokens = tokenize_text(ctext)
            if not tokens:
                continue

            if cid in existing_ids:
                # Update existing chunk in index
                idx = self.doc_ids.index(cid)
                self.documents[idx] = ctext
                self.metadatas[idx] = cmeta
                self.tokenized_corpus[idx] = tokens
            else:
                self.doc_ids.append(cid)
                self.documents.append(ctext)
                self.metadatas.append(cmeta)
                self.tokenized_corpus.append(tokens)
                existing_ids.add(cid)
                added_count += 1

        if self.tokenized_corpus:
            self.model = BM25Plus(self.tokenized_corpus)
            if persist:
                self.save()

        logger.info(
            f"BM25 index updated with {added_count} new chunks. Total: {len(self.doc_ids)}."
        )
        return added_count

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform sparse keyword search using BM25Plus.

        Args:
            query: The user search query.
            top_k: Number of highest-scoring passages to return.
            where: Optional metadata filter dict.

        Returns:
            List of result dicts with id, text, metadata, and sparse_score.
        """
        if self.is_empty() or not self.model:
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        scores = self.model.get_scores(query_tokens)
        query_token_set = set(query_tokens)

        scored_docs: list[tuple[int, float]] = []
        for idx, score in enumerate(scores):
            # Only include documents that match at least one query token
            if not query_token_set.intersection(self.tokenized_corpus[idx]):
                continue

            meta = self.metadatas[idx]
            if where:
                matches_where = all(meta.get(k) == v for k, v in where.items())
                if not matches_where:
                    continue

            scored_docs.append((idx, float(score)))

        # Sort by BM25 score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_indices = scored_docs[:top_k]

        results: list[dict[str, Any]] = []
        for idx, score in top_indices:
            results.append(
                {
                    "id": self.doc_ids[idx],
                    "text": self.documents[idx],
                    "metadata": self.metadatas[idx],
                    "sparse_score": round(score, 4),
                    "retrieval_source": "bm25",
                }
            )

        return results

    def save(self) -> None:
        """Persist the index data to disk."""
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.index_path, "wb") as f:
                pickle.dump(
                    {
                        "doc_ids": self.doc_ids,
                        "documents": self.documents,
                        "metadatas": self.metadatas,
                        "tokenized_corpus": self.tokenized_corpus,
                    },
                    f,
                )
        except Exception as e:
            logger.warning(f"Could not persist BM25 index to {self.index_path}: {e}")

    def _load(self) -> None:
        """Load the index data from disk if available."""
        if self.index_path.exists():
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    self.doc_ids = data.get("doc_ids", [])
                    self.documents = data.get("documents", [])
                    self.metadatas = data.get("metadatas", [])
                    self.tokenized_corpus = data.get("tokenized_corpus", [])
                    if self.tokenized_corpus:
                        self.model = BM25Plus(self.tokenized_corpus)
                logger.info(
                    f"Loaded existing BM25 index with {len(self.doc_ids)} chunks from {self.index_path}."
                )
            except Exception as e:
                logger.warning(f"Failed loading BM25 index from {self.index_path}: {e}")


class ReciprocalRankFusion:
    """Fuses ranked result lists from multiple retrieval systems using Reciprocal Rank Fusion (RRF)."""

    @staticmethod
    def fuse(
        dense_results: list[dict[str, Any]],
        sparse_results: list[dict[str, Any]],
        top_k: int = 5,
        k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Combine dense and sparse search results into a single fused ranking.

        RRF Score Formula:
            RRF(d) = sum(weight_i / (k + rank_i(d))) for each retrieval modality i

        Args:
            dense_results: Results from vector search (sorted by similarity desc).
            sparse_results: Results from BM25 search (sorted by BM25 score desc).
            top_k: Number of fused results to return.
            k: Smoothing constant preventing high-rank saturation (standard default is 60).
            dense_weight: Multiplier weight for dense rankings.
            sparse_weight: Multiplier weight for sparse rankings.

        Returns:
            Merged list of results sorted by reciprocal rank fusion score.
        """
        doc_map: dict[str, dict[str, Any]] = {}
        rrf_scores: dict[str, float] = {}

        # Process dense results
        for rank, item in enumerate(dense_results, start=1):
            doc_id = item["id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "id": doc_id,
                    "text": item.get("text", ""),
                    "metadata": item.get("metadata", {}),
                    "similarity": item.get("similarity", 0.0),
                    "sparse_score": 0.0,
                    "dense_rank": rank,
                    "sparse_rank": None,
                }
            else:
                doc_map[doc_id]["similarity"] = item.get("similarity", 0.0)
                doc_map[doc_id]["dense_rank"] = rank

            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (dense_weight / (k + rank))

        # Process sparse results
        for rank, item in enumerate(sparse_results, start=1):
            doc_id = item["id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "id": doc_id,
                    "text": item.get("text", ""),
                    "metadata": item.get("metadata", {}),
                    "similarity": 0.0,
                    "sparse_score": item.get("sparse_score", 0.0),
                    "dense_rank": None,
                    "sparse_rank": rank,
                }
            else:
                doc_map[doc_id]["sparse_score"] = item.get("sparse_score", 0.0)
                doc_map[doc_id]["sparse_rank"] = rank

            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (sparse_weight / (k + rank))

        # Sort documents by fused RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda did: rrf_scores[did], reverse=True)

        fused_results: list[dict[str, Any]] = []
        for doc_id in sorted_ids[:top_k]:
            item = doc_map[doc_id]
            item["rrf_score"] = round(rrf_scores[doc_id], 6)
            fused_results.append(item)

        return fused_results


class HybridSearchService:
    """Unified retrieval service providing dense, sparse, and hybrid search with RRF re-ranking."""

    def __init__(
        self,
        chroma_service: ChromaService | None = None,
        bm25_index: BM25Index | None = None,
    ):
        self.chroma = chroma_service or ChromaService()
        self.bm25 = bm25_index or BM25Index()

        # Sync BM25 from ChromaDB on startup if BM25 is empty but Chroma has chunks
        self._sync_if_needed()

    def _sync_if_needed(self) -> None:
        """Populate BM25 index from ChromaDB documents if the BM25 index is empty."""
        if self.bm25.is_empty():
            try:
                col = self.chroma.get_collection()
                data = col.get(include=["documents", "metadatas"])
                if data and data.get("ids"):
                    chunks: list[dict[str, Any]] = []
                    for i in range(len(data["ids"])):
                        chunks.append(
                            {
                                "id": data["ids"][i],
                                "text": data["documents"][i] if data.get("documents") else "",
                                "metadata": data["metadatas"][i] if data.get("metadatas") else {},
                            }
                        )
                    if chunks:
                        self.bm25.add_documents(chunks, persist=True)
                        logger.info(
                            f"Synchronized {len(chunks)} chunks from ChromaDB into BM25 index."
                        )
            except Exception as e:
                logger.warning(f"Could not auto-sync ChromaDB to BM25 index: {e}")

    def dense_search(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform semantic vector similarity search via ChromaDB."""
        try:
            client = genai.Client(api_key=settings.gemini_api_key)
            res = client.models.embed_content(
                model=settings.embedding_model,
                contents=query,
            )
            query_vector = res.embeddings[0].values
            return self.chroma.search(query_embedding=query_vector, top_k=top_k, where=where)
        except Exception as e:
            logger.error(f"Error executing dense search: {e}")
            return []

    def sparse_search(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform lexical keyword search via BM25."""
        return self.bm25.search(query=query, top_k=top_k, where=where)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute hybrid search combining dense vector search and BM25 with Reciprocal Rank Fusion.

        Args:
            query: The user search prompt or question.
            top_k: Number of final ranked chunks to return.
            dense_weight: Weight assigned to dense vector ranking (default 1.0).
            sparse_weight: Weight assigned to sparse BM25 ranking (default 1.0).
            where: Optional metadata filter dict.

        Returns:
            Merged list of top_k document chunks sorted by RRF relevance.
        """
        # Fetch a broader candidate pool (top_k * 3) from each modality
        candidate_k = max(top_k * 3, 10)

        dense_results = self.dense_search(query=query, top_k=candidate_k, where=where)
        sparse_results = self.sparse_search(query=query, top_k=candidate_k, where=where)

        # If one modality yielded no results, fallback gracefully to the other
        if not dense_results and sparse_results:
            return sparse_results[:top_k]
        if not sparse_results and dense_results:
            return dense_results[:top_k]
        if not dense_results and not sparse_results:
            return []

        # Fuse rankings using Reciprocal Rank Fusion (RRF)
        fused = ReciprocalRankFusion.fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=top_k,
            k=60,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        return fused
