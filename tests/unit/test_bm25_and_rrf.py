"""Unit tests for BM25 lexical sparse search and Reciprocal Rank Fusion (RRF)."""

import unittest
from pathlib import Path

from app.service.chunking import DocumentChunk
from app.service.retrieval import (
    BM25Index,
    ReciprocalRankFusion,
    tokenize_text,
)


class TestBM25Index(unittest.TestCase):
    """Unit tests for BM25 lexical tokenization, indexing, and keyword scoring."""

    def setUp(self):
        self.temp_index_path = Path("data/test_unit_bm25_index.pkl")
        if self.temp_index_path.exists():
            self.temp_index_path.unlink()
        self.index = BM25Index(index_path=self.temp_index_path)

    def tearDown(self):
        if self.temp_index_path.exists():
            self.temp_index_path.unlink()

    def test_tokenization(self):
        text = "Hello, World! AI-Powered Research Agent: 2026."
        tokens = tokenize_text(text)
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)
        self.assertIn("ai", tokens)
        self.assertIn("powered", tokens)
        self.assertIn("2026", tokens)

    def test_indexing_and_sparse_search(self):
        chunks = [
            DocumentChunk(
                chunk_index=0,
                text="Photosynthesis is the biochemical process by which plants convert sunlight into chemical energy.",
                metadata={"topic": "biology"},
            ),
            DocumentChunk(
                chunk_index=1,
                text="Quantum electrodynamics describes how light and matter interact at relativistic subatomic scales.",
                metadata={"topic": "physics"},
            ),
            DocumentChunk(
                chunk_index=2,
                text="Chlorophyll pigments in chloroplasts absorb blue and red light for plant cellular respiration.",
                metadata={"topic": "biology"},
            ),
        ]

        added = self.index.add_documents(chunks, persist=False)
        self.assertEqual(added, 3)
        self.assertFalse(self.index.is_empty())

        # Exact keyword search for physics
        results = self.index.search(query="quantum electrodynamics", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["metadata"]["topic"], "physics")
        self.assertIn("quantum", results[0]["text"].lower())
        self.assertGreater(results[0]["sparse_score"], 0.0)

        # Keyword search with metadata filter
        bio_results = self.index.search(query="light", top_k=5, where={"topic": "biology"})
        self.assertTrue(all(r["metadata"]["topic"] == "biology" for r in bio_results))

    def test_persistence_save_and_reload(self):
        chunks = [
            DocumentChunk(
                chunk_index=0,
                text="Persistence test chunk for BM25 serialization.",
                metadata={"source": "test"},
            )
        ]
        self.index.add_documents(chunks, persist=True)
        self.assertTrue(self.temp_index_path.exists())

        # Load into new instance
        reloaded_index = BM25Index(index_path=self.temp_index_path)
        self.assertEqual(len(reloaded_index.doc_ids), 1)
        res = reloaded_index.search(query="serialization", top_k=1)
        self.assertEqual(len(res), 1)
        self.assertIn("serialization", res[0]["text"])


class TestReciprocalRankFusion(unittest.TestCase):
    """Unit tests for Reciprocal Rank Fusion (RRF) formula and re-ranking."""

    def test_rrf_fuses_and_prioritizes_overlapping_documents(self):
        dense_results = [
            {"id": "doc_a", "text": "Text A", "similarity": 0.95},
            {"id": "doc_b", "text": "Text B", "similarity": 0.88},
        ]
        sparse_results = [
            {"id": "doc_a", "text": "Text A", "sparse_score": 3.4},
            {"id": "doc_c", "text": "Text C", "sparse_score": 2.8},
        ]

        fused = ReciprocalRankFusion.fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=3,
            k=60,
        )

        self.assertEqual(len(fused), 3)
        self.assertEqual(fused[0]["id"], "doc_a")
        self.assertGreater(fused[0]["rrf_score"], fused[1]["rrf_score"])
        self.assertEqual(fused[0]["dense_rank"], 1)
        self.assertEqual(fused[0]["sparse_rank"], 1)

    def test_rrf_weighting(self):
        dense_results = [{"id": "doc_dense", "text": "Dense item", "similarity": 0.9}]
        sparse_results = [{"id": "doc_sparse", "text": "Sparse item", "sparse_score": 4.0}]

        fused = ReciprocalRankFusion.fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            dense_weight=0.1,
            sparse_weight=2.0,
            top_k=2,
            k=60,
        )

        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0]["id"], "doc_sparse")


if __name__ == "__main__":
    unittest.main()
