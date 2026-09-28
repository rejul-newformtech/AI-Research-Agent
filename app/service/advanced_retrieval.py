"""Backward-compatible proxy to app.a2a_core.algorithms.advanced_retrieval."""

from app.a2a_core.algorithms.advanced_retrieval import (
    ChainedRAGPipeline,
    ChainedRAGResult,
    HyDEService,
    MultiQueryService,
)

__all__ = [
    "ChainedRAGPipeline",
    "ChainedRAGResult",
    "HyDEService",
    "MultiQueryService",
]
