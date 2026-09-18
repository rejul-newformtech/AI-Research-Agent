"""Advanced retrieval services: HyDE (Hypothetical Document Embeddings), Multi-Query expansion, and Chained RAG Pipeline."""

from typing import Any

from google import genai
from pydantic import BaseModel, Field

from research_agent.core.config import settings
from research_agent.core.logger import get_logger
from research_agent.service.retrieval import HybridSearchService

logger = get_logger("research_agent.service.advanced_retrieval")


class ChainedRAGResult(BaseModel):
    """Result from the 2-call chained retrieval and answer synthesis pipeline."""

    query: str
    hypothetical_document: str | None = None
    expanded_queries: list[str] = Field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    synthesized_answer: str
    total_llm_calls: int = 2


class HyDEService:
    """Hypothetical Document Embeddings (HyDE) retrieval service.

    Bridges the semantic gap between short, interrogative user queries and
    dense, declarative textbook/paper passages by generating a hypothetical answer
    and performing document-to-document vector search.
    """

    def __init__(self, hybrid_search_service: HybridSearchService | None = None):
        self.search_service = hybrid_search_service or HybridSearchService()

    def generate_hypothetical_document(self, query: str) -> str:
        """Use Gemini to generate a realistic, academic passage answering the query."""
        prompt = (
            "You are an expert scientific researcher and academic author.\n"
            "Write a realistic, concise academic passage (1 to 2 paragraphs) that directly answers "
            "the following research query as if it were an excerpt from a peer-reviewed research paper or textbook.\n"
            "Do not include conversational filler, introductory remarks, or meta-comments. "
            "Output only the informative passage.\n\n"
            f"Query: {query}"
        )
        try:
            client = genai.Client(api_key=settings.gemini_api_key)
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
            )
            hypo_doc = response.text.strip() if response and response.text else ""
            logger.info(
                f"Generated HyDE hypothetical document ({len(hypo_doc)} chars) for query: '{query[:50]}...'"
            )
            return hypo_doc
        except Exception as e:
            logger.error(f"Error generating HyDE hypothetical document: {e}")
            return query  # Gracefully fallback to original query

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Generate a hypothetical document and perform dense document-to-document search.

        Returns:
            Tuple of (hypothetical_document_text, retrieved_matches).
        """
        hypo_doc = self.generate_hypothetical_document(query)
        # Search using the hypothetical document text as the semantic query
        matches = self.search_service.dense_search(
            query=hypo_doc,
            top_k=top_k,
            where=where,
        )
        return hypo_doc, matches


class MultiQueryService:
    """Multi-Query expansion service.

    Generates multiple diverse search perspectives (varying technical depth, keywords,
    and phrasing) from a single user query to overcome vocabulary mismatches and maximize recall.
    """

    def __init__(self, hybrid_search_service: HybridSearchService | None = None):
        self.search_service = hybrid_search_service or HybridSearchService()

    def generate_query_variations(self, query: str, num_queries: int = 3) -> list[str]:
        """Generate diverse alternative formulations of the user's research query."""
        prompt = (
            "You are an expert AI research assistant.\n"
            f"Given the following research question, generate exactly {num_queries} distinct and specific "
            "search queries to retrieve relevant academic and technical documents.\n"
            "Each query should explore a different angle (e.g. underlying mechanisms/keywords, "
            "high-level conceptual overview, practical application or implications).\n"
            "Output each query on a new line with no numbering, bullet points, or quotation marks.\n\n"
            f"Question: {query}"
        )
        try:
            client = genai.Client(api_key=settings.gemini_api_key)
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
            )
            raw_text = response.text.strip() if response and response.text else ""
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            queries = [line for line in lines if not line.startswith(("#", "-", "*"))][:num_queries]
            if not queries:
                queries = [query]
            logger.info(f"Generated {len(queries)} query variations for: '{query[:50]}...'")
            return queries
        except Exception as e:
            logger.error(f"Error generating multi-query variations: {e}")
            return [query]

    def search(
        self,
        query: str,
        top_k: int = 5,
        num_queries: int = 3,
        where: dict[str, Any] | None = None,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Generate query variations, execute hybrid search on each, and fuse the results via RRF."""
        variations = self.generate_query_variations(query, num_queries=num_queries)
        all_queries = [query] + [q for q in variations if q.lower() != query.lower()]

        # Collect results for all query variants
        candidate_pool: list[list[dict[str, Any]]] = []
        for q in all_queries:
            results = self.search_service.hybrid_search(query=q, top_k=top_k * 2, where=where)
            if results:
                candidate_pool.append(results)

        if not candidate_pool:
            return all_queries, []

        # Multi-list RRF fusion
        fused = self._fuse_multiple_rankings(candidate_pool, top_k=top_k)
        return all_queries, fused

    @staticmethod
    def _fuse_multiple_rankings(
        rankings: list[list[dict[str, Any]]],
        top_k: int = 5,
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """General multi-list Reciprocal Rank Fusion."""
        doc_map: dict[str, dict[str, Any]] = {}
        rrf_scores: dict[str, float] = {}

        for r_list in rankings:
            for rank, item in enumerate(r_list, start=1):
                doc_id = item["id"]
                if doc_id not in doc_map:
                    doc_map[doc_id] = dict(item)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

        sorted_ids = sorted(rrf_scores.keys(), key=lambda did: rrf_scores[did], reverse=True)
        fused: list[dict[str, Any]] = []
        for doc_id in sorted_ids[:top_k]:
            item = doc_map[doc_id]
            item["rrf_score"] = round(rrf_scores[doc_id], 6)
            fused.append(item)
        return fused


class ChainedRAGPipeline:
    """2-call chained RAG pipeline satisfying the specification's chaining requirement:

    Call 1: Advanced query reasoning (HyDE hypothetical passage and/or Multi-Query expansion).
    Retrieval: Multi-perspective hybrid search fused with Reciprocal Rank Fusion (RRF).
    Call 2: Grounded answer synthesis conditioned on retrieved passages with citations.
    """

    def __init__(
        self,
        hybrid_search_service: HybridSearchService | None = None,
        hyde_service: HyDEService | None = None,
        multi_query_service: MultiQueryService | None = None,
    ):
        self.search_service = hybrid_search_service or HybridSearchService()
        self.hyde = hyde_service or HyDEService(self.search_service)
        self.multi_query = multi_query_service or MultiQueryService(self.search_service)

    def run(
        self,
        query: str,
        top_k: int = 5,
        use_hyde: bool = True,
        use_multiquery: bool = True,
        where: dict[str, Any] | None = None,
    ) -> ChainedRAGResult:
        """Execute the full 2-call chained research pipeline."""
        hypo_doc: str | None = None
        expanded_queries: list[str] = []
        candidate_lists: list[list[dict[str, Any]]] = []

        # =========================================================================
        # LLM CALL 1: Advanced Retrieval Reasoning (HyDE / Multi-Query)
        # =========================================================================
        # 1. Base hybrid search for original query
        base_results = self.search_service.hybrid_search(query=query, top_k=top_k * 2, where=where)
        if base_results:
            candidate_lists.append(base_results)

        # 2. HyDE search (if enabled)
        if use_hyde:
            hypo_doc, hyde_results = self.hyde.search(query=query, top_k=top_k * 2, where=where)
            if hyde_results:
                candidate_lists.append(hyde_results)

        # 3. Multi-Query expansion (if enabled)
        if use_multiquery:
            queries = self.multi_query.generate_query_variations(query, num_queries=3)
            expanded_queries = queries
            for q in queries:
                q_res = self.search_service.hybrid_search(query=q, top_k=top_k, where=where)
                if q_res:
                    candidate_lists.append(q_res)

        # Fuse all candidate lists via RRF
        if candidate_lists:
            top_chunks = MultiQueryService._fuse_multiple_rankings(candidate_lists, top_k=top_k)
        else:
            top_chunks = []

        # =========================================================================
        # LLM CALL 2: Grounded Answer Synthesis with Source Citations
        # =========================================================================
        synthesized_answer = self._synthesize_answer(query, top_chunks)

        return ChainedRAGResult(
            query=query,
            hypothetical_document=hypo_doc,
            expanded_queries=expanded_queries,
            retrieved_chunks=top_chunks,
            synthesized_answer=synthesized_answer,
            total_llm_calls=2,
        )

    def _synthesize_answer(self, query: str, chunks: list[dict[str, Any]]) -> str:
        """Synthesize a factual, grounded response citing sources and page numbers."""
        if not chunks:
            return (
                f"No relevant research documents or passages were found in the knowledge base "
                f"to answer the question: '{query}'. Please ingest relevant documents first."
            )

        context_blocks: list[str] = []
        for idx, c in enumerate(chunks, 1):
            source = c.get("metadata", {}).get("source", "unknown")
            page = c.get("metadata", {}).get("page_number", -1)
            page_str = f", Page {page}" if page and page != -1 else ""
            text = c.get("text", "").strip()
            context_blocks.append(f"[Excerpt {idx}] (Source: {source}{page_str})\n{text}")

        context_str = "\n\n".join(context_blocks)

        synthesis_prompt = (
            "You are an authoritative AI Research Assistant.\n"
            "Answer the research question below using ONLY the provided document excerpts.\n"
            "Guidelines:\n"
            "1. Be factual, structured, and concise.\n"
            "2. Cite your sources explicitly in the text using [Source: <filename>, Page: <page_number>] "
            "whenever stating a fact from the excerpts.\n"
            "3. If the excerpts do not contain sufficient evidence to answer completely, explicitly note what information is missing.\n"
            "4. Do not speculate or invent facts not present in the excerpts.\n\n"
            f"=== DOCUMENT EXCERPTS ===\n{context_str}\n\n"
            f"=== RESEARCH QUESTION ===\n{query}\n\n"
            "=== ANSWER ==="
        )

        try:
            client = genai.Client(api_key=settings.gemini_api_key)
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=synthesis_prompt,
            )
            return (
                response.text.strip()
                if response and response.text
                else "Failed to generate answer."
            )
        except Exception as e:
            logger.error(f"Error during LLM answer synthesis: {e}")
            return f"Error synthesizing research answer: {str(e)}"
