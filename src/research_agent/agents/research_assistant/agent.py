"""Google ADK Agent definition for Research Agent."""

from google.adk.agents import Agent

from research_agent.core.config import settings
from research_agent.core.logger import get_logger
from research_agent.db.chroma import ChromaService
from research_agent.service.ingestion import IngestionService
from research_agent.service.retrieval import BM25Index, HybridSearchService

logger = get_logger("research_agent.adk")


def search_research_documents(query: str, top_k: int = 5) -> str:
    """Search ingested research documents and papers using Hybrid Search (BM25 lexical + vector dense + RRF re-ranking).

    Args:
        query: The search question or semantic keywords to look up.
        top_k: Number of most relevant passages to retrieve (default 5).

    Returns:
        A string containing relevant document chunks, sources, and similarity scores.
    """
    try:
        service = HybridSearchService()
        matches = service.hybrid_search(query=query, top_k=top_k)

        if not matches:
            return f"No relevant research documents found for query: '{query}'"

        results_str = [
            f"Found {len(matches)} relevant passage(s) via hybrid search for '{query}':\n"
        ]
        for idx, match in enumerate(matches, 1):
            source = match.get("metadata", {}).get("source", "unknown")
            page = match.get("metadata", {}).get("page_number", "")
            strategy = match.get("metadata", {}).get("strategy", "")
            rrf_score = match.get("rrf_score")
            similarity = match.get("similarity", 0.0)
            sparse_score = match.get("sparse_score", 0.0)
            page_info = f" (Page {page})" if page and page != -1 else ""
            score_info = f"RRF Score: {rrf_score}" if rrf_score else f"Similarity: {similarity}"
            results_str.append(
                f"[{idx}] Source: {source}{page_info} | Strategy: {strategy} | {score_info} (Vector: {similarity}, BM25: {sparse_score})\n"
                f"Content:\n{match.get('text', '').strip()}\n"
            )

        return "\n".join(results_str)
    except Exception as e:
        logger.error(f"Error searching research documents: {e}")
        return f"Error executing search: {str(e)}"


def ingest_research_notes(text: str, source_name: str = "agent_notes") -> str:
    """Chunk, embed, and store new research text or findings into the persistent vector database.

    Args:
        text: The text content or research summary to ingest.
        source_name: A short name or identifier for this source (e.g. 'arxiv_notes', 'agent_summary').

    Returns:
        A status message confirming ingestion and the number of chunks stored.
    """
    try:
        service = IngestionService()
        result = service.ingest_text(
            text=text,
            source_name=source_name,
            strategy="both",
            generate_embeddings=True,
        )
        if result.chunks:
            chroma = ChromaService()
            count = chroma.add_chunks(result.chunks)
            bm25 = BM25Index()
            bm25.add_documents(result.chunks, persist=True)
            return f"Successfully ingested '{source_name}': created and stored {count} chunks into ChromaDB."
        return "No chunks generated from provided text."
    except Exception as e:
        logger.error(f"Error ingesting research notes: {e}")
        return f"Error during ingestion: {str(e)}"


def list_stored_documents() -> str:
    """List all research documents (PDFs, text files) available in the local storage directory (data/documents).

    Returns:
        A formatted list of files with their file sizes and indexing status in ChromaDB.
    """
    try:
        doc_dir = settings.document_storage_dir
        if not doc_dir.exists():
            return f"Document storage directory '{doc_dir}' does not exist."

        files = [f for f in doc_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
        if not files:
            return f"No documents found in '{doc_dir}'. Place PDF or text files there to ingest."

        chroma = ChromaService()
        existing_sources = set()
        try:
            col = chroma.get_collection()
            data = col.get(include=["metadatas"])
            for m in data.get("metadatas", []):
                if m and "source" in m:
                    existing_sources.add(m["source"])
        except Exception:
            pass

        lines = [f"Documents in storage ({doc_dir}):"]
        for f in sorted(files, key=lambda x: x.name.lower()):
            size_kb = f.stat().st_size / 1024
            size_str = f"{size_kb / 1024:.2f} MB" if size_kb >= 1024 else f"{size_kb:.1f} KB"
            status = "Indexed in ChromaDB" if f.name in existing_sources else "Not yet indexed"
            lines.append(f"- {f.name} ({size_str}) -> [{status}]")

        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Error listing stored documents: {e}")
        return f"Error listing stored documents: {str(e)}"


def ingest_stored_document(filename: str, strategy: str = "fixed") -> str:
    """Ingest, extract, chunk, embed, and store a document from the local documents folder (data/documents) into ChromaDB.

    Args:
        filename: Name of the file in data/documents (e.g. 'Eggleston - Basic Electronics for Scientists and Engineers.pdf').
        strategy: Chunking strategy to use: 'fixed' (fast, recommended for large books/papers), 'semantic', or 'both'.

    Returns:
        A status message confirming the number of pages processed and chunks indexed into ChromaDB.
    """
    try:
        doc_dir = settings.document_storage_dir
        file_path = doc_dir / filename
        if not file_path.exists():
            matches = [
                f for f in doc_dir.iterdir() if filename.lower() in f.name.lower() and f.is_file()
            ]
            if matches:
                file_path = matches[0]
                filename = file_path.name
            else:
                return f"File '{filename}' not found in '{doc_dir}'. Use `list_stored_documents` to see available files."

        service = IngestionService()
        chroma = ChromaService()

        strat = strategy if strategy in ("fixed", "semantic", "both") else "fixed"

        if file_path.suffix.lower() == ".pdf":
            result = service.ingest_pdf(
                file_input=file_path,
                filename=filename,
                strategy=strat,  # type: ignore
                generate_embeddings=True,
                save_to_storage=False,
            )
        else:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            result = service.ingest_text(
                text=text,
                source_name=filename,
                strategy=strat,  # type: ignore
                generate_embeddings=True,
            )

        if not result.chunks:
            return f"No text content could be extracted or chunked from '{filename}'."

        count = chroma.add_chunks(result.chunks)
        bm25 = BM25Index()
        bm25.add_documents(result.chunks, persist=True)
        pages_info = f"{result.total_pages} pages, " if result.total_pages else ""
        return (
            f"Successfully ingested '{filename}': processed {pages_info}"
            f"generated {len(result.chunks)} chunks using '{strat}' strategy, "
            f"and stored {count} embeddings into ChromaDB. You can now search it!"
        )
    except Exception as e:
        logger.error(f"Error ingesting stored document '{filename}': {e}")
        return f"Error ingesting '{filename}': {str(e)}"


def advanced_research_query(query: str, top_k: int = 5) -> str:
    """Execute deep research using HyDE (Hypothetical Document Embeddings) and Multi-Query expansion with chained synthesis.

    Args:
        query: The research question to investigate.
        top_k: Number of relevant passages to retrieve (default 5).

    Returns:
        A grounded academic response citing specific sources and page numbers.
    """
    try:
        from research_agent.service.advanced_retrieval import ChainedRAGPipeline

        pipeline = ChainedRAGPipeline()
        result = pipeline.run(query=query, top_k=top_k, use_hyde=True, use_multiquery=True)
        return result.synthesized_answer
    except Exception as e:
        logger.error(f"Error executing advanced research query: {e}")
        return f"Error executing advanced research query: {str(e)}"


# Define the root Google ADK Agent
root_agent = Agent(
    name="research_agent",
    model=settings.gemini_model,
    instruction=(
        "You are an expert AI Research Assistant equipped with document ingestion and semantic search tools.\n"
        "Your role is to:\n"
        "1. Answer research questions accurately using facts from the persistent research knowledge base.\n"
        "2. When asked about papers, books, topics, or literature, use `advanced_research_query` or `search_research_documents`.\n"
        "3. When asked what documents, papers, or books are available locally, use `list_stored_documents`.\n"
        "4. When asked to read, load, or ingest a document from local storage (such as a PDF in data/documents), use `ingest_stored_document`.\n"
        "5. When asked to record, save, or ingest raw research notes or findings from the conversation, use `ingest_research_notes`.\n"
        "6. Always cite your sources with document names, page numbers, and similarity metrics when available.\n"
        "7. Be concise, academic, and structured in your explanations."
    ),
    tools=[
        advanced_research_query,
        search_research_documents,
        list_stored_documents,
        ingest_stored_document,
        ingest_research_notes,
    ],
)
