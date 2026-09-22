"""Unified AI Research Assistant Agent.

Unifies the Google ADK Agent framework (for adk web / standard runner integration)
and the explicit ReAct (Reasoning + Action + Observation) execution engine with full
step-by-step trace observability and safety guardrails.
"""

import json
import re
from collections.abc import Callable
from typing import Any

from google import genai
from google.adk.agents import Agent
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logger import get_logger
from app.db.chroma import ChromaService
from app.schema.agent import ReActExecutionTrace, ReActStep
from app.schema.structured_output import CitationModel, ResearchSynthesisModel, UserProfileContext
from app.service.ingestion import IngestionService
from app.service.memory import ConversationMemoryService
from app.service.prompting import DynamicPromptBuilder
from app.service.retrieval import BM25Index, HybridSearchService

logger = get_logger("research_agent.unified")


# ============================================================================
# Core Research Tools (Shared by ADK and ReAct)
# ============================================================================


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
        from app.service.advanced_retrieval import ChainedRAGPipeline

        pipeline = ChainedRAGPipeline()
        result = pipeline.run(query=query, top_k=top_k, use_hyde=True, use_multiquery=True)
        return result.synthesized_answer
    except Exception as e:
        logger.error(f"Error executing advanced research query: {e}")
        return f"Error executing advanced research query: {str(e)}"


# ============================================================================
# 1. Google ADK Agent Definition (for adk web / standard runner integration)
# ============================================================================

root_agent = Agent(
    name="research_agent",
    model=settings.gemini_model,
    instruction=(
        "You are an expert AI Research Assistant operating under the ReAct (Reasoning + Action) framework.\n"
        "Before calling any tool or producing an answer, ALWAYS explicitly write out your internal thought:\n"
        "Thought: <Explain what you know, what information is missing, and which tool you will invoke>\n"
        "Then invoke the tool.\n"
        "After observing the tool results, reflect on the evidence:\n"
        "Thought: <Evaluate if the gathered evidence is sufficient to answer completely>\n"
        "Final Answer: <Synthesize your evidence-backed answer citing specific document sources and page numbers [Source: <filename>, Page: <page_number>]>\n\n"
        "Available capabilities:\n"
        "1. For deep research questions, use `advanced_research_query` or `search_research_documents`.\n"
        "2. When asked what documents are available locally, use `list_stored_documents`.\n"
        "3. When asked to ingest a document from local storage, use `ingest_stored_document`.\n"
        "4. When asked to record research findings or notes, use `ingest_research_notes`.\n"
        "5. Always cite your sources with document names and page numbers."
    ),
    tools=[
        advanced_research_query,
        search_research_documents,
        list_stored_documents,
        ingest_stored_document,
        ingest_research_notes,
    ],
)


# ============================================================================
# 2. ReAct (Reasoning + Action + Observation) Agent Loop Engine
# ============================================================================

REACT_SYSTEM_PROMPT_TEMPLATE = """You are an expert AI Research Assistant operating under the ReAct (Reasoning + Action + Observation) framework.
Your task is to answer the user's research query by iteratively reasoning, selecting tools, and observing results.

You have access to the following tools:
{tool_descriptions}

Use the following format for each step:

Thought: Consider what information you currently have and what you still need to find.
Action: The tool name to use (must be one of: {tool_names})
Action Input: A valid JSON dictionary of arguments for the tool, e.g. {{"query": "quantum computing", "top_k": 3}}

After you provide an Action and Action Input, an Observation will be returned to you.
You can repeat the Thought/Action/Action Input/Observation cycle multiple times until you have enough evidence.

When you have gathered sufficient information to completely and authoritatively answer the user's question, output:

Thought: I now have sufficient evidence from the retrieved literature to construct a comprehensive answer.
Final Answer: Your detailed, academic, evidence-backed answer citing specific document sources and page numbers [Source: <filename>, Page: <page_number>].
"""


class ReActAgentService:
    """Orchestrates the ReAct reasoning, action, and observation cycle."""

    def __init__(
        self,
        memory_service: ConversationMemoryService | None = None,
        prompt_builder: DynamicPromptBuilder | None = None,
    ):
        self.memory_service = memory_service or ConversationMemoryService(default_window_size=10)
        self.prompt_builder = prompt_builder or DynamicPromptBuilder()
        self.client = genai.Client(api_key=settings.gemini_api_key)

        # Shared Tool Registry
        self.tool_registry: dict[str, Callable[..., Any]] = {
            "search_research_documents": search_research_documents,
            "advanced_research_query": advanced_research_query,
            "list_stored_documents": list_stored_documents,
            "ingest_stored_document": ingest_stored_document,
            "ingest_research_notes": ingest_research_notes,
        }

    def _get_tool_descriptions(self) -> tuple[str, str]:
        """Generate human-readable tool signatures and list of tool names."""
        descriptions = []
        for name, fn in self.tool_registry.items():
            doc = (fn.__doc__ or "No description").strip().split("\n")[0]
            descriptions.append(f"- `{name}`: {doc}")
        names_str = ", ".join(self.tool_registry.keys())
        return "\n".join(descriptions), names_str

    def _execute_tool(self, action: str, action_input: dict[str, Any]) -> str:
        """Safely invoke a registered tool with parsed arguments."""
        tool_fn = self.tool_registry.get(action)
        if not tool_fn:
            available = ", ".join(self.tool_registry.keys())
            return f"Error: Tool '{action}' not found. Available tools are: {available}."

        try:
            result = tool_fn(**action_input)
            return str(result)
        except TypeError as te:
            logger.error(f"Invalid arguments for tool '{action}': {te}")
            return f"Error executing tool '{action}': Invalid arguments provided ({te})."
        except Exception as e:
            logger.error(f"Exception running tool '{action}': {e}")
            return f"Error executing tool '{action}': {str(e)}"

    def _parse_model_output(
        self, text: str
    ) -> tuple[str, str | None, dict[str, Any] | None, str | None]:
        """Parse Thought, Action, Action Input, or Final Answer from model generation."""
        thought = ""
        action = None
        action_input = None
        final_answer = None

        if "Final Answer:" in text:
            parts = text.split("Final Answer:", 1)
            thought_part = parts[0]
            final_answer = parts[1].strip()

            thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", thought_part, re.DOTALL)
            thought = (
                thought_match.group(1).strip()
                if thought_match
                else thought_part.replace("Thought:", "").strip()
            )
            return thought, None, None, final_answer

        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        else:
            thought = text.strip().split("\n")[0]

        action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)", text)
        if action_match:
            action = action_match.group(1).strip()

        input_match = re.search(r"Action Input:\s*(\{.*\}|[^\n]+)", text, re.DOTALL)
        if input_match:
            raw_input = input_match.group(1).strip()
            try:
                action_input = json.loads(raw_input)
            except json.JSONDecodeError:
                clean_val = raw_input.strip("\"'")
                action_input = {"query": clean_val}

        return thought, action, action_input, None

    async def run(
        self,
        query: str,
        db: AsyncSession,
        user_id: str = "anonymous",
        session_id: str | None = None,
        max_iterations: int = 5,
        user_profile: UserProfileContext | None = None,
    ) -> tuple[str, ReActExecutionTrace, ResearchSynthesisModel | None]:
        """Execute the ReAct loop up to max_iterations."""
        session_id = session_id or f"react_sess_{abs(hash(query)) % 1000000}"

        numeric_user_id = (
            int(user_id)
            if isinstance(user_id, int) or (isinstance(user_id, str) and user_id.isdigit())
            else 1
        )
        await self.memory_service.get_or_create_session(
            db=db,
            session_id=session_id,
            user_id=numeric_user_id,
            initial_prompt=query,
        )
        await self.memory_service.save_message(
            db=db,
            session_id=session_id,
            role="user",
            content=query,
        )

        tool_descs, tool_names = self._get_tool_descriptions()
        system_prompt = REACT_SYSTEM_PROMPT_TEMPLATE.format(
            tool_descriptions=tool_descs,
            tool_names=tool_names,
        )

        if user_profile:
            dynamic_prefix = self.prompt_builder.build_system_prompt(user_profile)
            system_prompt = f"{dynamic_prefix}\n\n{system_prompt}"

        history = await self.memory_service.get_windowed_history(db=db, session_id=session_id)
        history_str = ""
        if len(history) > 1:
            history_str = "\n".join(f"{m.role.capitalize()}: {m.content}" for m in history[:-1])
            history_str = f"\nRelevant Conversation History:\n{history_str}\n"

        prompt_scratchpad = f"{system_prompt}\n{history_str}\nUser Question: {query}\n"
        steps: list[ReActStep] = []
        is_terminated = False
        termination_reason = "max_iterations_exceeded"
        final_answer = ""

        for iteration in range(1, max_iterations + 1):
            logger.info(
                f"ReAct Loop [Session {session_id}] - Iteration {iteration}/{max_iterations}"
            )

            response = self.client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt_scratchpad,
            )
            raw_output = response.text or ""

            thought, action, action_input, direct_final_answer = self._parse_model_output(
                raw_output
            )

            if direct_final_answer is not None:
                final_answer = direct_final_answer
                is_terminated = True
                termination_reason = "final_answer_reached"
                steps.append(
                    ReActStep(
                        step_number=iteration,
                        thought=thought,
                        action=None,
                        action_input=None,
                        observation=None,
                    )
                )
                break

            if action and action_input is not None:
                observation = self._execute_tool(action, action_input)
                steps.append(
                    ReActStep(
                        step_number=iteration,
                        thought=thought,
                        action=action,
                        action_input=action_input,
                        observation=observation,
                    )
                )

                prompt_scratchpad += (
                    f"Thought: {thought}\n"
                    f"Action: {action}\n"
                    f"Action Input: {json.dumps(action_input)}\n"
                    f"Observation: {observation}\n"
                )
            else:
                final_answer = raw_output.strip()
                is_terminated = True
                termination_reason = "direct_answer"
                steps.append(
                    ReActStep(
                        step_number=iteration,
                        thought=thought or "Direct synthesis",
                        action=None,
                        action_input=None,
                        observation=None,
                    )
                )
                break

        if not final_answer:
            logger.warning(
                f"ReAct Loop hit max iterations ({max_iterations}). Synthesizing best available answer."
            )
            forced_prompt = (
                f"{prompt_scratchpad}\n"
                f"You have reached the maximum allowed tool iterations. Based on all the observations above, "
                f"synthesize the final, most accurate answer possible for the user query: '{query}'."
            )
            synth_resp = self.client.models.generate_content(
                model=settings.gemini_model,
                contents=forced_prompt,
            )
            final_answer = synth_resp.text or "Max iterations reached without sufficient evidence."
            termination_reason = "max_iterations_reached"

        await self.memory_service.save_message(
            db=db,
            session_id=session_id,
            role="assistant",
            content=final_answer,
        )

        trace = ReActExecutionTrace(
            steps=steps,
            total_iterations=len(steps),
            is_terminated=is_terminated,
            termination_reason=termination_reason,
        )

        structured_synthesis = self._build_synthesis_summary(final_answer, steps)

        return final_answer, trace, structured_synthesis

    def _build_synthesis_summary(
        self, answer: str, steps: list[ReActStep]
    ) -> ResearchSynthesisModel | None:
        """Extract citations from observations to populate a structured synthesis model."""
        citations: list[CitationModel] = []
        for step in steps:
            if step.observation and "Source:" in step.observation:
                matches = re.findall(
                    r"Source:\s*([^\s\|\]]+)(?:\s*\(Page\s*(\d+)\))?", step.observation
                )
                for src, page in matches:
                    page_num = int(page) if page and page.isdigit() else None
                    citations.append(
                        CitationModel(
                            source=src,
                            page_number=page_num,
                            quote_or_fact="Referenced in ReAct observation",
                        )
                    )

        if not citations and "Source:" not in answer:
            return None

        seen = set()
        unique_citations = []
        for c in citations:
            key = (c.source, c.page_number)
            if key not in seen:
                seen.add(key)
                unique_citations.append(c)

        return ResearchSynthesisModel(
            summary=answer[:200] + "..." if len(answer) > 200 else answer,
            detailed_findings=answer,
            citations=unique_citations,
            key_takeaways=["Synthesized via iterative ReAct reasoning cycles"],
            confidence_score=0.90 if unique_citations else 0.75,
        )


# Global singleton instance of the unified ReAct Agent
react_agent = ReActAgentService()
