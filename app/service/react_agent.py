"""Explicit ReAct (Reasoning + Action + Observation) Agent Loop Service.

Implements iterative multi-step reasoning cycles with tool execution,
reflection, max-iteration guards, and complete step-by-step trace observability.
"""

import json
import re
from collections.abc import Callable
from typing import Any

from google import genai
from sqlalchemy.orm import Session

from app.agents.research_assistant.agent import (
    advanced_research_query,
    ingest_research_notes,
    ingest_stored_document,
    list_stored_documents,
    search_research_documents,
)
from app.core.config import settings
from app.core.logger import get_logger
from app.schema.agent import (
    ReActExecutionTrace,
    ReActStep,
)
from app.schema.structured_output import CitationModel, ResearchSynthesisModel, UserProfileContext
from app.service.memory import ConversationMemoryService
from app.service.prompting import DynamicPromptBuilder

logger = get_logger("app.service.react_agent")


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

        # Tool Registry
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
        """Parse Thought, Action, Action Input, or Final Answer from model generation.

        Returns:
            (thought, action, action_input, final_answer)
        """
        thought = ""
        action = None
        action_input = None
        final_answer = None

        # Check for Final Answer first
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

        # Extract Thought
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        else:
            thought = text.strip().split("\n")[0]

        # Extract Action
        action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)", text)
        if action_match:
            action = action_match.group(1).strip()

        # Extract Action Input
        input_match = re.search(r"Action Input:\s*(\{.*\}|[^\n]+)", text, re.DOTALL)
        if input_match:
            raw_input = input_match.group(1).strip()
            try:
                action_input = json.loads(raw_input)
            except json.JSONDecodeError:
                # Fallback: if single string passed without JSON formatting
                clean_val = raw_input.strip("\"'")
                action_input = {"query": clean_val}

        return thought, action, action_input, None

    def run(
        self,
        query: str,
        db: Session,
        user_id: str = "anonymous",
        session_id: str | None = None,
        max_iterations: int = 5,
        user_profile: UserProfileContext | None = None,
    ) -> tuple[str, ReActExecutionTrace, ResearchSynthesisModel | None]:
        """Execute the ReAct loop up to max_iterations.

        Returns:
            (final_answer, execution_trace, optional_structured_synthesis)
        """
        session_id = session_id or f"react_sess_{abs(hash(query)) % 1000000}"

        # 1. Initialize session and record user query in memory
        self.memory_service.get_or_create_session(
            db=db,
            session_id=session_id,
            user_id=1,  # Default or resolved DB user ID
            initial_prompt=query,
        )
        self.memory_service.save_message(
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

        # Conversation history window
        history = self.memory_service.get_windowed_history(db=db, session_id=session_id)
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

            # Call Gemini
            response = self.client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt_scratchpad,
            )
            raw_output = response.text or ""

            thought, action, action_input, direct_final_answer = self._parse_model_output(
                raw_output
            )

            # Check if final answer reached
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

            # If an action was identified
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

                # Append to scratchpad for the next reasoning step
                prompt_scratchpad += (
                    f"Thought: {thought}\n"
                    f"Action: {action}\n"
                    f"Action Input: {json.dumps(action_input)}\n"
                    f"Observation: {observation}\n"
                )
            else:
                # Model answered directly without explicit Action or Final Answer prefix
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

        # If max iterations reached without final answer, force synthesis
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

        # Save assistant answer to persistent session memory
        self.memory_service.save_message(
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

        # Build structured synthesis if citations are found in observations
        structured_synthesis = self._build_synthesis_summary(final_answer, steps)

        return final_answer, trace, structured_synthesis

    def _build_synthesis_summary(
        self, answer: str, steps: list[ReActStep]
    ) -> ResearchSynthesisModel | None:
        """Extract citations from observations to populate a structured synthesis model."""
        citations: list[CitationModel] = []
        for step in steps:
            if step.observation and "Source:" in step.observation:
                # Simple extraction of cited sources
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

        # Deduplicate citations by (source, page_number)
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
