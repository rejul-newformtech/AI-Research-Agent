"""Dynamic system prompting service.

Constructs prompts programmatically conditioned on:
1. User profile (role, expertise level, preferred tone).
2. Retrieved context (source types, detected domain, citation metadata).
3. Conversational history (sliding window context).
"""

from typing import Any

from app.core.logger import get_logger
from app.schema.structured_output import UserProfileContext

logger = get_logger("app.service.prompting")


class DynamicPromptBuilder:
    """Builder for constructing context-aware, user-profile-steered system prompts."""

    # Role-specific base personas
    ROLE_PROMPTS = {
        "admin": (
            "You are acting as an Executive Research Auditor and Knowledge Administrator.\n"
            "Focus on high-level synthesis, verifiable provenance, document governance, "
            "and flagging unverified claims or conflicting records."
        ),
        "researcher": (
            "You are an authoritative Senior Research Scientist and Academic Peer Reviewer.\n"
            "Provide deep technical rigor, precise terminology, exhaustive citations, "
            "and nuanced analysis grounded strictly in the evidentiary texts."
        ),
        "user": (
            "You are an expert Research Tutor and Explanatory Assistant.\n"
            "Break down complex academic findings into clear, structured, and didactic explanations "
            "while maintaining complete factual accuracy and citing sources."
        ),
    }

    # Expertise-level instructions
    EXPERTISE_INSTRUCTIONS = {
        "expert": (
            "Audience Expertise: EXPERT.\n"
            "- Use precise domain terminology and theoretical formulations without redundant simplification.\n"
            "- Focus on granular nuances, methodology, edge cases, and quantitative figures."
        ),
        "intermediate": (
            "Audience Expertise: INTERMEDIATE.\n"
            "- Balance rigorous technical terms with contextual clarification.\n"
            "- Clearly connect theoretical concepts to their practical functions."
        ),
        "novice": (
            "Audience Expertise: NOVICE.\n"
            "- Define specialized jargon and acronyms on first mention.\n"
            "- Provide intuitive real-world analogies and clear sequential deductions."
        ),
    }

    # Tone directives
    TONE_DIRECTIVES = {
        "academic": "Tone: Rigorous, formal, objective, and evidentiary.",
        "executive": "Tone: Concise, actionable, high-impact, prioritizing key findings and risks.",
        "didactic": "Tone: Educational, engaging, clear, and step-by-step.",
    }

    def detect_context_domain(self, chunks: list[dict[str, Any]]) -> str:
        """Infer scientific or technical domain from source names and chunk text."""
        import re

        combined_text = " ".join(
            [f"{c.get('metadata', {}).get('source', '')} {c.get('text', '')[:200]}" for c in chunks]
        ).lower()
        words = set(re.findall(r"\b[a-zA-Z_]+\b", combined_text))

        domains = [
            (
                "electronics",
                [
                    "circuit",
                    "voltage",
                    "transistor",
                    "amplifier",
                    "diode",
                    "current",
                    "impedance",
                    "resistor",
                ],
            ),
            (
                "computer_science",
                [
                    "algorithm",
                    "complexity",
                    "database",
                    "neural",
                    "compiler",
                    "memory",
                    "latency",
                    "async",
                ],
            ),
            (
                "biomedical",
                [
                    "protein",
                    "gene",
                    "cell",
                    "clinical",
                    "molecular",
                    "disease",
                    "patient",
                    "tissue",
                ],
            ),
            (
                "physics",
                [
                    "quantum",
                    "thermodynamic",
                    "electromagnetic",
                    "particle",
                    "velocity",
                    "relativity",
                ],
            ),
        ]

        for domain_name, keywords in domains:
            if any(kw in words for kw in keywords):
                return domain_name

        return "general_academic"

    def build_system_prompt(
        self,
        user_profile: UserProfileContext | None = None,
        context_domain: str | None = None,
    ) -> str:
        """Generate a customized system instruction tailored to user attributes."""
        profile = user_profile or UserProfileContext()
        role = profile.role.lower()
        role_desc = self.ROLE_PROMPTS.get(role, self.ROLE_PROMPTS["researcher"])

        expertise_key = profile.expertise_level.lower()
        expertise_desc = self.EXPERTISE_INSTRUCTIONS.get(
            expertise_key, self.EXPERTISE_INSTRUCTIONS["expert"]
        )

        tone_key = profile.target_tone.lower()
        tone_desc = self.TONE_DIRECTIVES.get(tone_key, self.TONE_DIRECTIVES["academic"])

        domain_info = (
            f"Domain Focus: {context_domain.replace('_', ' ').title()}."
            if context_domain and context_domain != "general_academic"
            else "Domain Focus: Multidisciplinary Academic Research."
        )

        prompt_parts = [
            "=== SYSTEM IDENTITY & OBJECTIVE ===",
            role_desc,
            "",
            "=== ADAPTIVE CONSTRAINTS ===",
            domain_info,
            expertise_desc,
            tone_desc,
        ]

        if profile.custom_instructions:
            prompt_parts.extend(["", "=== USER SPECIAL DIRECTIVE ===", profile.custom_instructions])

        return "\n".join(prompt_parts)

    def build_synthesis_prompt(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        user_profile: UserProfileContext | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Construct the complete synthesis prompt combining dynamic system prompt,

        retrieved context excerpts, and formatted schema instructions.
        """
        profile = user_profile or UserProfileContext()
        domain = self.detect_context_domain(chunks)
        system_instruction = self.build_system_prompt(profile, context_domain=domain)

        # Build context blocks
        context_blocks: list[str] = []
        for idx, c in enumerate(chunks, 1):
            source = c.get("metadata", {}).get("source", "unknown")
            page = c.get("metadata", {}).get("page_number", -1)
            page_str = f", Page {page}" if page and page != -1 else ""
            section = c.get("metadata", {}).get("strategy", "")
            strat_str = f" | Strategy: {section}" if section else ""
            text = c.get("text", "").strip()
            context_blocks.append(
                f"[Excerpt {idx}] (Source: {source}{page_str}{strat_str})\n{text}"
            )

        context_str = (
            "\n\n".join(context_blocks) if context_blocks else "No relevant documents found."
        )

        # Format conversation history if available
        history_str = ""
        if history:
            hist_lines = [
                f"{turn.get('role', 'user').title()}: {turn.get('content', '')}" for turn in history
            ]
            history_str = "\n=== RECENT CONVERSATION CONTEXT ===\n" + "\n".join(hist_lines) + "\n"

        prompt = (
            f"{system_instruction}\n\n"
            f"{history_str}"
            "=== RETRIEVED EVIDENCE PASSAGES ===\n"
            f"{context_str}\n\n"
            "=== GROUNDING & CITATION RULES ===\n"
            "1. Answer the research question strictly using the provided excerpts.\n"
            "2. For every factual assertion, attribute it explicitly to the exact source and page number.\n"
            "3. If evidence is ambiguous, incomplete, or missing from the excerpts, explicitly state the gap.\n"
            "4. Do not invent or extrapolate unverified facts.\n\n"
            "=== USER RESEARCH INQUIRY ===\n"
            f"{query}\n\n"
            "Produce a structured JSON response matching the ResearchSynthesisModel schema."
        )

        return prompt

    def build_hyde_prompt(
        self,
        query: str,
        user_profile: UserProfileContext | None = None,
    ) -> str:
        """Construct the HyDE generation prompt conditioned on user profile."""
        profile = user_profile or UserProfileContext()
        expertise = profile.expertise_level.lower()
        depth = "advanced technical" if expertise == "expert" else "concise academic"

        return (
            f"You are a recognized scientific author writing a {depth} passage.\n"
            "Write a realistic academic paragraph answering the following research question "
            "as if it were extracted from a peer-reviewed research paper or reference textbook.\n"
            "Do not include conversational filler, meta commentary, or preambles.\n\n"
            f"Research Question: {query}"
        )

    def build_multiquery_prompt(
        self,
        query: str,
        num_queries: int = 3,
        user_profile: UserProfileContext | None = None,
    ) -> str:
        """Construct the Multi-Query expansion prompt conditioned on user profile."""
        profile = user_profile or UserProfileContext()
        role = profile.role.lower()
        role_directive = (
            "focusing on architectural, operational, and verification angles"
            if role == "admin"
            else "focusing on underlying mechanisms, practical applications, and theoretical principles"
        )

        return (
            "You are an expert AI scientific research assistant.\n"
            f"Given the following research question, generate exactly {num_queries} diverse search queries "
            f"{role_directive} to maximize retrieval recall from academic textbooks and research papers.\n\n"
            f"Question: {query}"
        )
