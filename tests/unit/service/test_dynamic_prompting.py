"""Unit tests for DynamicPromptBuilder system prompt generation and domain detection."""

from app.schema.structured_output import UserProfileContext
from app.service.prompting import DynamicPromptBuilder


def test_detect_context_domain():
    builder = DynamicPromptBuilder()

    electronics_chunks = [
        {
            "text": "The operational amplifier circuit utilizes negative feedback.",
            "metadata": {"source": "electronics.pdf"},
        }
    ]
    domain = builder.detect_context_domain(electronics_chunks)
    assert domain == "electronics"

    cs_chunks = [
        {
            "text": "The neural network compiler optimizes tensor memory latency.",
            "metadata": {"source": "ml_systems.txt"},
        }
    ]
    assert builder.detect_context_domain(cs_chunks) == "computer_science"

    unknown_chunks = [{"text": "General meeting notes and agenda.", "metadata": {}}]
    assert builder.detect_context_domain(unknown_chunks) == "general_academic"


def test_build_system_prompt_researcher():
    builder = DynamicPromptBuilder()
    profile = UserProfileContext(
        username="alice",
        role="researcher",
        expertise_level="expert",
        target_tone="academic",
    )
    prompt = builder.build_system_prompt(profile, context_domain="physics")

    assert "Senior Research Scientist" in prompt
    assert "Audience Expertise: EXPERT" in prompt
    assert "Domain Focus: Physics" in prompt
    assert "Tone: Rigorous, formal, objective" in prompt


def test_build_system_prompt_admin():
    builder = DynamicPromptBuilder()
    profile = UserProfileContext(
        username="admin_bob",
        role="admin",
        expertise_level="intermediate",
        target_tone="executive",
        custom_instructions="Highlight security and compliance risks.",
    )
    prompt = builder.build_system_prompt(profile)

    assert "Executive Research Auditor" in prompt
    assert "Audience Expertise: INTERMEDIATE" in prompt
    assert "Tone: Concise, actionable, high-impact" in prompt
    assert "Highlight security and compliance risks" in prompt


def test_build_system_prompt_novice_user():
    builder = DynamicPromptBuilder()
    profile = UserProfileContext(
        username="student_charlie",
        role="user",
        expertise_level="novice",
        target_tone="didactic",
    )
    prompt = builder.build_system_prompt(profile)

    assert "Research Tutor and Explanatory Assistant" in prompt
    assert "Define specialized jargon" in prompt
    assert "Tone: Educational, engaging, clear" in prompt


def test_build_synthesis_prompt_with_context_and_history():
    builder = DynamicPromptBuilder()
    profile = UserProfileContext(role="researcher")
    chunks = [
        {
            "id": "c1",
            "text": "Transistors act as current valves in common emitter configuration.",
            "metadata": {"source": "eggleston.pdf", "page_number": 45, "strategy": "fixed"},
        }
    ]
    history = [
        {"role": "user", "content": "What is a diode?"},
        {"role": "assistant", "content": "A diode allows current in one direction."},
    ]

    prompt = builder.build_synthesis_prompt(
        query="How does a BJT amplify signal?",
        chunks=chunks,
        user_profile=profile,
        history=history,
    )

    assert "Source: eggleston.pdf, Page 45" in prompt
    assert "RECENT CONVERSATION CONTEXT" in prompt
    assert "What is a diode?" in prompt
    assert "ResearchSynthesisModel" in prompt


def test_build_hyde_and_multiquery_prompts():
    builder = DynamicPromptBuilder()
    profile = UserProfileContext(role="admin", expertise_level="intermediate")
    hyde_prompt = builder.build_hyde_prompt("What are qubits?", user_profile=profile)
    assert "qubits" in hyde_prompt

    mq_prompt = builder.build_multiquery_prompt(
        "What are qubits?", num_queries=4, user_profile=profile
    )
    assert "exactly 4 diverse search queries" in mq_prompt
    assert "operational, and verification angles" in mq_prompt
