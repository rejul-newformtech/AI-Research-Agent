"""Unit tests for Dynamic System Prompting service."""

import unittest

from app.schema.structured_output import UserProfileContext
from app.service.prompting import DynamicPromptBuilder


class TestDynamicPromptBuilder(unittest.TestCase):
    """Test dynamic prompt construction based on user profile and retrieved context."""

    def setUp(self):
        self.builder = DynamicPromptBuilder()

    def test_detect_context_domain(self):
        electronics_chunks = [
            {
                "text": "The operational amplifier circuit utilizes negative feedback.",
                "metadata": {"source": "electronics.pdf"},
            }
        ]
        domain = self.builder.detect_context_domain(electronics_chunks)
        self.assertEqual(domain, "electronics")

        cs_chunks = [
            {
                "text": "The neural network compiler optimizes tensor memory latency.",
                "metadata": {"source": "ml_systems.txt"},
            }
        ]
        self.assertEqual(self.builder.detect_context_domain(cs_chunks), "computer_science")

        unknown_chunks = [{"text": "General meeting notes and agenda.", "metadata": {}}]
        self.assertEqual(self.builder.detect_context_domain(unknown_chunks), "general_academic")

    def test_build_system_prompt_researcher(self):
        profile = UserProfileContext(
            username="alice",
            role="researcher",
            expertise_level="expert",
            target_tone="academic",
        )
        prompt = self.builder.build_system_prompt(profile, context_domain="physics")

        self.assertIn("Senior Research Scientist", prompt)
        self.assertIn("Audience Expertise: EXPERT", prompt)
        self.assertIn("Domain Focus: Physics", prompt)
        self.assertIn("Tone: Rigorous, formal, objective", prompt)

    def test_build_system_prompt_admin(self):
        profile = UserProfileContext(
            username="admin_bob",
            role="admin",
            expertise_level="intermediate",
            target_tone="executive",
            custom_instructions="Highlight security and compliance risks.",
        )
        prompt = self.builder.build_system_prompt(profile)

        self.assertIn("Executive Research Auditor", prompt)
        self.assertIn("Audience Expertise: INTERMEDIATE", prompt)
        self.assertIn("Tone: Concise, actionable, high-impact", prompt)
        self.assertIn("Highlight security and compliance risks", prompt)

    def test_build_system_prompt_novice_user(self):
        profile = UserProfileContext(
            username="student_charlie",
            role="user",
            expertise_level="novice",
            target_tone="didactic",
        )
        prompt = self.builder.build_system_prompt(profile)

        self.assertIn("Research Tutor and Explanatory Assistant", prompt)
        self.assertIn("Define specialized jargon", prompt)
        self.assertIn("Tone: Educational, engaging, clear", prompt)

    def test_build_synthesis_prompt_with_context_and_history(self):
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

        prompt = self.builder.build_synthesis_prompt(
            query="How does a BJT amplify signal?",
            chunks=chunks,
            user_profile=profile,
            history=history,
        )

        self.assertIn("Source: eggleston.pdf, Page 45", prompt)
        self.assertIn("RECENT CONVERSATION CONTEXT", prompt)
        self.assertIn("What is a diode?", prompt)
        self.assertIn("ResearchSynthesisModel", prompt)

    def test_build_hyde_and_multiquery_prompts(self):
        profile = UserProfileContext(role="admin", expertise_level="intermediate")
        hyde_prompt = self.builder.build_hyde_prompt("What are qubits?", user_profile=profile)
        self.assertIn("qubits", hyde_prompt)

        mq_prompt = self.builder.build_multiquery_prompt(
            "What are qubits?", num_queries=4, user_profile=profile
        )
        self.assertIn("exactly 4 diverse search queries", mq_prompt)
        self.assertIn("operational, and verification angles", mq_prompt)


if __name__ == "__main__":
    unittest.main()
