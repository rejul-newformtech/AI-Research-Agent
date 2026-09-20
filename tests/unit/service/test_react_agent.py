"""Unit tests for the explicit ReActAgentService loop and tool execution."""

from unittest.mock import MagicMock, patch

from sqlalchemy.orm import Session

from app.schema.agent import ReActExecutionTrace
from app.service.react_agent import ReActAgentService


def test_react_tool_registry():
    service = ReActAgentService()
    assert "search_research_documents" in service.tool_registry
    assert "advanced_research_query" in service.tool_registry
    assert "list_stored_documents" in service.tool_registry

    descriptions, names = service._get_tool_descriptions()
    assert "search_research_documents" in names
    assert "search_research_documents" in descriptions


def test_react_parse_model_output_action_step():
    service = ReActAgentService()
    model_output = (
        "Thought: I need to check recent papers on transformer attention mechanisms.\n"
        "Action: search_research_documents\n"
        'Action Input: {"query": "transformer attention", "top_k": 3}'
    )
    thought, action, action_input, final_answer = service._parse_model_output(model_output)

    assert "transformer attention mechanisms" in thought
    assert action == "search_research_documents"
    assert action_input == {"query": "transformer attention", "top_k": 3}
    assert final_answer is None


def test_react_parse_model_output_final_answer():
    service = ReActAgentService()
    model_output = (
        "Thought: I have collected enough evidence from the textbooks.\n"
        "Final Answer: Transistors operate in active, saturation, and cutoff modes [Source: physics.pdf, Page 10]."
    )
    thought, action, action_input, final_answer = service._parse_model_output(model_output)

    assert "collected enough evidence" in thought
    assert action is None
    assert action_input is None
    assert "Transistors operate" in final_answer


@patch("google.genai.Client")
def test_react_loop_direct_final_answer(mock_client_cls, db: Session):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = (
        "Thought: The user is asking a basic concept that is well established.\n"
        "Final Answer: Moore's Law predicts transistor density doubles every 2 years [Source: electronics.pdf, Page 4]."
    )
    mock_client.models.generate_content.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    service = ReActAgentService()
    service.client = mock_client

    answer, trace, structured = service.run(
        query="What is Moore's Law?",
        db=db,
        session_id="test_react_direct_sess",
        max_iterations=3,
    )

    assert "Moore's Law" in answer
    assert isinstance(trace, ReActExecutionTrace)
    assert trace.is_terminated is True
    assert trace.termination_reason == "final_answer_reached"
    assert len(trace.steps) == 1
    assert trace.steps[0].action is None


@patch("google.genai.Client")
def test_react_loop_multi_step_action_then_answer(mock_client_cls, db: Session):
    mock_client = MagicMock()

    # Turn 1: Thought + Action (search)
    resp1 = MagicMock()
    resp1.text = (
        "Thought: Let me search the indexed documents for quantum entanglement.\n"
        "Action: search_research_documents\n"
        'Action Input: {"query": "quantum entanglement", "top_k": 2}'
    )

    # Turn 2: Thought + Final Answer
    resp2 = MagicMock()
    resp2.text = (
        "Thought: The search results provide direct evidence on Bell state correlations.\n"
        "Final Answer: Quantum entanglement occurs when quantum states cannot be factored [Source: physics.pdf, Page 22]."
    )

    mock_client.models.generate_content.side_effect = [resp1, resp2]
    mock_client_cls.return_value = mock_client

    service = ReActAgentService()
    service.client = mock_client

    # Mock tool execution
    service.tool_registry["search_research_documents"] = MagicMock(
        return_value="[1] Source: physics.pdf (Page 22) | Content: Bell states describe maximally entangled pairs."
    )

    answer, trace, structured = service.run(
        query="What is quantum entanglement?",
        db=db,
        session_id="test_react_multistep_sess",
        max_iterations=3,
    )

    assert "Quantum entanglement" in answer
    assert trace.total_iterations == 2
    assert trace.is_terminated is True
    assert trace.termination_reason == "final_answer_reached"

    # Step 1 was an action
    assert trace.steps[0].action == "search_research_documents"
    assert "Bell states" in trace.steps[0].observation

    # Step 2 was the final answer
    assert trace.steps[1].action is None


@patch("google.genai.Client")
def test_react_loop_max_iterations_guard(mock_client_cls, db: Session):
    mock_client = MagicMock()

    # Always return an action to test hitting max_iterations
    resp_action = MagicMock()
    resp_action.text = (
        "Thought: Need more info.\n"
        "Action: search_research_documents\n"
        'Action Input: {"query": "infinite loop test"}'
    )

    resp_forced_synth = MagicMock()
    resp_forced_synth.text = "Forced summary after reaching max iterations."

    # 2 iterations + 1 forced synthesis
    mock_client.models.generate_content.side_effect = [
        resp_action,
        resp_action,
        resp_forced_synth,
    ]
    mock_client_cls.return_value = mock_client

    service = ReActAgentService()
    service.client = mock_client
    service.tool_registry["search_research_documents"] = MagicMock(return_value="Sample doc text.")

    answer, trace, _ = service.run(
        query="Test query?",
        db=db,
        session_id="test_react_max_iter_sess",
        max_iterations=2,
    )

    assert answer == "Forced summary after reaching max iterations."
    assert trace.total_iterations == 2
    assert trace.termination_reason == "max_iterations_reached"
