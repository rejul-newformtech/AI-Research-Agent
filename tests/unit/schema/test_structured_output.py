"""Unit tests for structured output Pydantic models."""

from app.schema.structured_output import (
    CitationModel,
    HyDEPassageModel,
    MultiQueryExpansionModel,
    ResearchSynthesisModel,
)


def test_citation_model_validation():
    cite = CitationModel(
        source="electronics_handbook.pdf",
        page_number=42,
        section="BJT Amplifiers",
        quote_or_fact="Small-signal voltage gain is proportional to transconductance gm.",
    )
    assert cite.source == "electronics_handbook.pdf"
    assert cite.page_number == 42
    assert "voltage gain" in cite.quote_or_fact

    d = cite.model_dump()
    assert d["page_number"] == 42
    j = cite.model_dump_json()
    assert "electronics_handbook.pdf" in j


def test_research_synthesis_model_validation():
    synthesis = ResearchSynthesisModel(
        summary="Transistors function as electronically controlled switches or amplifiers.",
        detailed_findings="A bipolar junction transistor (BJT) utilizes both electron and hole carriers.",
        citations=[
            CitationModel(
                source="physics.pdf",
                page_number=10,
                quote_or_fact="BJT operates in active, saturation, and cutoff regions.",
            )
        ],
        key_takeaways=[
            "Active mode is used for linear amplification.",
            "Saturation and cutoff are used in digital logic.",
        ],
        confidence_score=0.95,
        missing_evidence=None,
    )
    assert len(synthesis.citations) == 1
    assert len(synthesis.key_takeaways) == 2
    assert synthesis.confidence_score == 0.95

    json_str = synthesis.model_dump_json()
    restored = ResearchSynthesisModel.model_validate_json(json_str)
    assert restored.summary == synthesis.summary
    assert restored.citations[0].page_number == 10


def test_multi_query_and_hyde_models():
    mq = MultiQueryExpansionModel(
        queries=[
            "transistor amplification mechanism",
            "bjt small signal equivalent circuit",
            "semiconductor current gain calculations",
        ]
    )
    assert len(mq.queries) == 3

    hyde = HyDEPassageModel(
        hypothetical_passage="A field-effect transistor controls conductance via an electric field.",
        domain="electronics",
    )
    assert hyde.domain == "electronics"
