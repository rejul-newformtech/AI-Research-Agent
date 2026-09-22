"""Automated evaluation benchmark measuring retrieval quality, answer relevance, and hallucination rate."""

import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

# Ensure workspace root is in sys.path when executed directly
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure stdout handles UTF-8 cleanly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.service.advanced_retrieval import ChainedRAGPipeline  # noqa: E402
from app.service.retrieval import HybridSearchService  # noqa: E402

# The 10 Curated Evaluation Questions
EVAL_QUESTIONS = [
    {
        "id": "Q1",
        "category": "Exact Fact",
        "question": "What is the scheduled shift window and role for Agent Sarah Jenkins?",
        "expected_facts": ["08:00 - 16:00", "Lead Supervisor", "AGT-101"],
        "is_in_corpus": True,
    },
    {
        "id": "Q2",
        "category": "Exact Fact",
        "question": "Who is serving as the Dispatch Coordinator and what is their current shift?",
        "expected_facts": ["Tom Lawson", "12:00 - 20:00", "AGT-106"],
        "is_in_corpus": True,
    },
    {
        "id": "Q3",
        "category": "Metrics",
        "question": "What are the total agents, agents on duty, and overall coverage percentage in the report?",
        "expected_facts": ["24", "18", "94.2%"],
        "is_in_corpus": True,
    },
    {
        "id": "Q4",
        "category": "Multi-Hop Filter",
        "question": "Which agents are assigned to the 08:00 - 16:00 shift window?",
        "expected_facts": ["Sarah Jenkins", "Marcus Chen", "Aisha Patel"],
        "is_in_corpus": True,
    },
    {
        "id": "Q5",
        "category": "Role Filter",
        "question": "Which agents currently hold Tier 1 Support roles?",
        "expected_facts": ["Aisha Patel", "Rachel Green"],
        "is_in_corpus": True,
    },
    {
        "id": "Q6",
        "category": "Status Filter",
        "question": "Identify all agents whose current status is either On Call or Offline.",
        "expected_facts": ["Marcus Chen", "Tom Lawson"],
        "is_in_corpus": True,
    },
    {
        "id": "Q7",
        "category": "Semantic Paraphrase",
        "question": "Who is the field operative scheduled for duty during late afternoon?",
        "expected_facts": ["David Miller", "10:00 - 18:00", "Field Agent"],
        "is_in_corpus": True,
    },
    {
        "id": "Q8",
        "category": "Semantic Paraphrase",
        "question": "What is the operational metric for staff actively handling waiting queues?",
        "expected_facts": ["5", "queue"],
        "is_in_corpus": True,
    },
    {
        "id": "Q9",
        "category": "Hallucination Bait (Negative)",
        "question": "What is the annual salary and bonus compensation package for Agent Sarah Jenkins?",
        "expected_facts": [],
        "is_in_corpus": False,
        "unanswerable_cues": [
            "not",
            "unspecified",
            "does not contain",
            "no information",
            "not provided",
            "missing",
            "does not mention",
            "no record",
            "no reference",
            "not stated",
        ],
    },
    {
        "id": "Q10",
        "category": "Hallucination Bait (Negative)",
        "question": "What emergency fire evacuation protocol is specified in the roster report?",
        "expected_facts": [],
        "is_in_corpus": False,
        "unanswerable_cues": [
            "not",
            "unspecified",
            "does not contain",
            "no information",
            "not provided",
            "missing",
            "does not outline",
            "no record",
            "no reference",
            "not mentioned",
            "not stated",
        ],
    },
]


def _normalize(text: str) -> str:
    """Normalize unicode characters and ligatures for reliable evaluation."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text).lower()
    normalized = normalized.replace("\ufb04", "ff").replace("—", "-").replace("–", "-")
    return normalized


def score_retrieval(chunks: list[dict[str, Any]], q_spec: dict[str, Any], k: int = 3) -> float:
    """Calculate Precision@k for retrieved chunks."""
    if not q_spec["is_in_corpus"]:
        # Out-of-corpus query: retrieval precision is 1.0 since corpus genuinely lacks facts
        return 1.0

    if not chunks:
        return 0.0

    top_chunks = chunks[:k]
    relevant_count = 0
    for chunk in top_chunks:
        text = _normalize(chunk.get("text", ""))
        matches = sum(1 for fact in q_spec["expected_facts"] if _normalize(fact) in text)
        if matches > 0:
            relevant_count += 1

    return round(relevant_count / min(len(top_chunks), k), 2)


def score_answer_relevance(answer: str, q_spec: dict[str, Any]) -> float:
    """Calculate Answer Relevance score (0.0 to 1.0)."""
    ans_norm = _normalize(answer)

    if not q_spec["is_in_corpus"]:
        # Out-of-corpus: high relevance means clearly acknowledging the absence of information
        has_cue = any(_normalize(cue) in ans_norm for cue in q_spec["unanswerable_cues"])
        return 1.0 if has_cue else 0.2

    # In-corpus: proportion of expected facts present in the answer
    facts = q_spec["expected_facts"]
    matches = sum(1 for fact in facts if _normalize(fact) in ans_norm)
    return round(matches / len(facts), 2)


def detect_hallucination(answer: str, chunks: list[dict[str, Any]], q_spec: dict[str, Any]) -> bool:
    """Detect if the answer hallucinates facts not supported by chunks or invents unverified data."""
    ans_norm = _normalize(answer)

    if not q_spec["is_in_corpus"]:
        # Negative question: check if the model fabricated answers instead of declining
        has_cue = any(_normalize(cue) in ans_norm for cue in q_spec["unanswerable_cues"])
        has_fabricated_salary = bool(re.search(r"\$\d+|\d+,\d+\s*(dollars|usd)", ans_norm))
        has_fabricated_fire = any(
            word in ans_norm
            for word in [
                "fire extinguisher",
                "alarm",
                "stairwell",
                "exit 4",
                "assembly point",
                "evacuation map",
            ]
        )
        if has_fabricated_salary or has_fabricated_fire or not has_cue:
            return True
        return False

    # In-corpus: check if the synthesized answer contradicts chunk contents
    combined_chunk_text = _normalize(" ".join([c.get("text", "") for c in chunks]))
    for fact in q_spec["expected_facts"]:
        norm_fact = _normalize(fact)
        if norm_fact in ans_norm and norm_fact not in combined_chunk_text:
            return True

    return False


def run_evaluation() -> dict[str, Any]:
    print("=" * 75)
    print("Starting AI Research Agent Evaluation Benchmark (10 Questions)")
    print("=" * 75)

    search_svc = HybridSearchService()
    rag_pipeline = ChainedRAGPipeline()

    results = []
    p1_scores = []
    p3_scores = []
    relevance_scores = []
    hallucination_flags = []

    for idx, item in enumerate(EVAL_QUESTIONS):
        qid = item["id"]
        q_text = item["question"]
        category = item["category"]
        print(f"\nEvaluating [{qid}] ({category}): {q_text}")

        # 1. Retrieval
        retrieved_chunks = search_svc.hybrid_search(query=q_text, top_k=3)
        p_at_1 = score_retrieval(retrieved_chunks, item, k=1)
        p_at_3 = score_retrieval(retrieved_chunks, item, k=3)
        p1_scores.append(p_at_1)
        p3_scores.append(p_at_3)

        # 2. Synthesis via Chained RAG with retry logic on 429/503 rate limit
        answer = ""
        rag_res = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                rag_res = rag_pipeline.run(
                    query=q_text, top_k=3, use_hyde=False, use_multiquery=False
                )
                raw_ans = rag_res.synthesized_answer
                is_error = any(
                    code in raw_ans for code in ["RESOURCE_EXHAUSTED", "429", "503", "UNAVAILABLE"]
                )
                if is_error:
                    print(
                        f"    [Rate limit / 503 hit on attempt {attempt + 1}, waiting 15s before retry...]"
                    )
                    time.sleep(15)
                    continue
                answer = raw_ans
                break
            except Exception as e:
                print(f"    [Exception on attempt {attempt + 1}: {e}. Waiting 15s before retry...]")
                time.sleep(15)

        if not answer and rag_res:
            answer = rag_res.synthesized_answer

        # 3. Scoring
        relevance = score_answer_relevance(answer, item)
        relevance_scores.append(relevance)

        is_hallucinated = detect_hallucination(answer, retrieved_chunks, item)
        hallucination_flags.append(is_hallucinated)

        confidence = (
            rag_res.structured_synthesis.confidence_score
            if (rag_res and rag_res.structured_synthesis)
            else 0.0
        )

        print(f"  -> Precision@1: {p_at_1:.2f} | Precision@3: {p_at_3:.2f}")
        print(f"  -> Answer Relevance: {relevance:.2f}")
        print(f"  -> Hallucination Detected: {is_hallucinated}")
        print(f"  -> Model Confidence: {confidence:.2f}")
        first_line = answer.split("\n")[0] if answer else "No answer"
        print(f"  -> Answer: {first_line[:120]}...")

        results.append(
            {
                "id": qid,
                "category": category,
                "question": q_text,
                "is_in_corpus": item["is_in_corpus"],
                "precision_at_1": p_at_1,
                "precision_at_3": p_at_3,
                "answer_relevance": relevance,
                "hallucination": is_hallucinated,
                "confidence_score": confidence,
                "synthesized_answer": answer,
                "citations": [c.model_dump() for c in rag_res.structured_synthesis.citations]
                if (rag_res and rag_res.structured_synthesis)
                else [],
                "retrieved_chunk_count": len(retrieved_chunks),
            }
        )

        # Short pacing delay between questions
        if idx < len(EVAL_QUESTIONS) - 1:
            time.sleep(5)

    # Summary Statistics
    mean_p1 = round(sum(p1_scores) / len(p1_scores), 2)
    mean_p3 = round(sum(p3_scores) / len(p3_scores), 2)
    mean_relevance = round(sum(relevance_scores) / len(relevance_scores), 2)
    hallucination_count = sum(1 for h in hallucination_flags if h)
    hallucination_rate = round((hallucination_count / len(hallucination_flags)) * 100.0, 1)

    summary = {
        "total_questions": len(EVAL_QUESTIONS),
        "mean_precision_at_1": mean_p1,
        "mean_precision_at_3": mean_p3,
        "mean_answer_relevance": mean_relevance,
        "hallucination_count": hallucination_count,
        "hallucination_rate_percent": hallucination_rate,
        "detailed_results": results,
    }

    print("\n" + "=" * 75)
    print("EVALUATION BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"Total Questions Evaluated: {len(EVAL_QUESTIONS)}")
    print(f"Mean Retrieval Precision@1: {mean_p1 * 100:.1f}%")
    print(f"Mean Retrieval Precision@3: {mean_p3 * 100:.1f}%")
    print(f"Mean Answer Relevance:      {mean_relevance * 100:.1f}%")
    print(
        f"Hallucination Rate:         {hallucination_rate:.1f}% ({hallucination_count}/{len(EVAL_QUESTIONS)})"
    )
    print("=" * 75)

    return summary


if __name__ == "__main__":
    out = run_evaluation()
    out_file = Path("eval_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_file.absolute()}")
