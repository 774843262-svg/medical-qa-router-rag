"""Prompt and aggregation helpers for an experimental LLM-as-judge audit.

No judge model or API client is included here. Scores produced by an external
model are model-based assessments, not clinician review.
"""

import json
import os
from typing import Dict, List

SYSTEM_PROMPT = """
You are a strict evaluator for a medical question-answering RAG system.

Your task is to judge whether the assistant's answer is faithful to the provided retrieved references and the gold answer.

Definitions:
- Retrieval relevance: whether the retrieved references are relevant to the user's question.
- Faithfulness: whether the assistant answer is supported by the retrieved references.
- Hallucination: the answer contains medical claims that are not supported by the retrieved references or the gold answer.

Important rules:
1. Do not judge whether the medical advice is clinically sufficient beyond the provided references.
2. Penalize unsupported medical claims, specific treatments, dosages, diagnoses, or causal claims that are not present in the references or gold answer.
3. If the answer is mostly copied or paraphrased from the retrieved references, it should receive high faithfulness.
4. If the answer says the information is insufficient when references are insufficient, this is faithful.
5. Return JSON only. No markdown. No extra text.
""".strip()

SCORING_GUIDE = """
Scoring guide:
- retrieval_relevance_score:
  5 = references directly answer the question
  4 = mostly relevant
  3 = partially relevant
  2 = weakly relevant
  1 = irrelevant

- faithfulness_score:
  5 = all important claims are supported
  4 = mostly supported, minor unsupported details
  3 = partially supported
  2 = many unsupported claims
  1 = mostly unsupported or contradictory
""".strip()


def build_judge_user_prompt(question: str, gold_answer: str, references: List[Dict], assistant_answer: str) -> str:
    ref_blocks = []
    for idx, ref in enumerate(references, start=1):
        ref_blocks.append(f"Reference {idx}: Q: {ref['question']} | A: {ref['answer']}")
    refs_str = "\n".join(ref_blocks)
    return (
        f"{SCORING_GUIDE}\n\n"
        f"User Question: {question}\n"
        f"Gold Answer: {gold_answer}\n"
        f"Retrieved References:\n{refs_str}\n\n"
        f"Assistant Answer: {assistant_answer}\n\n"
        "Respond strictly with a JSON object containing keys: "
        '"retrieval_relevance_score" (int 1-5), "faithfulness_score" (int 1-5), '
        '"hallucination" (bool), "unsupported_claims" (list of strings), and "reason" (string).'
    )


def summarize_judge_results(records: List[Dict]) -> Dict:
    n = len(records)
    if n == 0:
        return {}
    avg_rel = sum(r["judge"]["retrieval_relevance_score"] for r in records) / n
    avg_faith = sum(r["judge"]["faithfulness_score"] for r in records) / n
    hal_cnt = sum(1 for r in records if r["judge"]["hallucination"])
    unsup_cnt = sum(len(r["judge"].get("unsupported_claims", [])) for r in records)
    return {
        "n": n,
        "avg_retrieval_relevance": round(avg_rel, 2),
        "avg_faithfulness": round(avg_faith, 2),
        "hallucination_count": hal_cnt,
        "hallucination_rate": round(hal_cnt / n, 4),
        "unsupported_claims_count": unsup_cnt,
    }
