"""Router-first retrieval demo over a small, source-attributed MedQuAD sample.

The default public path is intentionally offline: it retrieves candidate QA
records and returns a clearly labelled source excerpt. No language model is
silently invoked. A caller may explicitly inject a generator callable for
experimentation, while retaining the prompt and evidence in the response.
"""

import json
import os
import re
from typing import Callable, Dict, List, Optional
import numpy as np

SAFETY_WARNING_BANNER = (
    "⚠️ Educational software demonstration only. Retrieved source excerpts may be incomplete "
    "and have not been clinically validated by this project. This is not medical advice, diagnosis, "
    "or treatment; consult a qualified healthcare professional for personal concerns."
)


class MedQuADVectorStore:
    """
    Manages semantic embedding and vector search over MedQuAD QA pairs.
    Uses `SentenceTransformer('all-MiniLM-L6-v2')` + `faiss.IndexFlatIP` when installed,
    and seamlessly falls back to normalized word/character n-gram cosine similarity offline.
    """
    def __init__(self, kb_records: List[Dict], embedding_model_name: str = "all-MiniLM-L6-v2"):
        if not kb_records:
            raise ValueError("kb_records must contain at least one QA record")
        self.kb_records = kb_records
        self.embedding_model_name = embedding_model_name
        self.use_dense = False
        self.backend = "lexical_ngram"
        self.dense_init_error = None
        self._init_index()

    def _init_index(self):
        questions = [item["question"] for item in self.kb_records]
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.embedding_model_name)
            embeddings = self.encoder.encode(questions, convert_to_numpy=True, normalize_embeddings=True)
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings.astype("float32"))
            self.use_dense = True
            self.backend = "minilm_faiss"
        except Exception as exc:
            self.dense_init_error = type(exc).__name__
            # Lightweight lexical + character 3-gram cosine embedding fallback for offline/CPU testing
            self.vocab = {}
            tokenized = [self._extract_tokens(q) for q in questions]
            for toks in tokenized:
                for t in toks:
                    if t not in self.vocab:
                        self.vocab[t] = len(self.vocab)
            mat = np.zeros((len(questions), max(1, len(self.vocab))), dtype=np.float32)
            for i, toks in enumerate(tokenized):
                for t in toks:
                    mat[i, self.vocab[t]] += 1.0
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.doc_matrix = mat / norms

    @staticmethod
    def _extract_tokens(text: str) -> List[str]:
        clean = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        words = [w for w in clean.split() if len(w) > 1]
        # Add character 4-grams for robustness on medical terms
        compact = "".join(words)
        cgrams = [compact[i:i + 4] for i in range(max(0, len(compact) - 3))]
        return words * 2 + cgrams

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        k = min(top_k, len(self.kb_records))
        if self.use_dense:
            q_emb = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
            scores, indices = self.index.search(q_emb, k)
            scores, indices = scores[0], indices[0]
        else:
            q_vec = np.zeros(max(1, len(self.vocab)), dtype=np.float32)
            for t in self._extract_tokens(query):
                if t in self.vocab:
                    q_vec[self.vocab[t]] += 1.0
            norm = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec = q_vec / norm
            sims = self.doc_matrix @ q_vec
            indices = np.argsort(sims)[::-1][:k]
            scores = sims[indices]

        results = []
        for rank, (idx, sc) in enumerate(zip(indices, scores), start=1):
            rec = dict(self.kb_records[int(idx)])
            rec["rank"] = rank
            rec["similarity"] = round(float(sc), 4)
            results.append(rec)
        return results


class RouterRAGSystem:
    """
    Implements the Router-First Medical QA workflow:
      1. Query Router computes top-k semantic similarity against MedQuAD.
      2. If top1_similarity > router_threshold (0.68): routes to Branch A (Candidate QA Confirmation).
      3. If top1_similarity <= router_threshold or user declines candidates: routes to Branch B (Cautious RAG).
    """
    def __init__(
        self,
        kb_path: str = "data/medquad_sample_kb.json",
        router_threshold: float = 0.68,
        top_k_retrieval: int = 3,
        generator: Optional[Callable[[str], str]] = None,
    ):
        with open(kb_path, "r", encoding="utf-8") as f:
            self.kb_records = json.load(f)
        self.router_threshold = router_threshold
        self.top_k_retrieval = top_k_retrieval
        self.generator = generator
        self.vector_store = MedQuADVectorStore(self.kb_records)

    def route_query(self, user_query: str) -> Dict:
        if not isinstance(user_query, str) or not user_query.strip():
            raise ValueError("user_query must be a non-empty string")
        user_query = user_query.strip()
        candidates = self.vector_store.search(user_query, top_k=max(5, self.top_k_retrieval))
        top1_sim = candidates[0]["similarity"] if candidates else 0.0
        is_high_confidence = top1_sim > self.router_threshold

        return {
            "query": user_query,
            "top1_similarity": top1_sim,
            "router_threshold": self.router_threshold,
            "retrieval_backend": self.vector_store.backend,
            "recommended_branch": (
                "BRANCH_A_DIRECT_CONFIRMATION" if is_high_confidence else "BRANCH_B_RAG_FALLBACK"
            ),
            "top3_candidates": candidates[:3],
            "top_k_evidence": candidates[: self.top_k_retrieval],
        }

    @staticmethod
    def build_rag_prompt(user_query: str, evidence_docs: List[Dict]) -> str:
        refs_text = []
        for i, doc in enumerate(evidence_docs, start=1):
            refs_text.append(
                f"[Reference {i}] (Source: {doc.get('source', 'NIH')} | Topic: {doc.get('focus_area', 'General')})\n"
                f"Matched Question: {doc['question']}\n"
                f"Source Answer: {doc['answer']}"
            )
        joined_refs = "\n\n".join(refs_text)
        return (
            "You are a cautious, evidence-based medical question-answering assistant.\n"
            "Answer the user's question strictly using the retrieved MedQuAD references below.\n"
            "Do NOT invent unsupported clinical claims, dosages, or diagnoses.\n"
            "If the retrieved references do not contain sufficient evidence, explicitly state that "
            "the information is insufficient.\n\n"
            f"### Retrieved MedQuAD Evidence:\n{joined_refs}\n\n"
            f"### User Question:\n{user_query}\n\n"
            "### Evidence-Grounded Draft:"
        )

    def answer_via_direct_database(self, selected_candidate: Dict) -> Dict:
        return {
            "pathway": "Branch A — Confirmed Source Answer",
            "matched_question": selected_candidate["question"],
            "similarity": selected_candidate["similarity"],
            "answer": selected_candidate["answer"],
            "source": selected_candidate.get("source", "NIH"),
            "focus_area": selected_candidate.get("focus_area", "General"),
            "warning": SAFETY_WARNING_BANNER,
        }

    def answer_via_evidence_fallback(
        self,
        user_query: str,
        evidence_docs: List[Dict],
    ) -> Dict:
        prompt = self.build_rag_prompt(user_query, evidence_docs)
        if self.generator is None:
            top_doc = evidence_docs[0] if evidence_docs else None
            answer = top_doc["answer"] if top_doc else "Insufficient retrieved evidence."
            generation_mode = "extractive_top_evidence"
            pathway = "Branch B — Retrieved Evidence Fallback (No LLM)"
        else:
            answer = self.generator(prompt)
            generation_mode = "explicit_external_generator"
            pathway = "Branch B — Caller-Provided Generator"
        return {
            "pathway": pathway,
            "generation_mode": generation_mode,
            "prompt": prompt,
            "answer": answer,
            "evidence": evidence_docs,
            "warning": SAFETY_WARNING_BANNER,
        }
