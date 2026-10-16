#!/usr/bin/env python3

import os
import sys
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.llm_judge_faithfulness import SYSTEM_PROMPT, build_judge_user_prompt
from src.router_rag import RouterRAGSystem, SAFETY_WARNING_BANNER


class TestMedicalQARouter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb_path = os.path.join(ROOT_DIR, "data", "medquad_sample_kb.json")
        cls.system = RouterRAGSystem(
            kb_path=cls.kb_path, router_threshold=0.68, top_k_retrieval=3
        )

    def test_exact_question_routes_to_confirmation(self):
        route = self.system.route_query(
            "What are the treatments for Paroxysmal Hemicrania ?"
        )
        self.assertGreater(route["top1_similarity"], 0.68)
        self.assertEqual(route["recommended_branch"], "BRANCH_A_DIRECT_CONFIRMATION")
        self.assertIn(route["retrieval_backend"], {"lexical_ngram", "minilm_faiss"})
        answer = self.system.answer_via_direct_database(route["top3_candidates"][0])
        self.assertIn("indomethacin", answer["answer"].lower())
        self.assertEqual(answer["warning"], SAFETY_WARNING_BANNER)

    def test_default_fallback_is_extractive_and_does_not_claim_an_llm(self):
        route = self.system.route_query(
            "I feel tired and thirsty all the time. Could this be diabetes?"
        )
        response = self.system.answer_via_evidence_fallback(
            route["query"], route["top_k_evidence"]
        )
        self.assertEqual(response["generation_mode"], "extractive_top_evidence")
        self.assertIn("No LLM", response["pathway"])
        self.assertEqual(response["answer"], route["top_k_evidence"][0]["answer"])

    def test_generator_must_be_explicitly_injected(self):
        system = RouterRAGSystem(
            kb_path=self.kb_path,
            generator=lambda prompt: "explicit test generator output",
        )
        route = system.route_query("What is asthma?")
        response = system.answer_via_evidence_fallback(
            route["query"], route["top_k_evidence"]
        )
        self.assertEqual(response["generation_mode"], "explicit_external_generator")
        self.assertEqual(response["answer"], "explicit test generator output")

    def test_blank_query_is_rejected(self):
        with self.assertRaises(ValueError):
            self.system.route_query("   ")

    def test_judge_helper_is_prompt_only(self):
        self.assertIn("Faithfulness", SYSTEM_PROMPT)
        prompt = build_judge_user_prompt(
            "What is asthma?",
            "Asthma is a chronic lung disease.",
            [{"question": "What is asthma?", "answer": "Asthma is a chronic lung disease."}],
            "Asthma is a chronic lung disease.",
        )
        self.assertIn("retrieval_relevance_score", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
