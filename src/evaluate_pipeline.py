#!/usr/bin/env python3
"""Run checks that are reproducible from the six-record public sample."""

import argparse
import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.router_rag import RouterRAGSystem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-archived-summaries",
        action="store_true",
        help="Print historical aggregate files that are not reproducible from this sample",
    )
    args = parser.parse_args()

    system = RouterRAGSystem(
        kb_path=os.path.join(ROOT_DIR, "data", "medquad_sample_kb.json")
    )
    queries = [
        "What are the treatments for Paroxysmal Hemicrania ?",
        "I feel tired and thirsty all the time. Could this be diabetes?",
    ]
    print("Public sample pathway checks")
    for query in queries:
        route = system.route_query(query)
        print(
            json.dumps(
                {
                    "query": query,
                    "backend": route["retrieval_backend"],
                    "top1_similarity": route["top1_similarity"],
                    "route": route["recommended_branch"],
                    "top_question": route["top3_candidates"][0]["question"],
                },
                indent=2,
            )
        )

    if args.show_archived_summaries:
        metrics_dir = os.path.join(
            ROOT_DIR, "archive", "course_evaluation_summaries", "metrics"
        )
        print(
            "\nARCHIVED AGGREGATES ONLY: raw splits, model outputs and judge calls are "
            "not included, so the following values are not reproduced here."
        )
        for filename in sorted(os.listdir(metrics_dir)):
            if filename.endswith(".json"):
                with open(os.path.join(metrics_dir, filename), encoding="utf-8") as handle:
                    print(filename, json.dumps(json.load(handle), indent=2))


if __name__ == "__main__":
    main()
