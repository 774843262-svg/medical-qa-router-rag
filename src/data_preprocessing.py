"""Convert an authorized MedQuAD-style CSV into KB and Alpaca JSON files.

This utility cleans and deduplicates records but does not create train,
validation or test splits. Callers are responsible for data licensing and for
freezing any evaluation split before model development.
"""

import json
import os
import re
from typing import Dict, List, Tuple
import pandas as pd


def clean_medical_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_medquad_csv(
    csv_path: str,
    output_dir: str = "data",
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Load a MedQuAD-style CSV, remove incomplete/duplicate entries, and export
    knowledge-base and Alpaca-format JSON files.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    required_cols = ["question", "answer", "source", "focus_area"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    df["question"] = df["question"].apply(clean_medical_text)
    df["answer"] = df["answer"].apply(clean_medical_text)
    df["source"] = df["source"].fillna("NIH").apply(clean_medical_text)
    df["focus_area"] = df["focus_area"].fillna("General Health").apply(clean_medical_text)

    # Filter out empty questions or answers and deduplicate
    df = df[(df["question"].str.len() > 5) & (df["answer"].str.len() > 15)].copy()
    df.drop_duplicates(subset=["question", "answer"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    kb_records = []
    alpaca_records = []
    for idx, row in df.iterrows():
        kb_records.append({
            "doc_id": f"MEDQUAD_{idx:05d}",
            "question": row["question"],
            "answer": row["answer"],
            "source": row["source"],
            "focus_area": row["focus_area"],
        })
        alpaca_records.append({
            "instruction": (
                "You are a cautious medical question-answering assistant. Answer the following medical "
                "question accurately based on verified clinical knowledge."
            ),
            "input": row["question"],
            "output": row["answer"],
            "system": (
                "Provide evidence-based medical information. For educational demonstration only; "
                "do not provide personal clinical diagnoses."
            ),
        })

    kb_out = os.path.join(output_dir, "medquad_kb_full.json")
    with open(kb_out, "w", encoding="utf-8") as f:
        json.dump(kb_records, f, indent=2, ensure_ascii=False)

    alpaca_out = os.path.join(output_dir, "medquad_alpaca_full.json")
    with open(alpaca_out, "w", encoding="utf-8") as f:
        json.dump(alpaca_records, f, indent=2, ensure_ascii=False)

    print(f"Preprocessed {len(df)} valid MedQuAD QA pairs -> saved to {output_dir}/")
    return df, kb_records
