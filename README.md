# MedQuAD Medical QA Router Demo

A transparent medical-information retrieval prototype with two routes:

- high-similarity queries show candidate source questions for user confirmation;
- low-similarity queries return the top retrieved source excerpt with an explicit limitation banner.

The default public demo does **not** invoke Qwen, QLoRA or any other language model. That boundary is visible in both the UI and response metadata, so a reviewer can tell retrieval from generation.

> Academic team project by Jiang Xianbo, Ding Ruiyu, Li Lingyi, Li Chenxue and Du Jiaxue. Student identifiers and submitted assessment files are excluded from this public edition.

## What runs from this repository

```text
question
  └─ retrieve over 6 source-attributed MedQuAD samples
       ├─ similarity > threshold → top-3 confirmation → selected source answer
       └─ otherwise              → top evidence excerpt (no LLM)
```

- **Default backend:** normalized word and character n-gram cosine similarity; fully offline.
- **Optional backend:** `all-MiniLM-L6-v2` with FAISS when `requirements-dense.txt` is installed. The active backend is returned in `route_query()` and displayed in Streamlit.
- **Optional generation hook:** a caller can explicitly inject a `generator(prompt)` callable. No model weights, adapter, or implicit remote call are bundled.
- **Safety scope:** this is an educational software demo, not clinical validation or medical advice.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m unittest discover -s tests -v
python src/evaluate_pipeline.py
streamlit run app.py
```

For dense retrieval:

```bash
pip install -r requirements-dense.txt
streamlit run app.py
```

## Evidence and reproducibility

The test suite checks:

- exact-question routing and source-answer selection;
- extractive fallback behavior without an LLM;
- explicit generator injection;
- non-empty query validation and warning-banner enforcement;
- construction of the archived LLM-judge prompt helper.

Historical course experiments reported retrieval, text-similarity and model-judge metrics. The repository preserves those **aggregate summaries** under `archive/course_evaluation_summaries/`, but it does not include the original full MedQuAD split, FAISS index, generation outputs, adapter weights, inference environment or complete judge calls. Those headline metrics are therefore labelled archived and are not presented as reproducible results of this six-record demo.

## Repository map

```text
app.py                         Streamlit candidate-confirmation UI
src/router_rag.py              retrieval, routing and explicit generator hook
src/evaluate_pipeline.py       checks reproducible from the public sample
src/llm_judge_faithfulness.py  prompt/aggregation helper; no judge API client
data/                          six-record sample + attribution
tests/                         offline behavior tests
archive/                       historical configs and aggregate summaries
```

## Data attribution

The sample is derived from [MedQuAD](https://github.com/abachaa/MedQuAD), created by Asma Ben Abacha and Dina Demner-Fushman and released under CC BY 4.0. See [`data/README.md`](data/README.md).

## License

Project code is [MIT licensed](LICENSE). The derived sample data retains its upstream CC BY 4.0 terms and attribution.
