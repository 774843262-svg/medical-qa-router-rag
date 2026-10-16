# Historical course artifacts

This directory preserves material from the original course presentation without treating it as the output of the public six-record demo.

- `course_evaluation_summaries/` contains aggregate JSON files, sample logs and figures from prior experiments. The original frozen split, full index, per-example generation files, adapter weights, environment lockfile and complete external judge calls are not present. The values are therefore archived claims, not independently reproduced results.
- `qlora_course_config/` contains an earlier LLaMA-Factory configuration and dataset registration example. It references data splits and model artifacts that are not included and is not part of the tested public execution path.

Run `python src/evaluate_pipeline.py` for the checks supported by the public sample. Add `--show-archived-summaries` only to inspect the historical aggregate JSON with its limitation notice.
