# Data note and attribution

The two JSON files in this directory are a six-record teaching sample derived from **MedQuAD**. MedQuAD was created by Asma Ben Abacha and Dina Demner-Fushman from 12 NIH websites and is published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- Upstream dataset: [abachaa/MedQuAD](https://github.com/abachaa/MedQuAD)
- Citation: A. Ben Abacha and D. Demner-Fushman, “A Question-Entailment Approach to Question Answering,” *BMC Bioinformatics*, 2019.
- Local modifications: records were selected, normalized into `question`, `answer`, `source`, and `focus_area` fields, and a parallel Alpaca-format sample was produced for a course demonstration.

The upstream repository currently describes 47,457 QA pairs. This public demo ships only six records; it does not contain the full dataset or the historical experiment split.
