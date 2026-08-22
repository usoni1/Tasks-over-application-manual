# Script Overview

This repository now contains the local dataset-build and standalone evaluation scripts needed to rerun the paper baselines from this repo.

- `mimic/build_mimic_icd10_dataset_2017_2019.py`: rebuilds `mimic_icd10_note_dataset_2017_2019_strict` from the upstream MIMIC hospital and note tables plus the committed ICD addenda files.
- `run_icd_chunking.py`: chunks the ICD manuals and uploads them to the configured Azure Search index.
- `run_legal_chunking.py`: chunks the legal manuals and uploads them to the configured Azure Search index.
- `evaluate_icd_rag.py`, `evaluate_icd_agentic_rag.py`, `evaluate_icd_react_v2.py`: standalone ICD baseline evaluators. They can read either existing Spark tables/views or local CSV/Parquet files via `--strict-path`.
- `evaluate_legal_rag.py`, `evaluate_legal_agentic_rag.py`, `evaluate_legal_react_v2_on_review_dataset.py`: standalone legal baseline evaluators. They can read existing Spark tables/views or local CSV/Parquet files via the new `--*-path` flags.
- `reproduce/run_tam_baseline.py`: convenience wrapper that runs one paper baseline locally from this repository.
- `reproduce/run_table_2.ps1`: convenience wrapper that runs the six Table 2 baselines in sequence.

The MIMIC build and evaluation scripts use local PySpark and no longer require the external `manual_deterministic_executor` repository.