# Experiment Reproduction Manual

This repository now contains the evaluation code needed to rerun the paper baselines locally once you supply the restricted datasets and model credentials.

## What is in scope

These experiment names match the paper baselines:

- `icd-single-pass-rag` -> `scripts/evaluate_icd_rag.py`
- `icd-agentic-rag` -> `scripts/evaluate_icd_agentic_rag.py`
- `icd-react-style-tool-use` -> `scripts/evaluate_icd_react_v2.py`
- `icd-deepagents` -> `scripts/evaluate_icd_deepagents.py`
- `legal-single-pass-rag` -> `scripts/evaluate_legal_rag.py`
- `legal-agentic-rag` -> `scripts/evaluate_legal_agentic_rag.py`
- `legal-react-style-tool-use` -> `scripts/evaluate_legal_react_v2_on_review_dataset.py`
- `legal-deepagents` -> `scripts/evaluate_legal_deepagents.py`

## Files to use

- `pyproject.toml`: local dependency manifest for the standalone evaluator stack.
- `.env.example`: environment template for Azure Search, Azure OpenAI, manual roots, and local dataset paths.
- `scripts/reproduce/run_tam_baseline.py`: runs one baseline locally and synthesizes the temporary Spark views needed by the evaluators.
- `scripts/reproduce/run_table_2.ps1`: convenience wrapper that runs all eight paper baselines in sequence.
- `scripts/evaluate_*.py`: direct evaluator entrypoints if you want to bypass the wrapper.
- `scripts/run_icd_chunking.py` and `scripts/run_legal_chunking.py`: manual chunking entrypoints for rebuilding the search indices.

## Prerequisites

1. Install the standalone dependencies from this repo, for example `pip install -e .`.
2. Copy `.env.example` to `.env` and fill in the Azure Search and Azure OpenAI settings.
3. For ICD experiments, build the local parquet dataset first with `scripts/mimic/build_mimic_icd10_dataset_2017_2019.py` from this repository.
4. For legal experiments, provide a local CSV or Parquet file with `docket_id` and one of `sentencing_year`, `guideline_year`, or `year`.
5. For legal chunking and legal ReAct runs, make sure the public manuals under `data/reference-manuals/legal/` are present and point `LEGAL_USSG_DOCINTEL_TEXT_ROOT` at your local USSG Doc Intelligence export if you use the review-dataset ReAct baseline.

The legal approved release committed here contains only `docket_id`, `input`, and `output`, so the wrapper synthesizes the richer review-dataset view needed by `legal_react_v2` by joining that CSV with the docket-year map.

## Single-baseline command

Run one baseline like this:

```powershell
python scripts/reproduce/run_tam_baseline.py ^
  --experiment icd-single-pass-rag ^
  -- --limit 1000
```

For legal baselines, also pass the year map CSV:

```powershell
python scripts/reproduce/run_tam_baseline.py ^
  --experiment legal-react-style-tool-use ^
  --legal-sentencing-year-map-csv "PUT_PATH_TO_LEGAL_SENTENCING_YEAR_MAP_CSV_HERE" ^
  -- --limit 200 --review-version 4
```

Anything after `--` is forwarded directly to the local evaluator script.

You can also run the evaluators directly if you prefer explicit file paths:

```powershell
python scripts/evaluate_icd_rag.py ^
  --strict-path "data\mimic\mimic_icd10_note_dataset_2017_2019_strict.parquet" ^
  --limit 100
```

```powershell
python scripts/evaluate_legal_rag.py ^
  --dataset-path "data\final-approved-200\federal_sentencing_legal_final_dataset_approved.csv" ^
  --sentencing-year-path "PUT_PATH_TO_LEGAL_SENTENCING_YEAR_MAP_CSV_HERE" ^
  --limit 100
```

## Table 2 convenience command

To rerun all eight paper baselines in sequence:

```powershell
scripts/reproduce/run_table_2.ps1 ^
  -LegalSentencingYearMapCsv "PUT_PATH_TO_LEGAL_SENTENCING_YEAR_MAP_CSV_HERE"
```

Use `-SkipIcd` or `-SkipLegal` if you only want one domain.

## Notes

- The baseline packages are now vendored into this repository; reruns no longer depend on a checkout of `manual_deterministic_executor`.
- The evaluator scripts now support local CSV or Parquet inputs via `--strict-path`, `--dataset-path`, `--final-dataset-path`, `--sentencing-year-path`, `--acceptance-path`, and `--case-source-path`.
- ICD local mode defaults to `data/mimic/mimic_icd10_note_dataset_2017_2019_strict.parquet` when you use the reproduction wrapper.
- Legal local mode still requires a docket-year map for final-dataset evaluation and review-dataset synthesis.
- The review-dataset legal ReAct baseline still requires local Azure Search, Azure OpenAI, the public legal manuals, and a local USSG Doc Intelligence export.