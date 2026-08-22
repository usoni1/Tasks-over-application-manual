# CIKM Submission 2026 Artifacts

This repository keeps the paper artifacts in one place: the final approved legal dataset, the public reference manuals used by the paper, and the code needed to rebuild and rerun the released ICD and legal baselines from this repo.

## What is in this repository

- `data/final-approved-200/`: the final legally approved 200-point release committed as a CSV file.
- `data/reference-manuals/`: public ICD and legal reference manuals used by the paper.
- `data/mimic/`: local summaries and notes related to rebuilding the MIMIC strict table.
- `manuals/`: project documentation for the data layout and MIMIC build path.
- `scripts/`: helper scripts used to rebuild the MIMIC strict table, rebuild the search indices, and rerun the paper baselines locally.
- `baselines/` and `review_pipeline_v1/`: vendored evaluation code needed for the standalone reproduction workflow.
- `pyproject.toml` and `.env.example`: the standalone dependency and environment scaffold for local reruns.

## Legal and review note

Only the final approved 200-point package should be committed under `data/final-approved-200/`.
Only public reference manuals are committed under `data/reference-manuals/`; extracted working artifacts, logs, and source-project traces are intentionally excluded to preserve anonymity.
Raw MIMIC data must not be redistributed in this repository. The MIMIC side of this repo documents and scripts how to rebuild the strict 2017-2019 ICD-10 note dataset from credentialed upstream tables.

## Quick start

1. Inspect the committed approved release in `data/final-approved-200/`.
2. Inspect the public manuals under `data/reference-manuals/`.
3. Read the project documents:
	- `manuals/data-manual.md`
	- `manuals/mimic-manual.md`
	- `manuals/experiment-reproduction-manual.md`
4. Install the local evaluator stack from this repo, for example `pip install -e .`, and create a `.env` from `.env.example`.
5. If you need to rebuild the MIMIC ICD-10 2017-2019 strict table, follow `manuals/mimic-manual.md` and run `scripts/mimic/build_mimic_icd10_dataset_2017_2019.py`.
6. If you need to rerun the paper baselines, follow `manuals/experiment-reproduction-manual.md` and use `scripts/reproduce/run_tam_baseline.py`, `scripts/reproduce/run_table_2.ps1`, or the direct `scripts/evaluate_*.py` entrypoints.

The repository includes eight paper baselines, including the ICD and legal Deep Agents skills baselines. All eight are available through `scripts/reproduce/run_tam_baseline.py` and the `scripts/reproduce/run_table_2.ps1` convenience wrapper.

## Repository layout

```text
.
|-- data/
|   |-- README.md
|   |-- final-approved-200/
|   |   |-- federal_sentencing_legal_final_dataset_approved.csv
|   |   `-- README.md
|   |-- reference-manuals/
|   |   |-- README.md
|   |   |-- ICD-addendums/
|   |   |-- ICD-2019-manual/
|   |   `-- legal/
|   `-- mimic/
|-- manuals/
|   |-- data-manual.md
|   |-- experiment-reproduction-manual.md
|   `-- mimic-manual.md
`-- scripts/
	|-- README.md
	|-- reproduce/
		|-- run_table_2.ps1
		`-- run_tam_baseline.py
	`-- mimic/
		|-- README.md
		`-- build_mimic_icd10_dataset_2017_2019.py
```
