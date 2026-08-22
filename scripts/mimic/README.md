# MIMIC Scripts

The helper in this directory rebuilds the strict ICD-10 note table used in the paper.

- `build_mimic_icd10_dataset_2017_2019.py`: rebuilds a local parquet version of `mimic_icd10_note_dataset_2017_2019_strict` from the MIMIC-IV `hosp/admissions.csv.gz`, `hosp/diagnoses_icd.csv.gz`, `hosp/patients.csv.gz`, and `note/discharge.csv.gz` files, while excluding admissions that touch ICD codes changed by the 2018-2020 addenda.