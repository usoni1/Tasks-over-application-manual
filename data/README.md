# Data Layout

This directory separates the material that can be distributed with the paper package from the material that must be rebuilt locally.

- `final-approved-200/`: the final approved 200-point legal dataset committed directly in the repository.
- `reference-manuals/`: public ICD and legal manuals, plus the ICD addenda files used by the strict MIMIC build.
- `mimic/`: local summaries and notes for rebuilding the MIMIC strict dataset.

Only public manual files belong in `reference-manuals/`. Do not commit extracted text, local manifests, logs, or other working artifacts that can reveal internal pipeline details.

Do not commit raw credentialed MIMIC tables.