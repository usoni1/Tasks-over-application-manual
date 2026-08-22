# Data Manual

This manual documents how to maintain the distributable data portion of the repository.

## Scope

The data intended for direct distribution in this repository are:

- the final legal dataset files in `data/final-approved-200/`,
- the public reference manuals in `data/reference-manuals/`.

## Update procedure

1. Replace `federal_sentencing_legal_final_dataset_approved.csv` only when a new final approved legal release is available.
2. Keep the reference-manuals directory limited to public source manuals and public ICD addenda files actually used in the paper.
3. Do not commit extracted text exports, download manifests, logs, or other working artifacts that can expose internal paths or pipeline details.
4. Remove any stale legal or manual files that are no longer part of the approved release.
5. Confirm that the repository still contains no raw MIMIC files.
6. If the release format changed, update `README.md` and this manual so the repository description stays current.

## Minimum documentation expectation

The approved release should always have:

- the final approved dataset file,
- the public manuals needed to understand the paper inputs,
- the public ICD addenda files needed to rebuild the strict MIMIC table,
- enough documentation to understand what the release contains.

## Files that should not be committed

- exploratory exports,
- intermediate preprocessing outputs,
- extracted manual-processing artifacts with local-path or tool metadata,
- raw or restricted MIMIC tables,
- any file that has not passed the final legal approval step.