param(
    [string]$PythonExe = "python",
    [string]$IcdDatasetPath = "data\mimic\mimic_icd10_note_dataset_2017_2019_strict.parquet",
    [string]$LegalFinalDatasetCsv = "data\final-approved-200\federal_sentencing_legal_final_dataset_approved.csv",
    [string]$LegalSentencingYearMapCsv = "",
    [int]$ReviewVersion = 4,
    [switch]$SkipIcd,
    [switch]$SkipLegal
)

$runnerPath = Join-Path $PSScriptRoot "run_tam_baseline.py"

if (-not (Test-Path $runnerPath)) {
    throw "Runner script not found: $runnerPath"
}

if (-not $SkipLegal -and [string]::IsNullOrWhiteSpace($LegalSentencingYearMapCsv)) {
    throw "Set -LegalSentencingYearMapCsv to a CSV with docket_id and sentencing_year/guideline_year/year before running legal baselines."
}

function Invoke-TamBaseline {
    param(
        [Parameter(Mandatory = $true)][string]$Experiment
    )

    $command = @(
        $runnerPath,
        "--experiment", $Experiment,
        "--icd-dataset-path", $IcdDatasetPath,
        "--legal-final-dataset-csv", $LegalFinalDatasetCsv,
        "--review-version", $ReviewVersion.ToString()
    )

    if (-not [string]::IsNullOrWhiteSpace($LegalSentencingYearMapCsv)) {
        $command += @("--legal-sentencing-year-map-csv", $LegalSentencingYearMapCsv)
    }

    Write-Host "==> Running $Experiment" -ForegroundColor Cyan
    & $PythonExe @command
    if ($LASTEXITCODE -ne 0) {
        throw "Baseline failed: $Experiment"
    }
}

if (-not $SkipIcd) {
    Invoke-TamBaseline -Experiment "icd-single-pass-rag"
    Invoke-TamBaseline -Experiment "icd-agentic-rag"
    Invoke-TamBaseline -Experiment "icd-react-style-tool-use"
    Invoke-TamBaseline -Experiment "icd-deepagents"
}

if (-not $SkipLegal) {
    Invoke-TamBaseline -Experiment "legal-single-pass-rag"
    Invoke-TamBaseline -Experiment "legal-agentic-rag"
    Invoke-TamBaseline -Experiment "legal-react-style-tool-use"
    Invoke-TamBaseline -Experiment "legal-deepagents"
}