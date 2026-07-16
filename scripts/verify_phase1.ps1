param(
    [switch]$SkipModelMeasurement,
    [switch]$SkipDocker
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$ruff = Join-Path $root ".venv\Scripts\ruff.exe"
$revision = "1e76775f628fbf1350fbe4dbb3d971ba64af25a1"

Push-Location $root
try {
    & $ruff check .
    if ($LASTEXITCODE -ne 0) { throw "Ruff failed." }

    & $python -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "Pytest failed." }

    & $python scripts\audit_references.py --tex paper\main.tex --bib paper\references.bib --output results\foundation\reference_audit.json
    if ($LASTEXITCODE -ne 0) { throw "Reference audit failed." }

    & $python scripts\audit_clasp_artifact.py --output results\foundation\clasp_artifact_audit.json
    if ($LASTEXITCODE -ne 0) { throw "CLASP artifact audit failed." }

    if (-not $SkipModelMeasurement) {
        & $python -m src.utils.measure_mamba_kappa --model-id state-spaces/mamba-130m-hf --revision $revision --output results\foundation\mamba_130m_kappa.json
        if ($LASTEXITCODE -ne 0) { throw "Model condition-number measurement failed." }
    }

    & scripts\build_paper.ps1
    if ($LASTEXITCODE -ne 0) { throw "Paper build failed." }

    & scripts\audit_anonymity.ps1
    if ($LASTEXITCODE -ne 0) { throw "Anonymity audit failed." }

    if (-not $SkipDocker) {
        docker build --pull -t spectralguard:phase1 -f docker\Dockerfile .
        if ($LASTEXITCODE -ne 0) { throw "Docker build failed." }
        docker run --rm spectralguard:phase1
        if ($LASTEXITCODE -ne 0) { throw "Docker validation failed." }
    }

    Write-Host "Phase 1 local verification passed."
}
finally {
    Pop-Location
}
