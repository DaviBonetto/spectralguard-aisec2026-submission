$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing local Python environment: $python"
}

Push-Location $root
try {
    & $python -m ruff check src tests
    if ($LASTEXITCODE -ne 0) { throw "ruff failed" }

    & $python -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "pytest failed" }

    & $python -c "from src.experiments.phase3_runner import prepare_phase3; r=prepare_phase3(mode='dry_run'); assert r['preflight']['passed']; print('PASS: Phase 3 dry-run preflight')"
    if ($LASTEXITCODE -ne 0) { throw "Phase 3 dry-run failed" }

    $raw = Join-Path $root "Experiments\raw"
    if ((Test-Path -LiteralPath $raw) -and (Get-ChildItem -LiteralPath $raw -File -ErrorAction SilentlyContinue)) {
        throw "Dry-run created raw model outputs"
    }
}
finally {
    Pop-Location
}
