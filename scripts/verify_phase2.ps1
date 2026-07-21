$ErrorActionPreference = 'Stop'
$repo = Resolve-Path (Join-Path $PSScriptRoot '..')
$python = Join-Path $repo '.venv\Scripts\python.exe'
$ruff = Join-Path $repo '.venv\Scripts\ruff.exe'

Push-Location $repo
try {
    & $python -m src.eval.phase2_theory_audit --config configs\phase2_theory.yaml
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $python -m src.attacks.directional_counterexample --config configs\phase2_directional.yaml
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $ruff check .
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $python -m pytest -q
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & (Join-Path $repo 'scripts\build_paper.ps1')
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $python -m scripts.verify_phase2_results
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & (Join-Path $repo 'scripts\audit_anonymity.ps1') `
        -Output 'results/theory/anonymity_audit.json' `
        -AnonymousRepositoryUrl 'https://anonymous.4open.science/r/spectralguardaisec26/'
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Output 'PASS: complete Phase 2 gate.'
}
finally {
    Pop-Location
}
