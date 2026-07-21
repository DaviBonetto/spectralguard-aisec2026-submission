$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
& .\.venv\Scripts\python.exe -m ruff check src tests scripts
if ($LASTEXITCODE -ne 0) { throw "Phase 5 ruff gate failed" }
& .\.venv\Scripts\python.exe -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Phase 5 pytest gate failed" }
& .\.venv\Scripts\python.exe -c "from src.experiments.phase5_common import phase5_preflight; print(phase5_preflight(require_h100=False)); print('PHASE 5 CPU PREFLIGHT: PASS')"
if ($LASTEXITCODE -ne 0) { throw "Phase 5 environment gate failed" }
