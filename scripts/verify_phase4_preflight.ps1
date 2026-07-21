$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
& .\.venv\Scripts\python.exe -m ruff check src tests scripts
if ($LASTEXITCODE -ne 0) { throw "Phase 4 ruff gate failed" }
& .\.venv\Scripts\python.exe -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Phase 4 pytest gate failed" }
& .\.venv\Scripts\python.exe -c "from src.experiments.phase4_common import phase4_preflight; print(phase4_preflight(require_h100=False)); print('PHASE 4 CPU PREFLIGHT: PASS')"
if ($LASTEXITCODE -ne 0) { throw "Phase 4 environment gate failed" }
