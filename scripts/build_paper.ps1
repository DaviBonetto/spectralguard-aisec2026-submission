$ErrorActionPreference = 'Stop'
$paper = Join-Path $PSScriptRoot '..\\paper'
Push-Location $paper
try {
    # MiKTeX's latexmk wrapper needs Perl, which is not installed on this host.
    # Two direct pdflatex passes suffice while the Phase 0 bibliography is empty.
    & pdflatex -interaction=nonstopmode -halt-on-error main.tex
    & pdflatex -interaction=nonstopmode -halt-on-error main.tex
    $pdf = Join-Path $paper 'main.pdf'
    if (-not (Test-Path $pdf)) { throw 'Paper PDF was not produced.' }
    # Count page objects directly so this works without a separate Poppler install.
    $pdfText = [System.Text.Encoding]::GetEncoding(28591).GetString([System.IO.File]::ReadAllBytes($pdf))
    $pageCount = [regex]::Matches($pdfText, '/Type\s*/Page(?!s)\b').Count
    if ($pageCount -lt 1) { throw 'Paper compiled, but its page count could not be determined.' }
    Write-Output "Paper page count: $pageCount"
}
finally { Pop-Location }
