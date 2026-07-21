param(
    [string]$Output = 'results/foundation/anonymity_audit.json',
    [string]$AnonymousRepositoryUrl = ''
)

$ErrorActionPreference = 'Stop'
$repo = Resolve-Path (Join-Path $PSScriptRoot '..')
Push-Location $repo
try {
    $blockedPatterns = @(
        '[A-Za-z]:[\\/]+Users[\\/]',
        '/(?:home|Users)/[^/\s]+/',
        '@(?:gmail|outlook|hotmail|yahoo)[.]'
    )
    $contentHits = @()
    foreach ($file in @(git ls-files)) {
        if ($file.Replace('\', '/') -eq $Output.Replace('\', '/')) { continue }
        $path = Join-Path $repo $file
        try {
            $content = Get-Content -LiteralPath $path -Raw -Encoding UTF8 -ErrorAction Stop
        }
        catch { continue }
        foreach ($pattern in $blockedPatterns) {
            if ($content -match $pattern) {
                $contentHits += "$file::$pattern"
            }
        }
    }

    $allowedAuthor = 'SpectralGuard Anonymous Authors <anonymous@invalid>'
    $authors = @(git log --format='%an <%ae>' | Sort-Object -Unique)
    $unexpectedAuthors = @($authors | Where-Object { $_ -ne $allowedAuthor })
    # Record only neutral remote names; URLs may identify the repository owner.
    $remotes = @(git remote | Sort-Object -Unique)

    $pdfMetadata = @{}
    $pdfInfoCommand = Get-Command 'pdfinfo' -ErrorAction SilentlyContinue
    if ((Test-Path 'paper/main.pdf') -and $pdfInfoCommand) {
        $previousPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $pdfInfoLines = @(& $pdfInfoCommand.Source (Resolve-Path 'paper/main.pdf') 2>$null)
        $ErrorActionPreference = $previousPreference
        foreach ($line in $pdfInfoLines) {
            if ($line -match '^(Author|Creator|Producer|Title):\s*(.*)$') {
                $pdfMetadata[$Matches[1]] = $Matches[2].Trim()
            }
        }
    }
    $pdfAuthor = [string]$pdfMetadata['Author']
    $pdfAuthorSafe = [string]::IsNullOrWhiteSpace($pdfAuthor) -or $pdfAuthor -eq 'Anonymous'

    $passed = ($contentHits.Count -eq 0) -and
        ($unexpectedAuthors.Count -eq 0) -and $pdfAuthorSafe
    $report = [ordered]@{
        schema_version = 1
        created_at_utc = [DateTime]::UtcNow.ToString('o')
        tracked_content_hits = $contentHits
        git_authors = $authors
        unexpected_git_authors = $unexpectedAuthors
        git_remote_names = $remotes
        pdf_metadata = $pdfMetadata
        local_anonymity_passed = $passed
        source_repository_configured = ($remotes.Count -gt 0)
        anonymous_repository_url = $AnonymousRepositoryUrl
        anonymous_external_repository_published = ($AnonymousRepositoryUrl -match '^https://anonymous[.]4open[.]science/r/[A-Za-z0-9-]+/?$')
    }
    $destination = Join-Path $repo $Output
    New-Item -ItemType Directory -Force -Path (Split-Path $destination) | Out-Null
    $json = $report | ConvertTo-Json -Depth 5
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText(
        $destination,
        $json + [Environment]::NewLine,
        $utf8NoBom
    )
    Write-Output $Output
    if (-not $passed) { exit 1 }
    exit 0
}
finally { Pop-Location }
