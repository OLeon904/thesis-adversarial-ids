# Connect thesis-adversarial-ids to GitHub and push.
# Run from repo root after: gh auth login

param(
    [string]$RepoName = "thesis-adversarial-ids",
    [ValidateSet("public", "private")]
    [string]$Visibility = "private"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Refresh PATH so gh is found after winget install
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [System.Environment]::GetEnvironmentVariable("Path", "User")

Write-Host "Checking GitHub authentication..."
gh auth status
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Not logged in. Run this first (follow the prompts):"
    Write-Host "  gh auth login"
    Write-Host ""
    Write-Host "Choose: GitHub.com -> HTTPS -> Login with a web browser"
    exit 1
}

$login = gh api user -q .login
Write-Host "Logged in as: $login"

Write-Host "Creating repo $login/$RepoName ($Visibility) and pushing..."
gh repo create $RepoName `
    --$Visibility `
    --source . `
    --remote origin `
    --description "Masters thesis: adversarial robustness of ML-based NIDS on CICIDS2017" `
    --push

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Done. Repository URL:"
    gh repo view --web 2>$null
    Write-Host "https://github.com/$login/$RepoName"
}
