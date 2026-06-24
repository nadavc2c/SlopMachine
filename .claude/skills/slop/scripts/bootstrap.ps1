#requires -Version 5
# slop skill bootstrap (Windows). Idempotent — run once, or after dependency changes.
$ErrorActionPreference = 'Stop'

# Locate the SlopMachine repo root (the dir with pyproject.toml) by walking up from this script.
$dir = $PSScriptRoot
while ($dir -and -not (Test-Path (Join-Path $dir 'pyproject.toml'))) { $dir = Split-Path $dir -Parent }
if (-not $dir) { Write-Error "Could not locate the SlopMachine repo (no pyproject.toml above this skill)."; exit 1 }
Set-Location $dir
Write-Host "SlopMachine repo: $dir"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "Astral 'uv' not found. Install it: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
}

Write-Host "Syncing dependencies (core + 'dance' extra) ..."
uv sync --extra dance

Write-Host "`nEnvironment / GPU check:"
uv run slop info

Write-Host "`nBootstrap complete. Weights download on first use, or pre-fetch:"
Write-Host "  uv run slop models download image       # ~7GB"
Write-Host "  uv run slop models download animation   # ~77GB (dance engine)"
Write-Host "  uv run slop assets download pose         # small ONNX models for dance"
