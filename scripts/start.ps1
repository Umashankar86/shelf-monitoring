$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    throw 'Run .\scripts\setup.ps1 first.'
}
Write-Host 'Open http://127.0.0.1:8765 in your browser. Ctrl+C stops the application.'
& '.\.venv\Scripts\python.exe' -m shelfwatch.cli serve

