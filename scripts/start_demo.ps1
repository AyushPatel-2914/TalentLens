# Start the TalentLens demo: backend (FastAPI) + frontend (Next.js) in two windows, then open the browser.
#
# Usage (from the project folder, in PowerShell):
#     .\scripts\start_demo.ps1            # starts both, waits for the API, opens http://localhost:3000
#     .\scripts\start_demo.ps1 -Warmup    # also pre-runs all demo examples (do this before an interview)
#
# Stop: close the two windows (or press Ctrl+C in each).

param([switch]$Warmup, [switch]$NoReranker)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$node = Join-Path $env:USERPROFILE "tools\node"
if (-not (Test-Path $python)) { throw "Python venv not found at $python. Run setup from README section 8." }
if (Test-Path $node) { $env:PATH = "$node;$env:PATH" }
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { throw "npm not found. Install Node.js 22 or put a portable copy in $node." }

$apiEnv = "`$env:HF_HUB_DISABLE_XET='1'; `$env:PYTHONIOENCODING='utf-8';"
if ($NoReranker) { $apiEnv += " `$env:TALENTLENS_NO_RERANKER='1';" }

Write-Host "Starting API on http://localhost:8000 ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "$apiEnv Set-Location '$root'; & '$python' -m uvicorn talentlens.api.main:app --port 8000"

Write-Host "Starting frontend on http://localhost:3000 ..."
$fe = Join-Path $root "frontend"
if (-not (Test-Path (Join-Path $fe "node_modules"))) { Push-Location $fe; npm install --no-audit --no-fund; Pop-Location }
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$env:PATH='$node;' + `$env:PATH; Set-Location '$fe'; npm run dev"

Write-Host "Waiting for the API to load its models (about 1 minute) ..."
$deadline = (Get-Date).AddMinutes(5)
while ((Get-Date) -lt $deadline) {
    try { $h = Invoke-RestMethod http://localhost:8000/health -TimeoutSec 5; break } catch { Start-Sleep -Seconds 3 }
}
if (-not $h) { throw "API did not start within 5 minutes. Check the API window for errors." }
Write-Host ("API ready: reranker={0}, LLM={1}, {2} resumes" -f $h.reranker, $h.llm_enabled, $h.resumes)

if ($Warmup) { & $python (Join-Path $root "scripts\warmup_demo.py") }
Start-Process "http://localhost:3000"
