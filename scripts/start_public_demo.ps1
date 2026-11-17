# Public demo: run the backend on this laptop and expose it through a free Cloudflare quick tunnel.
# No Cloudflare account, no card. The frontend stays on Vercel; this script prints the link to share.
#
# Usage (from the project folder):
#     start_public_demo.cmd                      # or: .\scripts\start_public_demo.ps1
#     .\scripts\start_public_demo.ps1 -Warmup    # also re-runs the demo examples through the tunnel
#
# Put your Vercel URL in .env once:   PUBLIC_FRONTEND_URL=https://talentlens-xxxx.vercel.app
# Stop: press Ctrl+C in this window (closes the tunnel), then close the API window.

param([switch]$Warmup, [string]$FrontendUrl = "")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Python venv not found at $python." }

# Frontend URL: parameter, else PUBLIC_FRONTEND_URL in .env.
if (-not $FrontendUrl -and (Test-Path (Join-Path $root ".env"))) {
    $line = Get-Content (Join-Path $root ".env") | Where-Object { $_ -match '^\s*PUBLIC_FRONTEND_URL\s*=' } | Select-Object -First 1
    if ($line) { $FrontendUrl = ($line -split '=', 2)[1].Trim().Trim('"').Trim("'") }
}
$FrontendUrl = $FrontendUrl.TrimEnd('/')

# 1) cloudflared: a single portable exe from Cloudflare's official GitHub releases.
$tools = Join-Path $env:USERPROFILE "tools"
$cf = Join-Path $tools "cloudflared.exe"
if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue) -and -not (Test-Path $cf)) {
    Write-Host "Downloading cloudflared (one time, about 60 MB) ..."
    New-Item -ItemType Directory -Force $tools | Out-Null
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile $cf -UseBasicParsing
}
if (Test-Path $cf) { $cfExe = $cf } else { $cfExe = (Get-Command cloudflared).Source }

# 2) Backend in public-demo mode (per-visitor rate limits; CORS for the Vercel frontend).
$already = $null
try { $already = Invoke-RestMethod http://localhost:8000/health -TimeoutSec 3 } catch { }
if ($already -and -not $already.public_demo) {
    throw "A local API (from start_demo) is already running on port 8000 without public rate limits. Close its window first, then run this again."
}
if (-not $already) {
    Write-Host "Starting API on http://localhost:8000 (public-demo mode) ..."
    $apiEnv = "`$env:HF_HUB_DISABLE_XET='1'; `$env:PYTHONIOENCODING='utf-8'; `$env:TALENTLENS_PUBLIC_DEMO='1';" +
              " `$env:ALLOWED_ORIGINS='http://localhost:3000,$FrontendUrl';`$env:ALLOWED_ORIGIN_REGEX='https://talentlens[a-z0-9-]*\.vercel\.app';"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "$apiEnv Set-Location '$root'; & '$python' -m uvicorn talentlens.api.main:app --port 8000"
    Write-Host "Waiting for the API to load its models (about 30-60 s) ..."
    $deadline = (Get-Date).AddMinutes(5)
    while ((Get-Date) -lt $deadline -and -not $already) {
        try { $already = Invoke-RestMethod http://localhost:8000/health -TimeoutSec 5 } catch { Start-Sleep -Seconds 3 }
    }
    if (-not $already) { throw "API did not start within 5 minutes. Check the API window for errors." }
}
Write-Host ("API ready: {0} resumes, {1} jobs, public_demo={2}" -f $already.resumes, $already.jobs, $already.public_demo)

# 3) Quick tunnel: prints a random https://<words>.trycloudflare.com URL to its log.
$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir "tunnel.log"
if (Test-Path $log) { Remove-Item $log -Force }
Write-Host "Opening the Cloudflare tunnel ..."
$tunnel = Start-Process $cfExe -ArgumentList "tunnel", "--no-autoupdate", "--url", "http://localhost:8000" `
          -RedirectStandardError $log -WindowStyle Hidden -PassThru

try {
    $url = $null
    $deadline = (Get-Date).AddMinutes(2)
    while ((Get-Date) -lt $deadline -and -not $url) {
        Start-Sleep -Seconds 1
        if ($tunnel.HasExited) { throw "cloudflared exited. See $log" }
        if (Test-Path $log) {
            $m = Select-String -Path $log -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' | Select-Object -First 1
            if ($m) { $url = $m.Matches[0].Value }
        }
    }
    if (-not $url) { throw "No tunnel URL after 2 minutes. See $log" }

    Write-Host "Tunnel: $url  (checking it answers ...)"
    $ok = $false
    $deadline = (Get-Date).AddMinutes(2)
    while ((Get-Date) -lt $deadline -and -not $ok) {
        try { Invoke-RestMethod "$url/health" -TimeoutSec 10 | Out-Null; $ok = $true } catch { Start-Sleep -Seconds 3 }
    }
    if (-not $ok) { Write-Warning "The tunnel doesn't answer yet; new tunnels can take a minute to resolve. The link below should work shortly." }

    if ($Warmup) { & $python (Join-Path $root "scripts\warmup_demo.py") --api $url }

    Write-Host ""
    if ($FrontendUrl) {
        $share = "$FrontendUrl/?api=$url"
        try { Set-Clipboard $share } catch { }
        Write-Host "=== Share this link (copied to the clipboard) ===" -ForegroundColor Green
        Write-Host "    $share" -ForegroundColor Green
    } else {
        Write-Host "Backend URL: $url" -ForegroundColor Green
        Write-Host "Add PUBLIC_FRONTEND_URL=https://<your-app>.vercel.app to .env to get a ready-to-share link." -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "The link works while this window and the API window stay open (and the laptop stays awake)."
    Write-Host "Press Ctrl+C here to close the tunnel."
    Wait-Process -Id $tunnel.Id
} finally {
    if (-not $tunnel.HasExited) { Stop-Process -Id $tunnel.Id -Force }
    Write-Host "Tunnel closed."
}
