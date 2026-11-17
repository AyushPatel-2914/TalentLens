@echo off
REM Public demo: backend on this laptop + free Cloudflare tunnel; prints the link to share.
REM Add "warmup" to re-run the demo examples through the tunnel first:  start_public_demo.cmd warmup
cd /d "%~dp0"
if /I "%1"=="warmup" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\start_public_demo.ps1" -Warmup
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\start_public_demo.ps1"
)
pause
