@echo off
REM Double-click to start the TalentLens demo (backend + frontend + browser).
REM Add "warmup" to pre-run all demo examples first:  start_demo.cmd warmup
cd /d "%~dp0"
if /I "%1"=="warmup" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\start_demo.ps1" -Warmup
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\start_demo.ps1"
)
pause
