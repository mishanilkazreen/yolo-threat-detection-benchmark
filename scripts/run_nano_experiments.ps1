# Runs all 8 nano model experiments sequentially, then generates the report.
# Each experiment logs to logs/<config>.log.
# If a config output directory already has final_test_metrics.json, it is skipped.

$ErrorActionPreference = "Continue"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$logsDir = Join-Path $repoRoot "logs"
if (-not (Test-Path $logsDir)) { New-Item -ItemType Directory -Path $logsDir | Out-Null }

$configs = @(
    "yolov8n_random",
    "yolov8n_pretrained",
    "yolo12n_random",
    "yolo12n_pretrained",
    "yolov8n_random_100ep",
    "yolov8n_pretrained_100ep",
    "yolo12n_random_100ep",
    "yolo12n_pretrained_100ep"
)

$startedAt = Get-Date
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  Nano experiments started at $startedAt" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

foreach ($cfg in $configs) {
    $configPath = "config/models/$cfg.yaml"
    $resultMarker = "outputs/$cfg/final_test_metrics.json"
    $logFile = Join-Path $logsDir "$cfg.log"

    if (Test-Path $resultMarker) {
        Write-Host "[SKIP] $cfg already has $resultMarker" -ForegroundColor Yellow
        continue
    }

    Write-Host ""
    Write-Host "[RUN ] $cfg -- log: $logFile" -ForegroundColor Green
    $runStart = Get-Date

    uv run python scripts/train_model.py $configPath *>&1 | Tee-Object -FilePath $logFile

    $runEnd = Get-Date
    $elapsed = ($runEnd - $runStart).TotalMinutes
    Write-Host "[DONE] $cfg finished in $([math]::Round($elapsed, 1)) min" -ForegroundColor Green
}

$endedAt = Get-Date
$totalMin = ($endedAt - $startedAt).TotalMinutes
Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  All nano experiments done in $([math]::Round($totalMin, 1)) min" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "Generating report..." -ForegroundColor Cyan
$reportLog = Join-Path $logsDir "nano_report.log"
uv run python scripts/report_results.py --phase nano | Tee-Object -FilePath $reportLog
