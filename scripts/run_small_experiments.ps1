# Runs all 8 small model experiments sequentially, then generates the report.
# Captures full stdout+stderr per config to logs/<config>.log and an
# aggregate master log to logs/small_master.log with timestamps.
#
# NOTE: Small models take significantly longer than nano variants.

$ErrorActionPreference = "Continue"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$logsDir = Join-Path $repoRoot "logs"
if (-not (Test-Path $logsDir)) { New-Item -ItemType Directory -Path $logsDir | Out-Null }

$masterLog = Join-Path $logsDir "small_master.log"
$reportLog = Join-Path $logsDir "small_report.log"

function Write-Both {
    param([string]$Message, [string]$Color = "White")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $Message"
    Write-Host $line -ForegroundColor $Color
    Add-Content -Path $masterLog -Value $line -Encoding utf8
}

$configs = @(
    "yolov8s_random",
    "yolov8s_pretrained",
    "yolo12s_random",
    "yolo12s_pretrained",
    "yolov8s_random_100ep",
    "yolov8s_pretrained_100ep",
    "yolo12s_random_100ep",
    "yolo12s_pretrained_100ep"
)

$startedAt = Get-Date
Write-Both "===============================================" "Cyan"
Write-Both "  Small experiments started" "Cyan"
Write-Both "===============================================" "Cyan"

$completed = 0
$skipped = 0
$failed = @()

foreach ($cfg in $configs) {
    $configPath = "config/models/$cfg.yaml"
    $resultMarker = "outputs/$cfg/final_test_metrics.json"
    $logFile = Join-Path $logsDir "$cfg.log"

    if (Test-Path $resultMarker) {
        Write-Both "[SKIP] $cfg already has $resultMarker" "Yellow"
        $skipped++
        continue
    }

    Write-Both "" "White"
    Write-Both "[RUN ] $cfg  (log: $logFile)" "Green"
    $runStart = Get-Date

    & uv run python scripts/train_model.py $configPath 2>&1 | Tee-Object -FilePath $logFile

    $exitCode = $LASTEXITCODE
    $runEnd = Get-Date
    $elapsed = ($runEnd - $runStart).TotalMinutes

    if ($exitCode -eq 0) {
        Write-Both "[DONE] $cfg finished in $([math]::Round($elapsed, 1)) min" "Green"
        $completed++
    } else {
        Write-Both "[FAIL] $cfg exited with code $exitCode after $([math]::Round($elapsed, 1)) min" "Red"
        $failed += $cfg
    }
}

$endedAt = Get-Date
$totalMin = ($endedAt - $startedAt).TotalMinutes

Write-Both "" "White"
Write-Both "===============================================" "Cyan"
Write-Both "  All small experiments done in $([math]::Round($totalMin, 1)) min" "Cyan"
Write-Both "  Completed: $completed  Skipped: $skipped  Failed: $($failed.Count)" "Cyan"
if ($failed.Count -gt 0) {
    Write-Both "  Failed configs: $($failed -join ', ')" "Red"
}
Write-Both "===============================================" "Cyan"

Write-Both "" "White"
Write-Both "Generating report..." "Cyan"
& uv run python scripts/report_results.py --phase small 2>&1 | Tee-Object -FilePath $reportLog

Write-Both "" "White"
Write-Both "Master log: $masterLog" "Cyan"
Write-Both "Per-config logs: $logsDir\<config>.log" "Cyan"
Write-Both "Report: $reportLog" "Cyan"
