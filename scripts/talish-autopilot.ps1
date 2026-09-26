# Talish ApplyPilot local autopilot
# Intended for the owner's Windows machine only.
# Do not run this repository's auto-apply workflow on an untrusted/public self-hosted runner.

$ErrorActionPreference = "Stop"
$Workers = 2
$ApplyBatch = 5
$MinScore = 8
$MaxAgeDays = 14
$SleepSeconds = 900

Write-Host "=== Talish ApplyPilot Autopilot ===" -ForegroundColor Cyan
Write-Host "Remote-first | Pakistan | 5-14h primary | <=20h absolute | score >= $MinScore" -ForegroundColor Gray

while ($true) {
    $started = Get-Date
    Write-Host "[$started] DISCOVER -> ENRICH -> SCORE -> TAILOR -> COVER" -ForegroundColor Yellow
    & applypilot run all --workers $Workers --min-score $MinScore --max-age-days $MaxAgeDays

    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Preparation pipeline failed; skipping this apply cycle."
    } else {
        Write-Host "[$(Get-Date)] APPLY eligible jobs" -ForegroundColor Yellow
        # Human-required cases are parked, not bypassed.
        & applypilot apply --workers $Workers --limit $ApplyBatch --min-score $MinScore --max-age-days $MaxAgeDays --no-hitl --no-focus
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Apply cycle returned a non-zero exit code; inspect status before continuing."
        }
    }

    Write-Host "[$(Get-Date)] TRACK responses" -ForegroundColor Yellow
    & applypilot track --days 30
    Write-Host "[$(Get-Date)] INTERVIEW PREP" -ForegroundColor Yellow
    & applypilot interview-prep --limit 10
    Write-Host "[$(Get-Date)] STATUS" -ForegroundColor Yellow
    & applypilot status
    Write-Host "Sleeping $SleepSeconds seconds..." -ForegroundColor DarkGray
    Start-Sleep -Seconds $SleepSeconds
}
