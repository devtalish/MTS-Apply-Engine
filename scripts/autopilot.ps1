# ApplyPilot unattended cycle for Windows PowerShell.
# Run this from the repository root after the environment is configured.
# CAPTCHA / bot-protection events are NOT bypassed.
# Jobs requiring human intervention are parked.

$ErrorActionPreference = "Continue"

$Workers = 2
$ApplyWorkers = 3
$ApplyBatch = 10
$MinScore = 8
$MaxAgeDays = 14
$SleepSeconds = 900

Write-Host "=== ApplyPilot Autopilot ===" -ForegroundColor Cyan
Write-Host "Remote-first | Pakistan | minimum score $MinScore | batch $ApplyBatch" -ForegroundColor Gray

while ($true) {
    $started = Get-Date

    Write-Host "`n[$started] DISCOVER -> ENRICH -> SCORE -> TAILOR -> COVER" -ForegroundColor Yellow

    & applypilot run all `
        --workers $Workers `
        --min-score $MinScore `
        --max-age-days $MaxAgeDays

    Write-Host "`n[$(Get-Date)] APPLY eligible jobs" -ForegroundColor Yellow

    & applypilot apply `
        --workers $ApplyWorkers `
        --limit $ApplyBatch `
        --min-score $MinScore `
        --max-age-days $MaxAgeDays `
        --no-hitl `
        --no-focus

    Write-Host "`n[$(Get-Date)] TRACK responses" -ForegroundColor Yellow

    & applypilot track --days 30

    Write-Host "`n[$(Get-Date)] STATUS" -ForegroundColor Yellow

    & applypilot status

    Write-Host "`nSleeping $SleepSeconds seconds before the next cycle..." -ForegroundColor DarkGray

    Start-Sleep -Seconds $SleepSeconds
}
