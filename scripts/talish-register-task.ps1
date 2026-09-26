# Talish ApplyPilot: register the local runner at Windows logon.
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Runner = Join-Path $RepoRoot "scripts\talish-run.ps1"
if (-not (Test-Path $Runner)) { throw "Runner not found: $Runner" }
$TaskName = "Talish ApplyPilot Autopilot"
$PowerShell = (Get-Command powershell.exe).Source
$Action = New-ScheduledTaskAction -Execute $PowerShell -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Runner`""
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Days 3650) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5)
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Force | Out-Null
Write-Host "Registered: $TaskName" -ForegroundColor Green
Write-Host "It will start at your next Windows logon. Browser sessions/API credentials must be configured first." -ForegroundColor Yellow