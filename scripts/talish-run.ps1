# Talish ApplyPilot unattended local runner
$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$LogDir = Join-Path $HOME ".applypilot\runner-logs"
$Lock = Join-Path $HOME ".applypilot\talish-runner.lock"
$SleepSeconds = 900
$MaxFailures = 3
$Failures = 0
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
if (Test-Path $Lock) {
  $oldPid = Get-Content $Lock -ErrorAction SilentlyContinue
  if ($oldPid -and (Get-Process -Id ([int]$oldPid) -ErrorAction SilentlyContinue)) { throw "Another Talish runner is already active." }
}
$PID | Set-Content $Lock
try {
  if (-not (Test-Path $Python)) { throw "Virtual environment missing. Run talish-setup.ps1 first." }
  while ($true) {
    $log = Join-Path $LogDir ((Get-Date -Format "yyyyMMdd-HHmmss") + ".log")
    Start-Transcript -Path $log -Append | Out-Null
    try {
      Set-Location $RepoRoot
      & $Python -m applypilot.cli run all --workers 2 --min-score 8 --max-age-days 14
      if ($LASTEXITCODE -ne 0) { throw "Preparation pipeline failed." }
      & $Python -m applypilot.cli apply --workers 2 --limit 5 --min-score 8 --max-age-days 14 --no-hitl --no-focus
      & $Python -m applypilot.cli track --days 30
      & $Python -m applypilot.cli interview-prep --limit 10
      & $Python -m applypilot.cli status
      $Failures = 0
    } catch {
      $Failures++
      Write-Warning $_
      if ($Failures -ge $MaxFailures) { break }
    } finally { Stop-Transcript | Out-Null }
    Start-Sleep -Seconds $SleepSeconds
  }
} finally { Remove-Item $Lock -Force -ErrorAction SilentlyContinue }