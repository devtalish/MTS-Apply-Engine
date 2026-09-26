# Talish ApplyPilot one-time Windows setup
$ErrorActionPreference = "Stop"
$RepoUrl = "https://github.com/devtalish/MTS-Apply-Engine.git"
$InstallDir = Join-Path $HOME "MTS-Apply-Engine"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw "Python launcher (py) is required." }
& py -3.11 --version
if ($LASTEXITCODE -ne 0) { throw "Python 3.11+ is required." }

if (-not (Test-Path $InstallDir)) {
  git clone $RepoUrl $InstallDir
} else {
  Push-Location $InstallDir
  try { git fetch origin; git checkout main; git pull --ff-only origin main } finally { Pop-Location }
}

Push-Location $InstallDir
try {
  if (-not (Test-Path ".venv\Scripts\python.exe")) { & py -3.11 -m venv .venv }
  $Python = Join-Path $InstallDir ".venv\Scripts\python.exe"
  & $Python -m pip install --upgrade pip
  & $Python -m pip install -e .
  & $Python -m compileall -q src
  & $Python -m applypilot.cli --help | Out-Null
  Write-Host "Setup and syntax check complete." -ForegroundColor Green
  Write-Host "Next: run applypilot init, configure your local profile + LLM key, authenticate browser sessions, then do ONE dry-run." -ForegroundColor Yellow
} finally { Pop-Location }