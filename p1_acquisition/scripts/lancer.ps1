# P2 - extraction des journaux chez P1.
# Usage (PowerShell, dans ce dossier) :
#   powershell -ExecutionPolicy Bypass -File lancer.ps1 -Pcaps "D:\chemin\vers\les\pcap"
#   powershell -ExecutionPolicy Bypass -File lancer.ps1 -Pcaps "D:\..." -Only "firefox-adguard,firefox-cloudflare" -Out logs_P2_lot2
param([Parameter(Mandatory = $true)][string]$Pcaps, [string]$Only = "", [string]$Out = "logs_P2")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host "Creation de l'environnement Python (une seule fois)..."
    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --quiet pandas pyarrow cryptography
}
$extra = @()
if ($Only) { $extra = @("--only", $Only) }
.\.venv\Scripts\python.exe p1_extract_logs.py --pcaps $Pcaps --out $Out @extra
