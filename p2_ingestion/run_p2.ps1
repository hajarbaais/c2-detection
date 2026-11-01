# P2 - chaîne complète : extraction tshark -> sessions -> découpage -> vérifications
$ErrorActionPreference = "Stop"
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$s = Join-Path $PSScriptRoot "scripts"
& $py -I "$s\p2_01_extract.py" --workers 4; if (-not $?) { exit 1 }
& $py -I "$s\p2_02_sessions.py" --window 3600; if (-not $?) { exit 1 }
& $py -I "$s\p2_03_split.py"; if (-not $?) { exit 1 }
& $py -I "$s\p2_04_check.py"
