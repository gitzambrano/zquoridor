# run_wallext.ps1 -- orchestrate pairwise matches for the wall extension
# experiment. Runs K matches concurrently (default 3), each a separate
# wallext_arena.exe process pinned to one core. Usage:
#   powershell -File tools\wallext\run_wallext.ps1 -Mode heuristic -TimeMs 100 -Games 300 -Tag screen
#   powershell -File tools\wallext\run_wallext.ps1 -Mode nnue -TimeMs 150 -Games 400 -Tag confirm
param(
    [string]$Mode = "heuristic",
    [int]$TimeMs = 100,
    [int]$Games = 300,
    [int]$Concurrent = 3,
    [string]$Tag = "screen",
    [string[]]$Variants = @("b1t2", "b1t4", "b1t6", "b2t2", "b2t4", "b2t6", "b3t2", "b3t4", "b3t6", "off")
)

$ErrorActionPreference = "Stop"
# Accept "-Variants b1t4,b3t6" (one comma-joined string, the shape that
# survives Start-Process -ArgumentList) as well as a real string array.
$Variants = @($Variants | ForEach-Object { $_ -split "[,\s]+" } | Where-Object { $_ })
$root = $PSScriptRoot | Split-Path | Split-Path   # repo root (tools/wallext -> root)
$bin = Join-Path $root "bin\wallext_arena.exe"
$outDir = Join-Path $root "data\wallext"
if (!(Test-Path $outDir)) { New-Item -ItemType Directory $outDir | Out-Null }

$jobs = @()
foreach ($v in $Variants) {
    $bonus = -1; $thr = -1; $maxExtra = -1
    if ($v -match "^b(\d+)t(\d+)$") { $bonus = [int]$Matches[1]; $thr = [int]$Matches[2] }
    elseif ($v -eq "off") { $maxExtra = 0 }
    else { throw "variante desconhecida: $v" }

    # Different start offsets spread each variant over a different corpus
    # slice; seeds stay fixed so every run of this script is reproducible.
    $seedOff = 17 * (1 + [array]::IndexOf($Variants, $v))
    $log = Join-Path $outDir ("{0}_{1}_{2}.log" -f $Tag, $Mode, $v)
    # --depth must come before --time-ms: the arena's parser treats a
    # trailing --depth as "fixed-depth mode", and this runner always wants
    # the time control to win.
    $args = @("--games", "$Games", "--depth", "8", "--time-ms", "$TimeMs",
              "--mode", $Mode, "--start", "$seedOff", "--report-every", "50")
    if ($bonus -ge 0) { $args += @("--bonus", "$bonus", "--threshold", "$thr") }
    if ($maxExtra -ge 0) { $args += @("--max-extra", "0") }
    $p = Start-Process -FilePath $bin -ArgumentList $args -WorkingDirectory $root `
         -RedirectStandardOutput $log -RedirectStandardError "$log.err" -NoNewWindow -PassThru
    $jobs += @{ Name = $v; Proc = $p; Log = $log }
    while (@($jobs | Where-Object { !$_.Proc.HasExited }).Count -ge $Concurrent) {
        Start-Sleep -Seconds 10
    }
}
foreach ($j in $jobs) {
    if (!$j.Proc.HasExited) { Wait-Process -Id $j.Proc.Id -ErrorAction SilentlyContinue }
}

Write-Output "=== RESUMO ($Tag / $Mode / ${TimeMs}ms / ${Games} jogos) ==="
foreach ($j in $jobs) {
    $last = Get-Content $j.Log | Select-String "resultado final" -Context 0,3 | Select-Object -Last 1
    Write-Output ("--- {0}" -f $j.Name)
    if ($last) { $last.Context.PostContext | ForEach-Object { Write-Output $_ } }
}
