# Local sequential benchmark queue. Edit the variables below for a new run.
$ErrorActionPreference = 'Stop'
$Repo = 'C:\Projetos\Zquoridor'
$WaitForPid = 18996
$Pairs = 20
$MoveTimeMs = 200
$Seed = 20260918
$Openings = 'tools\external\openings_titanium.jsonl'
$ClaustrophobiaDevice = 'cpu'
$Tasks = @(
    @{ Name = 'race512-search10-ft'; Executable = 'results\experiments\race512-search10-ft-s20260917\zquoridor.exe'; Nnue = 'results\experiments\race512-search10-ft-s20260917\student_int8.bin' },
    @{ Name = 'base512-search10-ft'; Executable = 'results\experiments\base512-search10-ft-s20260917\zquoridor.exe'; Nnue = 'results\experiments\base512-search10-ft-s20260917\student_int8.bin' }
)
while (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 5
}
foreach ($Task in $Tasks) {
    $Output = "results\benchmarks\$($Task.Name)-200ms-s20260918"
    New-Item -ItemType Directory -Force -Path (Join-Path $Repo $Output) | Out-Null
    & python tools\benchmark_candidate.py `
        --candidate-executable $Task.Executable `
        --candidate-nnue $Task.Nnue `
        --pairs $Pairs --move-time-ms $MoveTimeMs --workers 1 --seed $Seed `
        --openings $Openings --output $Output `
        --claustrophobia-device $ClaustrophobiaDevice `
        *> (Join-Path $Repo "$Output\queue.log")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
