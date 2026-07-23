<#
.SYNOPSIS
  FoldX: Repair -> N×Optimize -> один AnalyseComplex по финальной структуре, сбор Summary и экспорт таблицы.

.DESCRIPTION
  PowerShell 7+: при -Parallel > 1 — ForEach-Object -Parallel (-ThrottleLimit).
  Windows PowerShell 5.1: при -Parallel > 1 — runspace pool.

  N×Optimize подряд в scratch; в OutputDir сохраняется только финальный PDB (..._RunN_Optimized.pdb),
  без промежуточных Run1..N-1. AnalyseComplex один раз с -Chains (например A,B,C); в Summary
  может быть несколько строк — сумма столбца Interaction Energy в таблицу/Excel.

  Таблица/Excel: Summary_*.fxout для финального прогона (_Run{N}_Optimized, N = -NOptimize).

.EXAMPLE
  pwsh .\foldx_screen.ps1 -InputDir .\pdbs\2WT -OutputDir .\2WT_out -Parallel 8
#>
[CmdletBinding()]
param(
    [string]$InputDir = ".\pdbs\2WT",
    [string]$OutputDir = "",
    [string]$FoldXDir = "",
    [int]$NOptimize = 5,
    [string]$Chains = "A,B",
    [int]$Parallel = 1,
    [string]$OutputXlsx = "",
    [switch]$SkipExcel,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $FoldXDir) {
    $FoldXDir = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
}
$FoldXDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($FoldXDir)
$InputDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($InputDir)

$foldxExe = Join-Path $FoldXDir "FoldX.exe"
if (-not (Test-Path -LiteralPath $foldxExe)) {
    throw "FoldX.exe not found: $foldxExe (use -FoldXDir if exe is elsewhere)"
}

if (-not $OutputDir) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputDir = Join-Path $FoldXDir "screening_out_$stamp"
}
$OutputDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)

if (-not (Test-Path -LiteralPath $InputDir)) {
    throw "InputDir not found: $InputDir"
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# Путь к Excel: не папка и не "...\"
function Resolve-ExcelOutputPath {
    param(
        [string]$Candidate,
        [string]$FallbackDir
    )
    if ([string]::IsNullOrWhiteSpace($Candidate)) {
        return (Join-Path $FallbackDir "FoldX_BindingEnergies.xlsx")
    }
    $p = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Candidate.Trim())
    $sep = [IO.Path]::DirectorySeparatorChar
    while ($p.EndsWith($sep) -or $p.EndsWith('\') -or $p.EndsWith('/')) {
        $p = $p.TrimEnd('\', '/', $sep).Trim()
    }
    if ((Test-Path -LiteralPath $p) -and ((Get-Item -LiteralPath $p).PSIsContainer)) {
        return (Join-Path $p "FoldX_BindingEnergies.xlsx")
    }
    $ext = [IO.Path]::GetExtension($p)
    if ($ext -notin @('.xlsx', '.xls')) {
        return (Join-Path $FallbackDir "FoldX_BindingEnergies.xlsx")
    }
    return $p
}

$OutputXlsx = Resolve-ExcelOutputPath -Candidate $OutputXlsx -FallbackDir $OutputDir

$chainTokens = @($Chains -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($chainTokens.Count -lt 2) {
    throw "Chains: укажите минимум две цепи через запятую (например A,B или A,B,C). Сейчас: '$Chains'"
}
$Chains = $chainTokens -join ','

$scratchRoot = Join-Path $OutputDir "_scratch"
New-Item -ItemType Directory -Path $scratchRoot -Force | Out-Null

function Get-BindingEnergyFromSummaryFxout {
    param([string]$FxoutPath)
    $lines = Get-Content -LiteralPath $FxoutPath
    $ixInteraction = -1
    $inTable = $false
    $sum = 0.0
    foreach ($line in $lines) {
        if ($line -match '^Pdb\s+Group1') {
            $inTable = $true
            $hdr = if ($line.Contains("`t")) { $line -split "`t" } else { $line -split '\s{2,}' }
            for ($k = 0; $k -lt $hdr.Count; $k++) {
                if ($hdr[$k].Trim() -eq 'Interaction Energy') {
                    $ixInteraction = $k
                    break
                }
            }
            if ($ixInteraction -lt 0) { $ixInteraction = 5 }
            continue
        }
        if ($inTable -and $line -match '\S' -and $line -notmatch '^[-=]+$' -and $line -notmatch '^Pdb\s') {
            $cols = if ($line.Contains("`t")) { $line -split "`t" } else { $line -split '\s+' }
            if ($ixInteraction -ge 0 -and $cols.Count -gt $ixInteraction) {
                $sum += [double]$cols[$ixInteraction]
            }
        }
    }
    return $sum
}

# Windows: -Filter не поддерживает * в середине имени — в воркере свои копии функций (нужны для Parallel/runspace).

$TestPdbAnalysisCompleteSb = {
    param(
        [string]$OutputDir,
        [string]$PdbBase,
        [int]$NOptimize
    )
    if (-not (Test-Path -LiteralPath $OutputDir)) { return $false }
    $finalStem = "${PdbBase}_Run${NOptimize}_Optimized"
    $reStem = "(?i)" + [regex]::Escape($finalStem)
    $summaries = @(Get-ChildItem -LiteralPath $OutputDir -Filter "Summary_*.fxout" -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match $reStem })
    return ($summaries.Count -gt 0)
}

$foldxWorker = {
    param(
        [string]$PdbPath,
        [string]$FoldXDir,
        [string]$FoldXExe,
        [string]$OutputDir,
        [string]$ScratchRoot,
        [int]$NOptimize,
        [string]$Chains,
        [bool]$Force,
        [string]$TestCompleteText
    )

    function Get-OptimizedRunPdbRegex {
        param([string]$PdbBase)
        return ('^(?i){0}_Run(\d+)_Optimized\.pdb$' -f [regex]::Escape($PdbBase))
    }

    function Test-FinalOptimizedPdbPresent {
        param([string]$OutputDir, [string]$PdbBase, [int]$NOptimize)
        $re = Get-OptimizedRunPdbRegex -PdbBase $PdbBase
        foreach ($item in Get-ChildItem -LiteralPath $OutputDir -File -ErrorAction SilentlyContinue) {
            $m = [regex]::Match($item.Name, $re)
            if ($m.Success -and [int]$m.Groups[1].Value -eq $NOptimize) {
                return $true
            }
        }
        return $false
    }

    function Get-OptimizedRunPathInOutput {
        param([string]$OutputDir, [string]$PdbBase, [int]$RunIndex)
        $re = Get-OptimizedRunPdbRegex -PdbBase $PdbBase
        $hit = Get-ChildItem -LiteralPath $OutputDir -File -ErrorAction SilentlyContinue |
            Where-Object {
                $m = [regex]::Match($_.Name, $re)
                $m.Success -and ([int]$m.Groups[1].Value -eq $RunIndex)
            } | Select-Object -First 1
        if ($hit) { return $hit.FullName }
        if ($RunIndex -eq 1) {
            $altFoldx = Join-Path $OutputDir ('Optimized_{0}_Repair.pdb' -f $PdbBase)
            if (Test-Path -LiteralPath $altFoldx) { return $altFoldx }
        }
        return $null
    }

    function Test-SummaryExistsForOptimizedStem {
        param([string]$OutputDir, [string]$OptStem)
        $escStem = [regex]::Escape($OptStem)
        $reStem = "(?i)$escStem"
        $any = @(Get-ChildItem -LiteralPath $OutputDir -Filter "Summary_*.fxout" -File -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match $reStem })
        return ($any.Count -gt 0)
    }

    $ErrorActionPreference = "Stop"
    $testSb = [scriptblock]::Create($TestCompleteText)
    $pdbItem = Get-Item -LiteralPath $PdbPath
    $pdbBase = [IO.Path]::GetFileNameWithoutExtension($pdbItem.Name)

    if (-not $Force) {
        $done = & $testSb $OutputDir $pdbBase $NOptimize
        if ($done) {
            Write-Host "[${pdbBase}] skip — комплекс уже проанализирован (Summary в $OutputDir)"
            return
        }
    }

    $tag = [Guid]::NewGuid().ToString("N").Substring(0, 8)
    $work = Join-Path $ScratchRoot "${pdbBase}_$tag"
    New-Item -ItemType Directory -Path $work -Force | Out-Null
    Copy-Item -LiteralPath $FoldXExe -Destination $work -Force
    $rotabase = Join-Path $FoldXDir "rotabase.txt"
    if (Test-Path -LiteralPath $rotabase) {
        Copy-Item -LiteralPath $rotabase -Destination $work -Force
    }

    try {
        $foldxLog = Join-Path $work "foldx_run.log"
        $fx = Join-Path $work "FoldX.exe"
        $localName = $pdbItem.Name

        # --- только AnalyseComplex: все Run*_Optimized.pdb уже в OutputDir ---
        if (-not $Force) {
            $haveFinal = Test-FinalOptimizedPdbPresent -OutputDir $OutputDir -PdbBase $pdbBase -NOptimize $NOptimize
            if ($haveFinal) {
                Write-Host "[${pdbBase}] найден финальный *_Run${NOptimize}_Optimized.pdb в $OutputDir — Repair/Optimize пропускаются"
                $finalPdb = "${pdbBase}_Run${NOptimize}_Optimized.pdb"
                $srcFinal = Get-OptimizedRunPathInOutput -OutputDir $OutputDir -PdbBase $pdbBase -RunIndex $NOptimize
                Copy-Item -LiteralPath $srcFinal -Destination (Join-Path $work $finalPdb) -Force
                Push-Location $work
                try {
                    $finalStem = [IO.Path]::GetFileNameWithoutExtension($finalPdb)
                    if (-not $Force -and (Test-SummaryExistsForOptimizedStem -OutputDir $OutputDir -OptStem $finalStem)) {
                        Write-Host "[${pdbBase}] skip AnalyseComplex — $finalPdb (Summary уже есть)"
                    }
                    else {
                        Write-Host "[${pdbBase}] AnalyseComplex (финал, цепи $Chains): $finalPdb"
                        & $fx --command=AnalyseComplex --pdb="$finalPdb" --analyseComplexChains="$Chains" *>> $foldxLog
                    }
                }
                finally {
                    Pop-Location
                }
                Get-ChildItem -LiteralPath $work -File -ErrorAction SilentlyContinue | ForEach-Object {
                    if ($_.Extension -eq '.fxout') {
                        Move-Item -LiteralPath $_.FullName -Destination (Join-Path $OutputDir $_.Name) -Force
                    }
                }
                return
            }
        }

        if (-not $Force) {
            $reDiag = Get-OptimizedRunPdbRegex -PdbBase $pdbBase
            $haveFinalOut = $false
            Get-ChildItem -LiteralPath $OutputDir -File -ErrorAction SilentlyContinue | ForEach-Object {
                $md = [regex]::Match($_.Name, $reDiag)
                if ($md.Success -and [int]$md.Groups[1].Value -eq $NOptimize) { $haveFinalOut = $true }
            }
            if (-not $haveFinalOut) {
                Write-Host "[${pdbBase}] в $OutputDir нет финального *_Run${NOptimize}_Optimized.pdb — Repair→Optimize (или укажите -NOptimize под уже сохранённый финальный прогон)."
            }
        }

        # --- полный конвейер Repair -> Optimize -> AnalyseComplex ---
        Copy-Item -LiteralPath $pdbItem.FullName -Destination (Join-Path $work $localName) -Force
        $finalRunFile = $localName -replace '\.pdb$', "_Run${NOptimize}_Optimized.pdb"
        $repaired = $null
        Push-Location $work
        try {
            Write-Host "[${pdbBase}] Repair..."
            & $fx --command=RepairPDB --pdb="$localName" *>> $foldxLog
            $repaired = $localName -replace '\.pdb$', '_Repair.pdb'
            if (-not (Test-Path -LiteralPath $repaired)) {
                throw "RepairPDB failed for $localName"
            }

            for ($run = 1; $run -le $NOptimize; $run++) {
                $runFile = $localName -replace '\.pdb$', "_Run${run}_Optimized.pdb"
                $existingOpt = $null
                if (-not $Force -and $run -eq $NOptimize) {
                    $existingOpt = Get-OptimizedRunPathInOutput -OutputDir $OutputDir -PdbBase $pdbBase -RunIndex $NOptimize
                }
                if ($existingOpt) {
                    Write-Host "[${pdbBase}] Optimize $run / $NOptimize — $runFile уже в OutputDir, пропуск Optimize"
                    Copy-Item -LiteralPath $existingOpt -Destination (Join-Path $work $runFile) -Force
                    continue
                }
                Write-Host "[${pdbBase}] Optimize $run / $NOptimize"
                $repairedStem = [IO.Path]::GetFileNameWithoutExtension($repaired)
                $optimizedModern = "Optimized_$repairedStem.pdb"
                $optimizedLegacy = $localName -replace '\.pdb$', '_Repair_Optimized.pdb'
                foreach ($o in @($optimizedModern, $optimizedLegacy)) {
                    if (Test-Path -LiteralPath $o) {
                        Remove-Item -LiteralPath $o -Force -ErrorAction SilentlyContinue
                    }
                }
                & $fx --command=Optimize --pdb="$repaired" *>> $foldxLog
                $picked = $null
                foreach ($cand in @($optimizedLegacy, $optimizedModern)) {
                    if (Test-Path -LiteralPath $cand) {
                        $picked = $cand
                        break
                    }
                }
                if (-not $picked) {
                    $fallback = @(Get-ChildItem -LiteralPath $work -Filter "Optimized_*.pdb" -File -ErrorAction SilentlyContinue |
                            Sort-Object LastWriteTime -Descending | Select-Object -First 1)
                    if ($fallback.Count -gt 0) { $picked = $fallback[0].Name }
                }
                if ($picked) {
                    Rename-Item -LiteralPath $picked -NewName $runFile
                    if ($run -lt $NOptimize) {
                        Remove-Item -LiteralPath (Join-Path $work $runFile) -Force -ErrorAction SilentlyContinue
                    }
                }
                else {
                    Write-Warning ("[{0}] Optimize {1}/{2}: нет выходного PDB (ожидали {3} или {4}). См. лог: {5}" -f $pdbBase, $run, $NOptimize, $optimizedLegacy, $optimizedModern, $foldxLog)
                }
            }

            $finalPath = Join-Path $work $finalRunFile
            if (-not (Test-Path -LiteralPath $finalPath)) {
                $pullFinal = if (-not $Force) { Get-OptimizedRunPathInOutput -OutputDir $OutputDir -PdbBase $pdbBase -RunIndex $NOptimize } else { $null }
                if ($pullFinal) {
                    Copy-Item -LiteralPath $pullFinal -Destination $finalPath -Force
                }
            }
            if (-not (Test-Path -LiteralPath $finalPath)) {
                Write-Warning ("[{0}] нет финального {1} в рабочей папке после Optimize — AnalyseComplex не будет вызван. См. {2}" -f $pdbBase, $finalRunFile, $foldxLog)
            }
            else {
                $finalStem = [IO.Path]::GetFileNameWithoutExtension($finalRunFile)
                if (-not $Force -and (Test-SummaryExistsForOptimizedStem -OutputDir $OutputDir -OptStem $finalStem)) {
                    Write-Host "[${pdbBase}] skip AnalyseComplex — $finalRunFile (Summary уже есть)"
                }
                else {
                    Write-Host "[${pdbBase}] AnalyseComplex (финал, цепи $Chains): $finalRunFile"
                    & $fx --command=AnalyseComplex --pdb="$finalRunFile" --analyseComplexChains="$Chains" *>> $foldxLog
                }
            }
        }
        finally {
            Pop-Location
        }

        Remove-Item -LiteralPath (Join-Path $work $localName) -ErrorAction SilentlyContinue
        $repairedPath = Join-Path $work $repaired
        if ($null -ne $repaired -and (Test-Path -LiteralPath $repairedPath)) {
            Remove-Item -LiteralPath $repairedPath -Force -ErrorAction SilentlyContinue
        }
        Get-ChildItem -LiteralPath $work -File -ErrorAction SilentlyContinue | ForEach-Object {
            if ($_.Extension -eq '.fxout') {
                Move-Item -LiteralPath $_.FullName -Destination (Join-Path $OutputDir $_.Name) -Force
            }
            elseif ($_.Extension -eq '.pdb' -and $_.Name -eq $finalRunFile) {
                Move-Item -LiteralPath $_.FullName -Destination (Join-Path $OutputDir $_.Name) -Force
            }
        }
    }
    finally {
        Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
    }
}

$pdbs = @(Get-ChildItem -LiteralPath $InputDir -Filter "*.pdb" -File | Sort-Object Name)
if ($pdbs.Count -eq 0) {
    throw "No PDB files in $InputDir"
}

$useParallel = $Parallel -gt 1
$isPs7 = $PSVersionTable.PSVersion.Major -ge 7
$forceBool = [bool]$Force

$testPdbCompleteText = $TestPdbAnalysisCompleteSb.ToString()
$workerArgs = @(
    $FoldXDir,
    $foldxExe,
    $OutputDir,
    $scratchRoot,
    $NOptimize,
    $Chains,
    $forceBool,
    $testPdbCompleteText
)

$sw = [System.Diagnostics.Stopwatch]::StartNew()

if (-not $useParallel) {
    foreach ($pdb in $pdbs) {
        & $foldxWorker $pdb.FullName @workerArgs
    }
}
elseif ($isPs7) {
    $foldxWorkerText = $foldxWorker.ToString()
    $null = $pdbs | ForEach-Object -ThrottleLimit $Parallel -Parallel {
        $w = [scriptblock]::Create($using:foldxWorkerText)
        & $w $_.FullName $using:FoldXDir $using:foldxExe $using:OutputDir $using:scratchRoot $using:NOptimize $using:Chains $using:forceBool $using:testPdbCompleteText
    }
}
else {
    $pool = [runspacefactory]::CreateRunspacePool(1, [Math]::Max(1, $Parallel))
    $pool.Open()
    $instances = [System.Collections.ArrayList]::new()
    try {
        foreach ($pdb in $pdbs) {
            while ($true) {
                $busy = 0
                foreach ($i in $instances) {
                    if (-not $i.Handle.IsCompleted) { $busy++ }
                }
                if ($busy -lt $Parallel) { break }
                Start-Sleep -Milliseconds 150
            }
            $ps = [powershell]::Create()
            $ps.RunspacePool = $pool
            [void]$ps.AddScript($foldxWorker).
                AddArgument($pdb.FullName).
                AddArgument($FoldXDir).
                AddArgument($foldxExe).
                AddArgument($OutputDir).
                AddArgument($scratchRoot).
                AddArgument($NOptimize).
                AddArgument($Chains).
                AddArgument($forceBool).
                AddArgument($testPdbCompleteText)
            $handle = $ps.BeginInvoke()
            [void]$instances.Add([PSCustomObject]@{ Pipe = $ps; Handle = $handle })
        }
        foreach ($i in $instances) {
            try {
                $null = $i.Pipe.EndInvoke($i.Handle)
            }
            catch {
                Write-Error $_
            }
            finally {
                $i.Pipe.Dispose()
            }
        }
    }
    finally {
        $pool.Close()
        $pool.Dispose()
    }
}

$sw.Stop()
Write-Host ("Finished FoldX jobs in {0:g}" -f $sw.Elapsed)

$finalRunToken = "_Run${NOptimize}_Optimized"
$tokRe = "(?i)" + [regex]::Escape($finalRunToken)
$results = @(Get-ChildItem -LiteralPath $OutputDir -Filter "Summary_*.fxout" -File |
        Where-Object { $_.Name -match $tokRe } | ForEach-Object {
        $cluster = ""
        if ($_.Name -match 'cluster_(\d+)') { $cluster = "cluster_$($matches[1])" }
        [PSCustomObject]@{
            Cluster       = $cluster
            File          = $_.Name
            BindingEnergy = (Get-BindingEnergyFromSummaryFxout -FxoutPath $_.FullName)
        }
    } | Sort-Object BindingEnergy)

$results | Format-Table -AutoSize

if (-not $SkipExcel) {
    $excelMod = Get-Module -ListAvailable -Name ImportExcel
    if ($excelMod) {
        Import-Module ImportExcel
        $parent = Split-Path -Parent $OutputXlsx
        if (-not [string]::IsNullOrWhiteSpace($parent)) {
            New-Item -ItemType Directory -Path $parent -Force -ErrorAction SilentlyContinue | Out-Null
        }
        $results | Export-Excel -Path $OutputXlsx -WorksheetName "FoldX_Summary" -AutoSize -TableName "BindingEnergies"
        Write-Host "Excel: $OutputXlsx"
    }
    else {
        $csvPath = [IO.Path]::ChangeExtension($OutputXlsx, "csv")
        $results | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8
        Write-Warning "Module ImportExcel not installed; wrote CSV instead: $csvPath (Install-Module ImportExcel for .xlsx)"
    }
}

Write-Host "Artifacts in: $OutputDir"
