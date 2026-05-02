# ====== foldx_screen.ps1 (fixed) ======
$foldxExe = ".\FoldX.exe"
$inputDir = ".\pdbs"                      # folder with your relaxed PDBs
$workDir = (Get-Location).Path             # where the script runs
$nOptimize = 5
$chains = "A,B"                            # change for your oligomer (e.g., A,B,C)
$foldxLog = "foldx_run.log"

# Get all PDB files from input directory
$pdbs = Get-ChildItem -Path $inputDir -Filter "*.pdb" | Sort-Object Name
if ($pdbs.Count -eq 0) {
    Write-Error "No PDB files found in $inputDir"
    exit 1
}

foreach ($pdb in $pdbs) {
    # Copy file to current working directory
    $localName = $pdb.Name
    Copy-Item $pdb.FullName -Destination $workDir -Force
    if (-not (Test-Path $localName)) {
        Write-Warning "Could not copy $($pdb.Name), skipping."
        continue
    }

    Write-Host "Processing $localName ..."

    # ----- 1. RepairPDB -----
    Write-Host "  Repairing..."
    & $foldxExe --command=RepairPDB --pdb="$localName" *>> $foldxLog
    $repaired = $localName -replace '\.pdb$', '_Repair.pdb'

    if (-not (Test-Path $repaired)) {
        Write-Warning "Repair step failed for $localName"
        Remove-Item $localName
        continue
    }

    # ----- 2. Multiple optimizations -----
    for ($run = 1; $run -le $nOptimize; $run++) {
        Write-Host "  Optimization run $run / $nOptimize"
        & $foldxExe --command=Optimize --pdb="$repaired" *>> $foldxLog
        $optimized = $localName -replace '\.pdb$', '_Repair_Optimized.pdb'
        if (Test-Path $optimized) {
            $runFile = $localName -replace '\.pdb$', "_Run${run}_Optimized.pdb"
            Rename-Item $optimized $runFile
            Write-Host "    -> $runFile"
        }
    }
Get-ChildItem -Filter "*Optimized*.pdb" | ForEach-Object {
Write-Host "AnalyseComplex: $($_.Name)"
& .\FoldX.exe --command=AnalyseComplex --pdb="$($_.Name)" --analyseComplexChains A,B
}
    # Remove the temporarily copied input PDB (keep the outputs)
    Remove-Item $localName -ErrorAction SilentlyContinue
    # Optionally also remove repaired file if you don't need it later
    # Remove-Item $repaired -ErrorAction SilentlyContinue
}

Get-ChildItem -Filter "*Summary_Optimized*.fxout" | ForEach-Object {
>>     $content = Get-Content $_.FullName -Raw
>>     if ($content -match "Interaction Energy\s*:\s*(-?\d+\.\d+)") {
>>         [PSCustomObject]@{ File = $_.Name; BindingEnergy = [double]$matches[1] }
>>     }
>> } | Sort-Object BindingEnergy | Format-Table -AutoSize



Write-Host "All done. Check .fxout files for results."