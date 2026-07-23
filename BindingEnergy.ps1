# ExtractFoldX.ps1
$outputXlsx = "FoldX_BindingEnergies.xlsx"

# Collect results
$results = Get-ChildItem "Summary_Optimized*.fxout" | ForEach-Object {
    $lines = Get-Content $_.FullName
    $sum = 0.0
    $inTable = $false
    $cluster = ""
    # Extract cluster identifier from filename (e.g., cluster_47)
    if ($_.Name -match 'cluster_(\d+)') { $cluster = "cluster_$($matches[1])" }

    foreach ($line in $lines) {
        if ($line -match '^Pdb\s+Group1') {
            $inTable = $true
            continue
        }
        if ($inTable -and $line -match '\S') {
            $cols = $line -split '\s+'
            # Interaction Energy is the 6th column (index 5)
            if ($cols.Count -ge 6) {
                $sum += [double]$cols[5]
            }
        }
    }

    [PSCustomObject]@{
        Cluster          = $cluster
        File             = $_.Name
        BindingEnergy    = $sum
    }
}

# Sort by binding energy (most negative first)
$results = $results | Sort-Object BindingEnergy

# Export to Excel
$results | Export-Excel -Path $outputXlsx -WorksheetName "FoldX_Summary" -AutoSize -TableName "BindingEnergies"

Write-Host "Results saved to $outputXlsx"