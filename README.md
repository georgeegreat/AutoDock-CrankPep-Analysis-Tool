# AutoDock CrankPep Analysis Tool

A comprehensive Python script for visualization and analysis of protein-peptide interactions from AutoDock CrankPep docking results, with data preparation capabilities for downstream processing.

---

## Overview

This tool provides extensive contact frequency analysis, energy rescoring visualization, and multi-format data export for AutoDock CrankPep docking campaigns. It processes molecular dynamics trajectories or docking pose ensembles to identify and quantify protein-peptide interactions.

> **Note:** The term "frame" refers to a representative structure from a cluster of docked poses in the PDB output from CrankPep. These clusters are formed after AutoDock CrankPep's own clustering procedure.

---

## How the Script Works

### 1. Structure Loading
- Loads protein structure from PDBQT, PDB, or MOL2 format
- Loads ligand/peptide structures with multiple models/conformations
- Supports trajectory analysis and ensemble docking results

### 2. Atom Selection
- Uses MDAnalysis selection strings to identify relevant atoms
- Supports custom residue ranges (e.g., `"resid 31:40"`)
- Separates protein and peptide atoms for distance calculations
- Flexible selection for complex systems

### 3. Contact Analysis
- Iterates through all frames/models
- Calculates pairwise distances between protein and peptide atoms
- Identifies contacts below the specified cutoff distance
- Counts unique residue-residue contacts per frame
- Normalizes by total frames to get contact frequency (0.0-1.0)

### 4. Residue Identification
- Generates unique identifiers: `ChainID_ResidueNumber_ResidueType`
- Optionally sorts residues from N-terminus to C-terminus
- Handles chain separations and multiple segments
- Maintains residue ordering for sequential analysis

### 5. Matrix Generation
- Creates pivot tables with protein residues as rows, peptides as columns
- Values represent contact frequencies (0.0 to 1.0)
- Generates row and column-normalized versions for relative analysis
- Filters data based on user-specified threshold

### 6. Visualization
- Generates heatmaps showing contact patterns
- Creates bar plots for residue-level interaction profiles
- Produces histograms for energy distributions
- Supports multiple image formats (PNG, PDF, SVG, JPG)
- Customizable DPI and orientation settings

### 7. DLG File Analysis *(Optional)*
- Parses AutoDock CrankPep DLG (Docking Log) files
- Extracts ADCP clustering data and binding affinities
- Extracts OMM (OpenMM) rescoring results
- Maps OMM data to ADCP clusters using `adcp_rank` column
- Generates sorted energy visualizations
- Provides comprehensive energy statistics and distributions

### 8. Monomer Analysis *(Optional)*
- For multimeric proteins, aggregates contacts per monomer unit
- Sums contact frequencies across all monomer copies
- Generates separate monomer interaction heatmap
- Useful for symmetric proteins and protein complexes

---

## Output Files

### Contact Frequency Data (.xlsx)

| File | Purpose |
|------|---------|
| `contacts_full_data.xlsx` | Complete contact frequency data with all protein-peptide residue pairs |
| `contacts_matrix.xlsx` | Contact frequency matrix (protein residues × peptide residues) |
| `contacts_matrix_row_normalized.xlsx` | Row-normalized matrix showing relative interaction strength per protein residue |
| `contacts_matrix_column_normalized.xlsx` | Column-normalized matrix showing relative interaction strength per peptide residue |
| `contacts_filtered_matrix.xlsx` | Filtered matrix containing only contacts above threshold |

**Columns in full data:**
- `Protein_Residue`: Identifier (ChainID_ResidueNumber_ResidueType)
- `Peptide_Residue`: Identifier
- `Contact_Count`: Number of frames with contact
- `Frequency`: Contact frequency (0-1), normalized by total frames

### Barplot Data (.xlsx)

| File | Purpose |
|------|---------|
| `contacts_peptide_barplot_data.xlsx` | Summed contact frequencies per peptide residue |
| `contacts_protein_barplot_data.xlsx` | Summed contact frequencies per protein residue (N→C order) |

### DLG Analysis Files (.xlsx) *[Generated with `--dlg` option]*

| File | Purpose | Columns |
|------|---------|---------|
| `contacts_adcp_cluster_analysis.xlsx` | AutoDock CrankPep cluster data sorted by affinity | cluster_num, affinity, cluster_size |
| `contacts_omm_cluster_analysis.xlsx` | OpenMM energy rescoring results | cluster_num, interaction_energy, dE_interaction |
| `contacts_sorted_omm_interaction_energy.xlsx` | OMM interaction energy sorted best→worst | cluster, cluster_num, interaction_energy |
| `contacts_sorted_omm_dE_interaction.xlsx` | OMM dE_interaction sorted best→worst | cluster, cluster_num, dE_interaction |

### Visualization Files (Image Format)

#### General Contact Analysis
- `contacts_heatmap.png` — Full contact frequency heatmap (Yellow→Red: 0.0→1.0)
- `contacts_filtered_heatmap.png` — Filtered heatmap (threshold contacts only)
- `contacts_peptide_barplot.png` — Total contacts per peptide residue
- `contacts_protein_barplot.png` — Total contacts per protein residue (N→C)

#### Energy Analysis *(requires `--dlg` option)*
- `contacts_best_energies_histogram.png` — Distribution of raw bestEnergies (with KDE, mean, median)
- `contacts_cluster_analysis.png` — Three-panel cluster visualization:
  - Panel 1: ADCP affinity vs cluster size
  - Panel 2: OMM interaction energy by cluster
  - Panel 3: OMM dE_interaction by cluster
- `contacts_sorted_omm_interaction_energy.png` — Sorted OMM interaction energies
- `contacts_sorted_omm_dE_interaction.png` — Sorted OMM dE_interaction values

#### Multimer Analysis *(requires `--num-monomers > 1`)*
- `contacts_monomer_heatmap.png` — Summed interactions across monomer units

### Monomer Files (.xlsx) *[Generated with `--num-monomers > 1`]*

| File | Purpose |
|------|---------|
| `contacts_monomer_matrix.xlsx` | Contact frequency matrix summed across monomers |

---

## Data Format

All data files are saved in **XLSX (Excel)** format for maximum compatibility:

✓ Microsoft Excel  
✓ LibreOffice Calc  
✓ Python `pandas` (`pd.read_excel()`)  
✓ R `readxl` package  
✓ Other data analysis tools  

Each XLSX file contains multiple sheets for organized data presentation.

---

## Command-Line Reference

```
usage: crankpep_analysis.py [-h] [--version] -p PROTEIN -l LIGAND -c CUTOFF
                            [-o OUTPUT] [--dpi DPI] [--heatmap-format FORMAT]
                            [--prot-sel PROT_SEL] [--pep-sel PEPTIDE_SEL]
                            [--threshold THRESHOLD] [--num-monomers NUM_MONOMERS]
                            [--dlg DLG] [-v] [--silent]

AutoDock CrankPep Analysis Tool

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -p PROTEIN, --protein PROTEIN
                        Input protein structure file (PDBQT, PDB, MOL2)
  -l LIGAND, --ligand LIGAND
                        Input ligand/peptide structure file (PDB, MOL2)
  -c CUTOFF, --cutoff CUTOFF
                        Distance cutoff for contact analysis (Angstroms)
  -o OUTPUT, --output OUTPUT
                        Output directory for results
  --dpi DPI             DPI for output images (default: 300)
  --heatmap-format FORMAT
                        Image format for heatmaps (PNG, PDF, SVG, JPG)
  --prot-sel PROT_SEL  MDAnalysis selection string for protein atoms
  --pep-sel PEPTIDE_SEL
                        MDAnalysis selection string for peptide atoms
  --threshold THRESHOLD
                        Frequency threshold for filtered output
  --num-monomers NUM_MONOMERS
                        Number of monomers in the complex (for multimer analysis)
  --dlg DLG             AutoDock CrankPep DLG file for energy analysis
  -v, --verbose         Verbose output with additional details
  --silent              Suppress all output messages
```

---

## Usage Examples

### ⚙️ Basic Contact Analysis

```bash
python crankpep_analysis.py -p protein.pdb -l peptide.pdb -c 4.0 -o results
```

### ⚙️ With DLG File Analysis

```bash
python crankpep_analysis.py --dlg docking.dlg -p protein.pdb \
  -l ligand_poses.pdb -c 4.0 -o dlg_analysis
```

### ⚙️ With Custom Selections and Filtering

```bash
python crankpep_analysis.py -p protein.pdb -l peptide.pdb \
  --prot-sel "protein and resid 31:100" \
  --pep-sel "resname LIG" \
  --threshold 0.1 -o filtered_results
```

### ⚙️ With Monomer Analysis

```bash
python crankpep_analysis.py -p tetramer.pdb -l peptide.pdb \
  --num-monomers 4 -o monomer_analysis
```

### ⚙️ Verbose Output with Custom Format

```bash
python crankpep_analysis.py -p protein.pdb -l peptide.pdb -v \
  --heatmap-format pdf --dpi 300 -o high_quality
```
