# AutoDock-CrankPep-Analysis-Tool
### This directory contains the CrankPep Contact Analysis Tool, a python script for vizualization of protein-peptide interactions from AutoDock CrankPep docking results and data preparation for possible downstream processing
## How the script works
### 1. Structure loading
   - Loads protein structure from PDBQT, PDB, or MOL2 format
   - Loads ligand/peptide structures with multiple models/conformations

### 2. Atom selection
   - Uses MDAnalysis selection strings to identify relevant atoms
   - Supports custom residue ranges (e.g., "resid 31:40")
   - Separates protein and peptide atoms for distance calculations

### 3. Contact analysis
   - Iterates through all frames/models
   - Calculates pairwise distances between protein and peptide atoms
   - Identifies contacts below the specified cutoff distance
   - Counts unique residue-residue contacts per frame
   - Normalizes by total frames to get contact frequency

### 4. Residue identification
   - Generates unique identifiers: ChainID_ResidueNumber_ResidueType
   - Optionally sorts residues from N-terminus to C-terminus
   - Handles chain separations and multiple segments

### 5. Matrix generation
   - Creates pivot table with protein residues as rows, peptides as columns
   - Values represent contact frequencies (0.0 to 1.0)
   - Generates row and column-normalized versions
   - Filters data based on user-specified threshold

### 6. Vizualization
   - Generates heatmaps showing contact patterns
   - Creates bar plots for residue-level analysis
   - Produces histograms for energy distributions
   - Supports multiple image formats (PNG, PDF, SVG, JPG)

### 7. DLG file analysis (Optional)
   - Parses AutoDock CrankPep DLG (Docking Log) files
   - Extracts ADCP clustering data and affinities
   - Extracts OMM (OpenMM) rescoring results
   - Maps OMM data to ADCP clusters using adcp_rank column
   - Generates sorted energy visualizations
   - Provides comprehensive energy statistics

### 8. Monomer analysis (Optional)
   - For multimeric proteins, aggregates contacts per monomer unit
   - Sums contact frequencies across all monomer copies
   - Generates separate monomer interaction heatmap


## Output file descriptions
### Note: term "frame" means a cluster of docked poses in the .pdb output file form CrankPep. That file contains a structures of representative clusters (such clasters are formed after AutoDock CrankPep own clusterization procedure)
### Contact frequency data files (.xlsx format):
───────────────────────────────────────────────────────────────────────────────
1. contacts_full_data.xlsx
- Description: Complete contact frequency data with all protein-peptide residue pairs
- Sheets:
   - Contact_Data: Contains all contact pairs with counts and frequencies
- Columns:
-    Protein_Residue: Identifier of protein residue (ChainID_ResidueNumber_ResidueType)
-    Peptide_Residue: Identifier of peptide residue
-    Contact_Count: Number of frames where contact occurred
-    Frequency: Contact frequency (0-1), normalized by total frames

2. contacts_matrix.xlsx
   Description: Contact frequency matrix (protein residues × peptide residues)
   Sheets:
     - Contact_Matrix: Frequency matrix with residues as row/column labels
   Use: Direct visualization/heatmap generation

3. contacts_matrix_row_normalized.xlsx
   Description: Row-normalized contact frequency matrix
   Sheets:
     - Row_Normalized: Each row normalized by its sum
   Use: Shows relative interaction strength within each protein residue

4. contacts_matrix_column_normalized.xlsx
   Description: Column-normalized contact frequency matrix
   Sheets:
     - Column_Normalized: Each column normalized by its sum
   Use: Shows relative interaction strength within each peptide residue

5. contacts_filtered_matrix.xlsx
   Description: Filtered matrix containing only contacts above threshold
   Sheets:
     - Filtered_Matrix: Only residue pairs with frequency >= threshold value
   Note: Generated only if --threshold is specified

### Barplot data files (.xlsx format):
───────────────────────────────────────────────────────────────────────────────

6. contacts_peptide_barplot_data.xlsx
   Description: Summed contact frequencies for each peptide residue
   Sheets:
     - Peptide_Barplot: Aggregate contact data per peptide residue
   Columns:
     - residue: Peptide residue identifier
     - frequency: Total contact frequency for that residue

7. contacts_protein_barplot_data.xlsx
   Description: Summed contact frequencies for each protein residue
   Sheets:
     - Protein_Barplot: Aggregate contact data per protein residue
   Columns:
     - residue: Protein residue identifier (N to C terminus order)
     - frequency: Total contact frequency for that residue

### DLG analysis files (.xlsx format) - !Generated *only** with --dlg option:
───────────────────────────────────────────────────────────────────────────────

8. contacts_adcp_cluster_analysis.xlsx
   Description: AutoDock CrankPep cluster analysis
   Sheets:
     - ADCP_Clusters: ADCP cluster information sorted by affinity
   Columns:
     - cluster_num: Cluster number (1-100)
     - affinity: Binding affinity (kcal/mol)
     - cluster_size: Number of poses in cluster

9. contacts_omm_cluster_analysis.xlsx
   Description: OpenMM energy rescoring results
   Sheets:
     - OMM_Clusters: OMM rescored cluster data
   Columns:
     - cluster_num: ADCP cluster number
     - interaction_energy: E_Complex-E_Receptor
     - dE_interaction: E_Complex-E_rec-E_pep

10. contacts_sorted_omm_interaction_energy.xlsx
    Description: OMM interaction energy sorted from best to worst
    Sheets:
      - Sorted_Energy: Clusters sorted by interaction energy
    Columns:
      - cluster: Cluster label (C1, C2, etc.)
      - cluster_num: Original cluster number
      - interaction_energy: E_Complex-E_Receptor value

11. contacts_sorted_omm_dE_interaction.xlsx
    Description: OMM dE_interaction sorted from best to worst
    Sheets:
      - Sorted_dE: Clusters sorted by dE_interaction
    Columns:
      - cluster: Cluster label (C1, C2, etc.)
      - cluster_num: Original cluster number
      - dE_interaction: E_Complex-E_rec-E_pep value

### Vizualization files (Image format):
───────────────────────────────────────────────────────────────────────────────

12. contacts_heatmap.png
    Description: Full contact frequency heatmap
    Content: Visual representation of all protein-peptide contacts
    Color: Yellow (0.0 frequency) → Red (1.0 frequency)

13. contacts_filtered_heatmap.png
    Description: Filtered contact heatmap (contacts above threshold only)
    Content: Shows only significant contacts

14. contacts_peptide_barplot.png
    Description: Bar plot of peptide residue interaction strength
    Content: Total contacts per peptide residue

15. contacts_protein_barplot.png
    Description: Bar plot of protein residue interaction strength
    Content: Total contacts per protein residue (N to C terminus order)

16. contacts_best_energies_histogram.png
    Description: Distribution of raw bestEnergies from docking
    Content: Histogram with KDE, mean, and median lines
    Note: Generated only with --dlg option

17. contacts_cluster_analysis.png
    Description: Three-panel cluster analysis visualization
    Content:
      - Panel 1: ADCP affinity vs cluster size
      - Panel 2: OMM interaction energy by cluster
      - Panel 3: OMM dE_interaction by cluster
    Note: Generated only with --dlg option

18. contacts_sorted_omm_interaction_energy.png
    Description: OMM interaction energy sorted visualization
    Content: Clusters sorted from best to worst energy
    Note: Generated only with --dlg option

19. contacts_sorted_omm_dE_interaction.png
    Description: OMM dE_interaction sorted visualization
    Content: Clusters sorted from best to worst dE_interaction
    Note: Generated only with --dlg option

20. contacts_monomer_heatmap.png
    Description: Monomer interaction heatmap
    Content: Summed interactions across all monomers
    Note: Generated only if --num-monomers > 1

### Monomer files (.xlsx format) - !Generated *only** with --num-monomers > 1:
───────────────────────────────────────────────────────────────────────────────

21. contacts_monomer_matrix.xlsx
    Description: Contact frequency matrix summed across monomers
    Sheets:
      - Monomer_Matrix: Aggregated contact data per monomer unit


## Data format


All data files are saved in XLSX (Excel) format for better compatibility with:
  - Microsoft Excel
  - LibreOffice Calc
  - Python pandas (direct import)
  - R (readxl package)
  - Other data analysis tools

Each XLSX file may contain multiple sheets for organized data presentation.


## Usage examples


### Basic contact analysis:
  python crankpep_analysis.py -p protein.pdb -l peptide.pdb -c 4.0 -o results

With DLG file analysis:
  python crankpep_analysis.py --dlg docking.dlg -p protein.pdb \
    -l ligand_poses.pdb -c 4.0 -o dlg_analysis

With custom selections and filtering:
  python crankpep_analysis.py -p protein.pdb -l peptide.pdb \
    --prot-sel "protein and resid 31:100" \
    --pep-sel "resname LIG" \
    --threshold 0.1 -o filtered_results

With monomer analysis:
  python crankpep_analysis.py -p tetramer.pdb -l peptide.pdb \
    --num-monomers 4 -o monomer_analysis

Verbose output with custom format:
  python crankpep_analysis.py -p protein.pdb -l peptide.pdb -v \
    --heatmap-format pdf --dpi 300 -o high_quality



