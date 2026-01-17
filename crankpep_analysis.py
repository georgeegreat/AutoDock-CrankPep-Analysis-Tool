#!/usr/bin/env python3
"""
This script provides extensive analysis and vizualization for the data
produced by AutoDock CrankPep.
Contact frequency analyzer for protein-peptide interactions.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import MDAnalysis as mda
from MDAnalysis.analysis import distances
import numpy as np
import pandas as pd
from collections import Counter
import argparse
import sys
import os
from pathlib import Path
import re
from scipy.stats import gaussian_kde
from datetime import datetime

# regex patterns for reuse
_RESIDUE_ID_PATTERN = re.compile(r'^(\S+)_(\d+)_(\S+)$')
_ENERGY_PATTERN = re.compile(r'bestEnergies\s*\[(.*?)\]')


class ArgumentParser:
    """Handles command-line argument parsing."""
    
    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(
            description="Calculate residue-residue contact frequencies between protein and peptide.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python crankpep_analysis.py -p protein.pdbqt -l peptide_poses.pdb
  python crankpep_analysis.py -p protein.pdbqt -l peptide.pdb -c 5.0 -o my_contacts
  python crankpep_analysis.py --dlg docking.dlg -p protein.pdb -l ligand_poses.pdb
  python crankpep_analysis.py -p protein.pdb -l peptide.pdb --num-monomers 4
            """
)
        
        parser.add_argument("-p", "--protein", required=True,
                           help="Input protein structure file (PDB, PDBQT, MOL2)")
        parser.add_argument("-l", "--ligand",
                           help="Input ligand/peptide file (PDB, PDBQT, MOL2 with multiple models)")
        parser.add_argument("--dlg", help="Input AutoDock DLG file")
        parser.add_argument("-c", "--cutoff", type=float, default=4.0,
                           help="Distance cutoff in Angstroms (default: 4.0)")
        parser.add_argument("-o", "--output", default="contacts",
                           help="Base name for output files (default: 'contacts')")
        parser.add_argument("--prot-sel", default="protein",
                           help="Selection for protein atoms (MDAnalysis string or range like '31:40')")
        parser.add_argument("--pep-sel", default="protein",
                           help="Selection for peptide atoms (MDAnalysis string or range)")
        parser.add_argument("--heatmap-format", choices=["png", "pdf", "svg", "jpg"], default="png",
                           help="Format for heatmap (default: png)")
        parser.add_argument("--dpi", type=int, default=600, help="DPI for images (default: 600)")
        parser.add_argument("--no-plot", action="store_true", help="Skip heatmap generation")
        parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
        parser.add_argument("--horizontal", action="store_true", help="Horizontal heatmap")
        parser.add_argument("--threshold", type=float, default=0.05,
                           help="Threshold for filtered heatmap (default: 0.05)")
        parser.add_argument("--max-models", type=int, default=None,
                           help="Maximum number of models to analyze")
        parser.add_argument("--num-monomers", type=int, default=1,
                           help="Number of monomers in protein (default: 1)")
        parser.add_argument("--sort-residues", action="store_true", default=True,
                           help="Sort residues N to C terminus (default: True)")
        
        return parser.parse_args()


class FileValidator:
    """Validates input files and manages file operations."""
    
    @staticmethod
    def validate_files(protein_file, ligand_file=None, dlg_file=None):
        """Check if input files exist."""
        files = {'Protein': protein_file, 'Ligand': ligand_file, 'DLG': dlg_file}
        for name, filepath in files.items():
            if filepath and not os.path.exists(filepath):
                print(f"ERROR: {name} file '{filepath}' not found!", file=sys.stderr)
                sys.exit(1)
    
    @staticmethod
    def create_output_directory(protein_file, ligand_file, dlg_file=None):
        """Create and return output directory path."""
        protein_stem = Path(protein_file).stem
        ligand_stem = Path(ligand_file).stem
        
        if dlg_file:
            dlg_stem = Path(dlg_file).stem
            output_dir = f"analysis_{protein_stem}_and_{ligand_stem}_with_{dlg_stem}"
        else:
            output_dir = f"analysis_{protein_stem}_and_{ligand_stem}"
        
        os.makedirs(output_dir, exist_ok=True)
        return output_dir


class ResidueHandler:
    """Handles residue identification and manipulation."""
    
    @staticmethod
    def get_residue_identifier(atom):
        """Generate unique residue identifier."""
        segid = atom.segid if atom.segid else "X"
        return f"{segid}_{atom.resid}_{atom.resname}"
    
    @staticmethod
    def parse_residue_range(selection_string):
        """Parse residue range (e.g., '31:40') or MDAnalysis selection."""
        if ":" in selection_string and not any(char.isalpha() for char in selection_string.replace(":", "")):
            try:
                start, end = map(int, selection_string.split(":"))
                return f"resid {start}:{end}"
            except ValueError:
                return selection_string
        
        if "resid" in selection_string.lower() and ":" in selection_string:
            return selection_string
        
        return selection_string
    
    @staticmethod
    def sort_residues_by_position(residue_list, universe):
        """Sort residues from N to C terminus."""
        residue_info = []
        for res_id in residue_list:
            match = _RESIDUE_ID_PATTERN.match(res_id)
            if match:
                chain, resnum_str, resname = match.groups()
                try:
                    resnum = int(resnum_str)
                except ValueError:
                    resnum = 0
                residue_info.append((chain, resnum, resname, res_id))
        
        sorted_info = sorted(residue_info, key=lambda x: (x[0], x[1]))
        return [item[3] for item in sorted_info]


class DLGParser:
    """Parses AutoDock CrankPep DLG files."""
    
    def __init__(self, dlg_file, verbose=False):
        self.dlg_file = dlg_file
        self.verbose = verbose
        self.lines = self._load_file()
    
    def _load_file(self):
        """Load DLG file content."""
        if not os.path.exists(self.dlg_file):
            print(f"ERROR: DLG file '{self.dlg_file}' not found!", file=sys.stderr)
            sys.exit(1)
        
        try:
            with open(self.dlg_file, 'r') as f:
                return f.readlines()
        except Exception as e:
            print(f"ERROR: Failed to read DLG file: {e}", file=sys.stderr)
            sys.exit(1)
    
    def parse(self, max_models=None):
        """Parse and extract all DLG data."""
        if self.verbose:
            print(f"Parsing DLG file: {self.dlg_file}")
        
        results = {
            'best_energies_raw': self._extract_best_energies(),
            'contact_cutoff': self._extract_contact_cutoff(),
            'total_poses': self._extract_total_poses(),
            'adcp_clusters': [],
            'omm_clusters': []
        }
        
        adcp_clusters = self._parse_adcp_clusters()
        omm_clusters = self._parse_omm_clusters()
        
        results['adcp_clusters'] = sorted(adcp_clusters, key=lambda x: x['cluster_num'])
        results['omm_clusters'] = sorted(omm_clusters, key=lambda x: x['model_num'])
        
        # Create ADCP clusters from OMM if needed
        if not adcp_clusters and omm_clusters:
            if self.verbose:
                print("Creating ADCP clusters from OMM table data...")
            results['adcp_clusters'] = [
                {
                    'cluster_num': omm['model_num'],
                    'affinity': omm['affinity'],
                    'cluster_size': omm['cluster_size'],
                    'energy_type': 'ADCP'
                }
                for omm in omm_clusters
            ]
        
        return results
    
    def _extract_best_energies(self):
        """Extract raw bestEnergies list."""
        for line in self.lines:
            match = _ENERGY_PATTERN.search(line)
            if match:
                try:
                    energies = [float(x.strip()) for x in match.group(1).split(',')]
                    if self.verbose:
                        print(f"Found {len(energies)} bestEnergies")
                    return energies
                except ValueError as e:
                    print(f"WARNING: Could not parse bestEnergies: {e}", file=sys.stderr)
        return []
    
    def _extract_contact_cutoff(self):
        """Extract contact cutoff."""
        for line in self.lines:
            if 'Clustering MC trajectories based in contacts using cutoff:' in line:
                try:
                    return float(line.split(':')[-1].strip())
                except ValueError:
                    pass
        return 0.8
    
    def _extract_total_poses(self):
        """Extract total poses."""
        for line in self.lines:
            if 'finished calculating neighbors for' in line and 'poses' in line:
                for part in line.split():
                    if part.isdigit():
                        return int(part)
        return 0
    
    def _parse_adcp_clusters(self):
        """Extract ADCP clustering table."""
        adcp_clusters = []
        in_adcp_table = False
        
        for line in self.lines:
            if '|  affinity  |' in line or '-----+------------' in line:
                in_adcp_table = True
                continue
            
            if in_adcp_table:
                if not line.strip() or '-----' in line or 'OMM Ranking:' in line:
                    in_adcp_table = False
                    continue
                
                parts = line.split()
                if parts and parts[0].isdigit():
                    try:
                        if len(parts) >= 4:
                            adcp_clusters.append({
                                'cluster_num': int(parts[0]),
                                'affinity': float(parts[1]),
                                'cluster_size': int(parts[3]),
                                'energy_type': 'ADCP'
                            })
                    except (ValueError, IndexError):
                        if self.verbose:
                            print(f"Warning parsing ADCP line: {line}")
        
        return adcp_clusters
    
    def _parse_omm_clusters(self):
        """Extract OMM ranking table with correct ADCP rank mapping."""
        omm_clusters = []
        in_omm_table = False
        omm_data_started = False
        
        for line in self.lines:
            # Look for OMM ranking table header
            if 'OMM Ranking:' in line and 'Model' in line and 'Rank' in line:
                in_omm_table = True
                continue
            
            # Skip to data section after separator
            if in_omm_table and '-------+------+------' in line and not omm_data_started:
                omm_data_started = True
                continue
            
            # Exit table at end separator
            if in_omm_table and omm_data_started and '-------+------+------' in line:
                in_omm_table = False
                continue
            
            # Process data lines
            if in_omm_table and omm_data_started:
                if not line.strip() or 'OMM Ranking:' not in line:
                    continue
                
                # Remove the "OMM Ranking:" prefix
                line_clean = line.replace('OMM Ranking:', '').strip()
                parts = line_clean.split()
                
                # Parse the columns: Model# | Rank(OpenMM) | Rank(ADCP) | E_Complex-E_Receptor | E_Complex-E_rec-E_pep | affinity | ref_fnc | cluster_size | rmsd_stdv | energy_stdv | best_run
                if parts and parts[0].isdigit():
                    try:
                        if len(parts) >= 11:
                            model_num = int(parts[0])
                            openmm_rank = int(parts[1])
                            adcp_rank = int(parts[2])  # This is the KEY: ADCP cluster number
                            interaction_energy = float(parts[3])
                            dE_interaction = float(parts[4])
                            affinity = float(parts[5])
                            ref_fnc = float(parts[6])
                            cluster_size = int(parts[7])
                            best_run = int(parts[10])
                            
                            omm_clusters.append({
                                'model_num': model_num,
                                'openmm_rank': openmm_rank,
                                'adcp_rank': adcp_rank,  # CRITICAL: This maps to ADCP cluster_num
                                'interaction_energy': interaction_energy,
                                'dE_interaction': dE_interaction,
                                'affinity': affinity,
                                'ref_fnc': ref_fnc,
                                'cluster_size': cluster_size,
                                'best_run': best_run,
                                'energy_type': 'OMM'
                            })
                    except (ValueError, IndexError) as e:
                        if self.verbose:
                            print(f"Warning parsing OMM line: {line_clean} - Error: {e}")
        
        return omm_clusters


class ContactAnalyzer:
    """Analyzes protein-ligand contacts."""
    
    def __init__(self, protein_file, ligand_file, cutoff, prot_sel, pep_sel,
                 verbose=False, max_models=None, sort_residues=True):
        self.protein_file = protein_file
        self.ligand_file = ligand_file
        self.cutoff = cutoff
        self.prot_sel = prot_sel
        self.pep_sel = pep_sel
        self.verbose = verbose
        self.max_models = max_models
        self.sort_residues = sort_residues
        self.u_prot = None
        self.u_pep = None
        self.prot_atoms = None
        self.pep_atoms = None
    
    def load_structures(self):
        """Load protein and ligand structures."""
        try:
            self.u_prot = mda.Universe(self.protein_file)
            self.u_pep = mda.Universe(self.ligand_file)
            if self.verbose:
                print(f"Loaded protein: {self.protein_file}")
                print(f"Loaded ligand: {self.ligand_file}")
        except Exception as e:
            print(f"ERROR: Failed to load structures: {e}", file=sys.stderr)
            sys.exit(1)
    
    def select_atoms(self):
        """Select atoms based on criteria."""
        try:
            self.prot_atoms = self.u_prot.select_atoms(self.prot_sel)
            self.pep_atoms = self.u_pep.select_atoms(self.pep_sel)
            
            if self.verbose:
                print(f"Protein: {len(self.prot_atoms)} atoms, {len(self.prot_atoms.residues)} residues")
                print(f"Peptide: {len(self.pep_atoms)} atoms, {len(self.pep_atoms.residues)} residues")
        except Exception as e:
            print(f"ERROR: Invalid atom selection: {e}", file=sys.stderr)
            sys.exit(1)
    
    def analyze(self):
        """Perform contact analysis."""
        self.load_structures()
        self.select_atoms()
        
        total_frames = len(self.u_pep.trajectory)
        if self.max_models and self.max_models < total_frames:
            total_frames = self.max_models
        
        if self.verbose:
            print(f"Cutoff: {self.cutoff} Å, Frames: {total_frames}")
        
        # Get residue identifiers
        prot_res_ids = [ResidueHandler.get_residue_identifier(atom) for atom in self.prot_atoms]
        pep_res_ids = [ResidueHandler.get_residue_identifier(atom) for atom in self.pep_atoms]
        
        prot_res_set = list(dict.fromkeys(prot_res_ids))
        pep_res_set = list(dict.fromkeys(pep_res_ids))
        
        # Optional sorting
        if self.sort_residues:
            try:
                prot_res_set = ResidueHandler.sort_residues_by_position(prot_res_set, self.u_prot)
                pep_res_set = ResidueHandler.sort_residues_by_position(pep_res_set, self.u_pep)
            except Exception as e:
                if self.verbose:
                    print(f"WARNING: Could not sort residues: {e}")
        
        # Count contacts
        contact_counter = Counter()
        protein_residue_counts = Counter()
        peptide_residue_counts = Counter()
        
        for frame_num, ts in enumerate(self.u_pep.trajectory, 1):
            if frame_num > total_frames:
                break
            
            if self.verbose and frame_num % 10 == 0:
                print(f"  Frame {frame_num}/{total_frames}")
            
            pairs = distances.capped_distance(
                self.prot_atoms.positions,
                self.pep_atoms.positions,
                max_cutoff=self.cutoff,
                return_distances=False)
            
            frame_contacts = set()
            protein_res_in_frame = set()
            peptide_res_in_frame = set()
            
            for i, j in pairs:
                prot_res = prot_res_ids[i]
                pep_res = pep_res_ids[j]
                frame_contacts.add((prot_res, pep_res))
                protein_res_in_frame.add(prot_res)
                peptide_res_in_frame.add(pep_res)
            
            for pair in frame_contacts:
                contact_counter[pair] += 1
            
            for res in protein_res_in_frame:
                protein_residue_counts[res] += 1
            for res in peptide_res_in_frame:
                peptide_residue_counts[res] += 1
        
        protein_res_freq = {res: count/total_frames for res, count in protein_residue_counts.items()}
        peptide_res_freq = {res: count/total_frames for res, count in peptide_residue_counts.items()}
        
        return contact_counter, self.prot_atoms, self.pep_atoms, total_frames, prot_res_set, pep_res_set, protein_res_freq, peptide_res_freq


class ContactMatrixBuilder:
    """Builds and saves contact frequency matrices."""
    
    @staticmethod
    def create_matrix(contact_counter, prot_res_set, pep_res_set, output_prefix, total_frames, max_residues=100):
        """Create and save contact matrices."""
        rows = []
        for (prot_res, pep_res), count in contact_counter.items():
            frequency = count / total_frames
            rows.append([prot_res, pep_res, count, frequency])
        
        df = pd.DataFrame(rows, columns=["Protein_Residue", "Peptide_Residue", "Contact_Count", "Frequency"])
        df_sorted = df.sort_values("Frequency", ascending=False)
        
        # Save full data in xlsx format
        xlsx_file = f"{output_prefix}_full_data.xlsx"
        df_sorted.to_excel(xlsx_file, sheet_name="Contact_Data", index=False, engine='openpyxl')
        
        # Limit residues for heatmap
        if len(prot_res_set) > max_residues:
            top_prot = df['Protein_Residue'].value_counts().head(max_residues).index.tolist()
            prot_res_set = [res for res in prot_res_set if res in top_prot]
        
        if len(pep_res_set) > max_residues:
            top_pep = df['Peptide_Residue'].value_counts().head(max_residues).index.tolist()
            pep_res_set = [res for res in pep_res_set if res in top_pep]
        
        # Create pivot matrix
        matrix = pd.DataFrame(0.0, index=prot_res_set, columns=pep_res_set)
        for _, row in df.iterrows():
            prot_res = row['Protein_Residue']
            pep_res = row['Peptide_Residue']
            if prot_res in matrix.index and pep_res in matrix.columns:
                matrix.loc[prot_res, pep_res] = row['Frequency']
        
        # Save matrices in xlsx format
        matrix_file = f"{output_prefix}_matrix.xlsx"
        matrix.to_excel(matrix_file, sheet_name="Contact_Matrix", engine='openpyxl')
        
        row_norm = matrix.div(matrix.sum(axis=1), axis=0).fillna(0)
        row_norm_file = f"{output_prefix}_matrix_row_normalized.xlsx"
        row_norm.to_excel(row_norm_file, sheet_name="Row_Normalized", engine='openpyxl')
        
        col_norm = matrix.div(matrix.sum(axis=0), axis=1).fillna(0)
        col_norm_file = f"{output_prefix}_matrix_column_normalized.xlsx"
        col_norm.to_excel(col_norm_file, sheet_name="Column_Normalized", engine='openpyxl')
        
        return matrix, df_sorted, row_norm, col_norm


class HeatmapPlotter:
    """Generates heatmap visualizations."""
    
    @staticmethod
    def _prepare_ticks(data_size, max_labels=50):
        """Generate tick spacing."""
        if data_size <= max_labels:
            return None
        return max(1, data_size // (max_labels // 2))
    
    @staticmethod
    def _sparse_labels(labels, interval):
        """Create sparse labels."""
        if interval is None:
            return labels
        return [label if i % interval == 0 else '' for i, label in enumerate(labels)]
    
    @staticmethod
    def plot(matrix, output_prefix, image_format, dpi, cutoff,
            horizontal=False, threshold=None, fig_width=12, fig_height=8):
        """Create heatmap plots."""
        filtered_matrix = None
        if threshold is not None:
            filtered_matrix = matrix[matrix >= threshold].fillna(0)
            filtered_matrix = filtered_matrix.loc[(filtered_matrix > 0).any(axis=1)]
            filtered_matrix = filtered_matrix.loc[:, (filtered_matrix > 0).any(axis=0)]
        
        heatmap_file = HeatmapPlotter._create_main(
            matrix, output_prefix, image_format, dpi, cutoff, horizontal, fig_width, fig_height
        )
        
        filtered_file = None
        if filtered_matrix is not None and not filtered_matrix.empty:
            filtered_file = HeatmapPlotter._create_filtered(
                filtered_matrix, output_prefix, image_format, dpi, threshold, horizontal, fig_width
            )
            filtered_matrix.to_csv(f"{output_prefix}_filtered_matrix.csv")
        
        return heatmap_file, filtered_file, filtered_matrix
    
    @staticmethod
    def _create_main(matrix, output_prefix, image_format, dpi, cutoff, horizontal, fig_width, fig_height):
        """Create a single heatmap plot."""
        try:
            plt.figure(figsize=(fig_width, fig_height))
            
            plot_matrix = matrix.T if horizontal else matrix
            xlabel = "Protein Residue" if horizontal else "Peptide Residue"
            ylabel = "Peptide Residue" if horizontal else "Protein Residue"
            
            tick_interval = HeatmapPlotter._prepare_ticks(max(plot_matrix.shape))
            xtick_labels = HeatmapPlotter._sparse_labels(plot_matrix.columns, tick_interval)
            ytick_labels = HeatmapPlotter._sparse_labels(plot_matrix.index, tick_interval)
            
            sns.heatmap(plot_matrix, cmap="YlOrRd", linewidths=0.2, linecolor="gray",
                       cbar_kws={"label": "Contact Frequency (0-1)"}, square=False,
                       xticklabels=xtick_labels, yticklabels=ytick_labels)
            
            plt.title(f"Protein-Peptide Contact Frequency (≤ {cutoff} Å)")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            output_file = f"{output_prefix}_heatmap.{image_format}"
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
            plt.close()
            return output_file
            
        except Exception as e:
            print(f"WARNING: Could not create heatmap: {e}", file=sys.stderr)
            return None
    
    @staticmethod
    def _create_filtered(filtered_matrix, output_prefix, image_format, dpi, threshold, horizontal, fig_width):
        """Create filtered heatmap with all protein residue labels."""
        try:
            num_residues = filtered_matrix.shape[0]
            adjusted_height = max(15, num_residues * 0.2)
            plt.figure(figsize=(fig_width, adjusted_height))
            
            plot_matrix = filtered_matrix.T if horizontal else filtered_matrix
            xlabel = "Protein Residue" if horizontal else "Peptide Residue"
            ylabel = "Peptide Residue" if horizontal else "Protein Residue"
            
            ytick_labels = plot_matrix.index.tolist()
            xtick_interval = HeatmapPlotter._prepare_ticks(len(plot_matrix.columns), 50)
            xtick_labels = HeatmapPlotter._sparse_labels(plot_matrix.columns, xtick_interval)
            
            sns.heatmap(plot_matrix, cmap="YlOrRd", linewidths=0.2, linecolor="gray",
                       cbar_kws={"label": f"Contact Frequency (>={threshold})"}, square=False,
                       xticklabels=xtick_labels, yticklabels=ytick_labels)
            
            plt.title(f"Filtered Contacts (freq ≥ {threshold})")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            output_file = f"{output_prefix}_filtered_heatmap.{image_format}"
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
            plt.close()
            return output_file
            
        except Exception as e:
            print(f"WARNING: Could not create filtered heatmap: {e}", file=sys.stderr)
            return None


class BarplotPlotter:
    """Generates barplot visualizations."""
    
    @staticmethod
    def plot_peptide(filtered_matrix, output_prefix, image_format, dpi=600, horizontal=False, total_frames=1):
        """Create bar plot for peptide residues."""
        try:
            plt.figure(figsize=(10, 6))
            
            if filtered_matrix.empty:
                plt.text(0.5, 0.5, "No contacts above threshold", ha='center', va='center',
                        transform=plt.gca().transAxes)
            else:
                if horizontal:
                    residues = filtered_matrix.index.tolist()
                    frequencies = filtered_matrix.sum(axis=1).values
                else:
                    residues = filtered_matrix.columns.tolist()
                    frequencies = filtered_matrix.sum(axis=0).values
                
                plt.bar(range(len(residues)), frequencies, color='steelblue', edgecolor='black', alpha=0.7)
                plt.ylabel('Average contacts per frame')
                plt.xticks(range(len(residues)), residues, rotation=45, ha='right')
                
                barplot_data = pd.DataFrame({'residue': residues, 'frequency': frequencies})
                barplot_xlsx_file = f"{output_prefix}_peptide_barplot_data.xlsx"
                barplot_data.to_excel(barplot_xlsx_file, sheet_name="Peptide_Barplot", index=False, engine='openpyxl')
            
            plt.xlabel('Peptide Residue')
            plt.title(f'Peptide Residue Interaction Profile (Total: {total_frames} frames)')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            barplot_file = f"{output_prefix}_peptide_barplot.{image_format}"
            plt.savefig(barplot_file, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            return barplot_file, f"{output_prefix}_peptide_barplot_data.xlsx"
        except Exception as e:
            print(f"WARNING: Could not create peptide barplot: {e}", file=sys.stderr)
            return None, None
    
    @staticmethod
    def plot_protein(filtered_matrix, output_prefix, image_format, dpi=600, horizontal=False, total_frames=1):
        """Create bar plot for protein residues."""
        try:
            plt.figure(figsize=(20, 12))
            
            if filtered_matrix.empty:
                plt.text(0.5, 0.5, "No contacts above threshold", ha='center', va='center',
                        transform=plt.gca().transAxes)
            else:
                if horizontal:
                    residues = filtered_matrix.columns.tolist()
                    frequencies = filtered_matrix.sum(axis=0).values
                else:
                    residues = filtered_matrix.index.tolist()
                    frequencies = filtered_matrix.sum(axis=1).values
                
                plt.bar(range(len(residues)), frequencies, color='darkorange', edgecolor='black', alpha=0.7)
                plt.ylabel('Average contacts per frame')
                plt.xticks(range(len(residues)), residues, rotation=45, ha='right')
                
                barplot_data = pd.DataFrame({'residue': residues, 'frequency': frequencies})
                barplot_xlsx_file = f"{output_prefix}_protein_barplot_data.xlsx"
                barplot_data.to_excel(barplot_xlsx_file, sheet_name="Protein_Barplot", index=False, engine='openpyxl')
            
            plt.xlabel('Protein Residue (N to C terminus)')
            plt.title(f'Protein Residue Interaction Profile (Total: {total_frames} frames)')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            barplot_file = f"{output_prefix}_protein_barplot.{image_format}"
            plt.savefig(barplot_file, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            return barplot_file, f"{output_prefix}_protein_barplot_data.xlsx"
        except Exception as e:
            print(f"WARNING: Could not create protein barplot: {e}", file=sys.stderr)
            return None, None


class MonomericInteractionPlotter:
    """Generates monomer interaction heatmaps."""
    
    @staticmethod
    def create_heatmap(matrix, num_monomers, output_prefix, image_format, dpi=600, fig_width=12, fig_height=10, verbose=False):
        """Create heatmap for peptide interactions with protein monomers."""
        if num_monomers <= 1:
            if verbose:
                print(f"Skipping monomer heatmap (monomers: {num_monomers})")
            return None
        
        try:
            if verbose:
                print(f"Creating monomer heatmap for {num_monomers} monomers...")
            
            protein_residues = matrix.index.tolist()
            total_residues = len(protein_residues)
            residues_per_monomer = total_residues // num_monomers
            
            monomer_residues = protein_residues[:residues_per_monomer]
            peptide_residues = matrix.columns.tolist()
            
            monomer_matrix = pd.DataFrame(0.0, index=monomer_residues, columns=peptide_residues)
            
            for i, residue in enumerate(monomer_residues):
                for monomer_idx in range(num_monomers):
                    full_index = i + (monomer_idx * residues_per_monomer)
                    if full_index < total_residues:
                        full_residue = protein_residues[full_index]
                        for peptide_res in peptide_residues:
                            if peptide_res in matrix.columns and full_residue in matrix.index:
                                monomer_matrix.loc[residue, peptide_res] += matrix.loc[full_residue, peptide_res]
            
            plt.figure(figsize=(fig_width, fig_height))
            
            if monomer_matrix.shape[0] > 50 or monomer_matrix.shape[1] > 50:
                xtick_interval = max(1, monomer_matrix.shape[1] // 20)
                ytick_interval = max(1, monomer_matrix.shape[0] // 20)
                xtick_labels = [label if i % xtick_interval == 0 else '' for i, label in enumerate(monomer_matrix.columns)]
                ytick_labels = [label if i % ytick_interval == 0 else '' for i, label in enumerate(monomer_matrix.index)]
            else:
                xtick_labels = monomer_matrix.columns
                ytick_labels = monomer_matrix.index
            
            sns.heatmap(monomer_matrix, cmap="YlOrRd", linewidths=0.2, linecolor="gray",
                       cbar_kws={"label": "Total Contact Frequency (sum across monomers)"},
                       square=False, xticklabels=xtick_labels, yticklabels=ytick_labels, vmax=0.5)
            
            plt.title(f"Peptide Interaction with Protein Monomer (sum across {num_monomers} monomers)")
            plt.xlabel('Peptide Residue')
            plt.ylabel('Monomer Residue')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            heatmap_file = f"{output_prefix}_monomer_heatmap.{image_format}"
            plt.savefig(heatmap_file, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            monomer_matrix.to_csv(f"{output_prefix}_monomer_matrix.csv")
            
            if verbose:
                print(f"Saved monomer heatmap to: {heatmap_file}")
            
            return heatmap_file
            
        except Exception as e:
            print(f"WARNING: Could not create monomer heatmap: {e}", file=sys.stderr)
            return None


class DLGPlotter:
    """Generates DLG analysis plots."""
    
    @staticmethod
    def plot_histogram(dlg_data, output_prefix, dpi=600, verbose=False):
        """Create histogram for bestEnergies with density estimation."""
        try:
            best_energies = dlg_data['best_energies_raw']
            if not best_energies:
                return None
            
            plt.figure(figsize=(10, 6))
            plt.hist(best_energies, bins=30, edgecolor='black', alpha=0.7, color='skyblue', density=True)
            
            if len(best_energies) > 1:
                kde = gaussian_kde(best_energies)
                x_range = np.linspace(min(best_energies), max(best_energies), 1000)
                plt.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            
            mean_val = np.mean(best_energies)
            median_val = np.median(best_energies)
            std_val = np.std(best_energies)
            
            plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
            
            stats_text = f'N = {len(best_energies)}\nMean = {mean_val:.2f}\nMedian = {median_val:.2f}\nStd = {std_val:.2f}'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.xlabel('Raw bestEnergies')
            plt.ylabel('Density')
            plt.title(f'Histogram of Raw bestEnergies (N={len(best_energies)})')
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            plt.tight_layout()
            
            histogram_file = f"{output_prefix}_best_energies_histogram.png"
            plt.savefig(histogram_file, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            if verbose:
                print(f"Saved histogram to: {histogram_file}")
            
            return histogram_file
            
        except Exception as e:
            print(f"WARNING: Could not create histogram: {e}", file=sys.stderr)
            return None
    
    @staticmethod
    def plot_cluster_analysis(dlg_data, output_prefix, dpi=600, verbose=False):
        """Create enhanced cluster analysis plots."""
        try:
            adcp_clusters = dlg_data['adcp_clusters']
            omm_clusters = dlg_data['omm_clusters']
            
            if not adcp_clusters:
                return None, None, None
            
            # Debug: Print what we have
            if verbose:
                print(f"\n=== OMM Mapping Debug ===")
                print(f"Total ADCP clusters: {len(adcp_clusters)}")
                print(f"Total OMM clusters: {len(omm_clusters)}")
                if adcp_clusters:
                    print(f"ADCP cluster numbers: {[c['cluster_num'] for c in adcp_clusters[:10]]}")
                if omm_clusters:
                    print(f"OMM model_num: {[o['model_num'] for o in omm_clusters[:10]]}")
                    print(f"OMM adcp_rank: {[o.get('adcp_rank', 'N/A') for o in omm_clusters[:10]]}")
                    print(f"OMM cluster_size: {[o.get('cluster_size', 'N/A') for o in omm_clusters[:10]]}")
            
            fig, axes = plt.subplots(3, 1, figsize=(16, 18))
            
            # PLOT 1: ADCP affinities vs cluster size
            sorted_adcp = sorted(adcp_clusters, key=lambda x: x['affinity'])
            cluster_nums_sorted = [c['cluster_num'] for c in sorted_adcp]
            affinities_sorted = [c['affinity'] for c in sorted_adcp]
            cluster_sizes_sorted = [c['cluster_size'] for c in sorted_adcp]
            
            x_pos = np.arange(len(cluster_nums_sorted))
            axes[0].bar(x_pos, cluster_sizes_sorted, color='lightcoral', edgecolor='black', alpha=0.7)
            axes[0].set_xlabel('ADCP Affinity (kcal/mol, sorted)')
            axes[0].set_ylabel('Cluster Size')
            axes[0].set_title(f'ADCP Affinity vs Cluster Size ({len(cluster_nums_sorted)} Clusters)')
            axes[0].grid(True, alpha=0.3, axis='y')
            
            xtick_labels = [f'{aff:.1f}' for aff in affinities_sorted]
            if len(cluster_nums_sorted) > 50:
                interval = max(1, len(cluster_nums_sorted) // 20)
                xtick_labels = [label if i % interval == 0 else '' for i, label in enumerate(xtick_labels)]
            # Fix: Set ticks before setting tick labels
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels(xtick_labels, fontsize=8, rotation=90)
            
            adcp_data = pd.DataFrame({
                'cluster_num': cluster_nums_sorted,
                'affinity': affinities_sorted,
                'cluster_size': cluster_sizes_sorted
            })
            adcp_xlsx_file = f"{output_prefix}_adcp_cluster_analysis.xlsx"
            adcp_data.to_excel(adcp_xlsx_file, sheet_name="ADCP_Clusters", index=False, engine='openpyxl')
            
            # PLOT 2 & 3: OMM energies - CORRECTED MAPPING USING adcp_rank
            if omm_clusters and len(omm_clusters) > 0:
                # Build mapping: OMM adcp_rank -> OMM data
                # The adcp_rank column in OMM table indicates which ADCP cluster this model belongs to
                omm_by_adcp_cluster = {}
                
                for omm in omm_clusters:
                    # Use adcp_rank as the key (this is the ADCP cluster number)
                    adcp_rank = omm.get('adcp_rank')
                    if adcp_rank is not None:
                        # Store OMM data indexed by ADCP cluster number
                        omm_by_adcp_cluster[adcp_rank] = omm
                
                if verbose:
                    print(f"Built OMM mapping with {len(omm_by_adcp_cluster)} entries")
                    print(f"OMM ADCP cluster keys (first 10): {sorted(list(omm_by_adcp_cluster.keys()))[:10]}")
                
                # Get ADCP clusters in original order (1-100)
                adcp_clusters_original = sorted(adcp_clusters, key=lambda x: x['cluster_num'])
                omm_interaction_energies = []
                omm_dE_interaction = []
                cluster_nums_for_plot = []
                matched_count = 0
                
                for adcp_cluster in adcp_clusters_original:
                    cluster_num = adcp_cluster['cluster_num']
                    cluster_nums_for_plot.append(cluster_num)
                    
                    # Look up OMM data using ADCP cluster number
                    if cluster_num in omm_by_adcp_cluster:
                        omm_entry = omm_by_adcp_cluster[cluster_num]
                        omm_interaction_energies.append(omm_entry.get('interaction_energy', np.nan))
                        omm_dE_interaction.append(omm_entry.get('dE_interaction', np.nan))
                        matched_count += 1
                        
                        if verbose and matched_count <= 5:
                            print(f"Matched ADCP cluster {cluster_num} to OMM model {omm_entry.get('model_num')}")
                    else:
                        omm_interaction_energies.append(np.nan)
                        omm_dE_interaction.append(np.nan)
                
                if verbose:
                    print(f"Successfully matched {matched_count}/{len(adcp_clusters_original)} ADCP clusters to OMM data")
                
                x_pos = np.arange(len(adcp_clusters_original))
                
                # Plot 2: Interaction Energy
                interaction_energies = np.array(omm_interaction_energies)
                mask = np.isnan(interaction_energies)
                colors = ['lightgreen' if not m else 'lightgray' for m in mask]
                
                axes[1].bar(x_pos, interaction_energies, color=colors, edgecolor='black', alpha=0.7)
                axes[1].set_xlabel('ADCP Cluster')
                axes[1].set_ylabel('OMM Interaction Energy (E_Complex-E_Receptor)')
                axes[1].set_title(f'OMM Interaction Energy by ADCP Cluster (Data: {sum(~mask)}/{len(cluster_nums_for_plot)})')
                axes[1].grid(True, alpha=0.3, axis='y')
                
                # Plot 3: dE Interaction
                dE_data = np.array(omm_dE_interaction)
                mask_dE = np.isnan(dE_data)
                colors_dE = ['gold' if not m else 'lightgray' for m in mask_dE]
                
                axes[2].bar(x_pos, dE_data, color=colors_dE, edgecolor='black', alpha=0.7)
                axes[2].set_xlabel('ADCP Cluster')
                axes[2].set_ylabel('OMM dE_interaction (E_Complex-E_rec-E_pep)')
                axes[2].set_title(f'OMM dE_interaction by ADCP Cluster (Data: {sum(~mask_dE)}/{len(cluster_nums_for_plot)})')
                axes[2].grid(True, alpha=0.3, axis='y')
                
                # Add diagnostic info if any data is missing
                num_missing = np.sum(mask)
                if num_missing > 0:
                    diagnostic_text = f'Note: {num_missing}/{len(cluster_nums_for_plot)} clusters have no OMM data'
                    axes[1].text(0.02, 0.98, diagnostic_text, ha='left', va='top',
                                transform=axes[1].transAxes, fontsize=9,
                                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                    axes[2].text(0.02, 0.98, diagnostic_text, ha='left', va='top',
                                transform=axes[2].transAxes, fontsize=9,
                                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                
                # Save OMM data to CSV
                omm_data_to_save = []
                for i, cluster_num in enumerate(cluster_nums_for_plot):
                    omm_data_to_save.append({
                        'cluster_num': cluster_num,
                        'interaction_energy': omm_interaction_energies[i],
                        'dE_interaction': omm_dE_interaction[i]
                    })
                
                omm_df = pd.DataFrame(omm_data_to_save)
                omm_xlsx_file = f"{output_prefix}_omm_cluster_analysis.xlsx"
                omm_df.to_excel(omm_xlsx_file, sheet_name="OMM_Clusters", index=False, engine='openpyxl')
                
            else:
                # No OMM data available
                axes[1].text(0.5, 0.5, 'No OMM clusters in DLG file', ha='center', va='center',
                            transform=axes[1].transAxes, fontsize=12, color='red', weight='bold')
                axes[1].set_title('OMM Interaction Energy by ADCP Cluster')
                
                axes[2].text(0.5, 0.5, 'No OMM clusters in DLG file', ha='center', va='center',
                            transform=axes[2].transAxes, fontsize=12, color='red', weight='bold')
                axes[2].set_title('OMM dE_interaction by ADCP Cluster')
                
                omm_xlsx_file = None
                
                if verbose:
                    print("WARNING: No OMM clusters found in DLG data")
            
            fig.suptitle(f'DLG Cluster Analysis\nTotal Poses: {dlg_data.get("total_poses", 0)}, Contact Cutoff: {dlg_data.get("contact_cutoff", 0.8)}', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            cluster_file = f"{output_prefix}_cluster_analysis.png"
            plt.savefig(cluster_file, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            return cluster_file, adcp_xlsx_file, omm_xlsx_file
            
        except Exception as e:
            print(f"ERROR: Could not create cluster analysis: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return None, None, None
    
    @staticmethod
    def plot_sorted_omm_interaction_energy(dlg_data, output_prefix, dpi=600, verbose=False):
        """Create sorted OMM interaction energy plot (best to worst)."""
        try:
            adcp_clusters = dlg_data['adcp_clusters']
            omm_clusters = dlg_data['omm_clusters']
            
            if not omm_clusters or len(omm_clusters) == 0:
                if verbose:
                    print("No OMM clusters available for sorted interaction energy plot")
                return None, None
            
            # Build mapping from adcp_rank to OMM data
            omm_by_adcp_cluster = {}
            for omm in omm_clusters:
                adcp_rank = omm.get('adcp_rank')
                if adcp_rank is not None:
                    omm_by_adcp_cluster[adcp_rank] = omm
            
            # Collect interaction energies with cluster numbers
            energy_data = []
            adcp_clusters_original = sorted(adcp_clusters, key=lambda x: x['cluster_num'])
            
            for adcp_cluster in adcp_clusters_original:
                cluster_num = adcp_cluster['cluster_num']
                if cluster_num in omm_by_adcp_cluster:
                    energy = omm_by_adcp_cluster[cluster_num].get('interaction_energy', np.nan)
                    energy_data.append((cluster_num, energy))
                else:
                    energy_data.append((cluster_num, np.nan))
            
            # Remove NaN entries and sort by energy (ascending - best to worst)
            energy_data_valid = [(c, e) for c, e in energy_data if not np.isnan(e)]
            energy_data_sorted = sorted(energy_data_valid, key=lambda x: x[1])
            
            if not energy_data_sorted:
                if verbose:
                    print("No valid OMM interaction energy data found")
                return None, None
            
            cluster_labels = [f"C{num}" for num, _ in energy_data_sorted]
            energies_sorted = [energy for _, energy in energy_data_sorted]
            
            # Create plot
            plt.figure(figsize=(20, 10))
            x_pos = np.arange(len(cluster_labels))
            plt.bar(x_pos, energies_sorted, color='lightgreen', edgecolor='black', alpha=0.7)
            
            plt.xlabel('ADCP Cluster (sorted by interaction energy)', fontsize=12)
            plt.ylabel('OMM Interaction Energy (E_Complex-E_Receptor)', fontsize=12)
            plt.title(f'Sorted OMM Interaction Energy - All Clusters (N={len(energies_sorted)})', fontsize=14)
            plt.xticks(x_pos, cluster_labels, rotation=90, fontsize=8)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on select bars
            for i, (bar, energy) in enumerate(zip(range(len(energies_sorted)), energies_sorted)):
                if i < 10 or i >= len(energies_sorted) - 10 or i % max(1, len(energies_sorted) // 20) == 0:
                    plt.text(bar, energy, f'{energy:.0f}', ha='center', va='bottom', fontsize=6)
            
            plt.tight_layout()
            
            # Save plot
            sorted_energy_file = f"{output_prefix}_sorted_omm_interaction_energy.png"
            plt.savefig(sorted_energy_file, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            # Save data to xlsx
            sorted_energy_xlsx = f"{output_prefix}_sorted_omm_interaction_energy.xlsx"
            sorted_df = pd.DataFrame({
                'cluster': cluster_labels,
                'cluster_num': [num for num, _ in energy_data_sorted],
                'interaction_energy': energies_sorted
            })
            sorted_df.to_excel(sorted_energy_xlsx, sheet_name="Sorted_Energy", index=False, engine='openpyxl')
            
            if verbose:
                print(f"Saved sorted interaction energy plot to: {sorted_energy_file}")
            
            return sorted_energy_file, sorted_energy_xlsx
            
        except Exception as e:
            print(f"WARNING: Could not create sorted interaction energy plot: {e}", file=sys.stderr)
            return None, None
    
    @staticmethod
    def plot_sorted_omm_dE_interaction(dlg_data, output_prefix, dpi=600, verbose=False):
        """Create sorted OMM dE_interaction plot (best to worst)."""
        try:
            adcp_clusters = dlg_data['adcp_clusters']
            omm_clusters = dlg_data['omm_clusters']
            
            if not omm_clusters or len(omm_clusters) == 0:
                if verbose:
                    print("No OMM clusters available for sorted dE_interaction plot")
                return None, None
            
            # Build mapping from adcp_rank to OMM data
            omm_by_adcp_cluster = {}
            for omm in omm_clusters:
                adcp_rank = omm.get('adcp_rank')
                if adcp_rank is not None:
                    omm_by_adcp_cluster[adcp_rank] = omm
            
            # Collect dE_interaction values with cluster numbers
            dE_data = []
            adcp_clusters_original = sorted(adcp_clusters, key=lambda x: x['cluster_num'])
            
            for adcp_cluster in adcp_clusters_original:
                cluster_num = adcp_cluster['cluster_num']
                if cluster_num in omm_by_adcp_cluster:
                    dE = omm_by_adcp_cluster[cluster_num].get('dE_interaction', np.nan)
                    dE_data.append((cluster_num, dE))
                else:
                    dE_data.append((cluster_num, np.nan))
            
            # Remove NaN entries and sort by dE (ascending - best to worst)
            dE_data_valid = [(c, d) for c, d in dE_data if not np.isnan(d)]
            dE_data_sorted = sorted(dE_data_valid, key=lambda x: x[1])
            
            if not dE_data_sorted:
                if verbose:
                    print("No valid OMM dE_interaction data found")
                return None, None
            
            cluster_labels = [f"C{num}" for num, _ in dE_data_sorted]
            dE_sorted = [dE for _, dE in dE_data_sorted]
            
            # Create plot
            plt.figure(figsize=(20, 10))
            x_pos = np.arange(len(cluster_labels))
            plt.bar(x_pos, dE_sorted, color='gold', edgecolor='black', alpha=0.7)
            
            plt.xlabel('ADCP Cluster (sorted by dE_interaction)', fontsize=12)
            plt.ylabel('OMM dE_interaction (E_Complex-E_rec-E_pep)', fontsize=12)
            plt.title(f'Sorted OMM dE_interaction - All Clusters (N={len(dE_sorted)})', fontsize=14)
            plt.xticks(x_pos, cluster_labels, rotation=90, fontsize=8)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on select bars
            for i, (bar, dE) in enumerate(zip(range(len(dE_sorted)), dE_sorted)):
                if i < 10 or i >= len(dE_sorted) - 10 or i % max(1, len(dE_sorted) // 20) == 0:
                    plt.text(bar, dE, f'{dE:.0f}', ha='center', va='bottom', fontsize=6)
            
            plt.tight_layout()
            
            # Save plot
            sorted_dE_file = f"{output_prefix}_sorted_omm_dE_interaction.png"
            plt.savefig(sorted_dE_file, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            # Save data to xlsx
            sorted_dE_xlsx = f"{output_prefix}_sorted_omm_dE_interaction.xlsx"
            sorted_df = pd.DataFrame({
                'cluster': cluster_labels,
                'cluster_num': [num for num, _ in dE_data_sorted],
                'dE_interaction': dE_sorted
            })
            sorted_df.to_excel(sorted_dE_xlsx, sheet_name="Sorted_dE", index=False, engine='openpyxl')
            
            if verbose:
                print(f"Saved sorted dE_interaction plot to: {sorted_dE_file}")
            
            return sorted_dE_file, sorted_dE_xlsx
            
        except Exception as e:
            print(f"WARNING: Could not create sorted dE_interaction plot: {e}", file=sys.stderr)
            return None, None


class SummaryPrinter:
    """Prints analysis summaries."""
    
    @staticmethod
    def print_summary(df, matrix, output_prefix, total_frames, args,
                     histogram_file=None, cluster_analysis_file=None,
                     sorted_energy_file=None, sorted_dE_file=None,
                     adcp_csv_file=None, omm_csv_file=None,
                     sorted_energy_csv=None, sorted_dE_csv=None,
                     peptide_barplot_csv=None, protein_barplot_csv=None,
                     row_norm_file=None, col_norm_file=None):
        """Print analysis summary to console."""
        print("\n" + "="*60)
        print("CONTACT ANALYSIS SUMMARY")
        print("="*60)
        if args.dlg:
            print(f"DLG file:               {args.dlg}")
        if args.protein:
            print(f"Protein file:          {args.protein}")
        if args.ligand:
            print(f"Ligand file:           {args.ligand}")
        print(f"Distance cutoff:       {args.cutoff} Å")
        print(f"Total frames analyzed: {total_frames}")
        print(f"Unique residue pairs:  {len(df)}")
        print(f"Heatmap dimensions:    {matrix.shape[0]} x {matrix.shape[1]}")
        print()
        
        print("TOP 10 INTERACTIONS (by frequency):")
        print("-"*60)
        print(f"{'Protein Residue':<25} {'Peptide Residue':<25} {'Frequency':<10}")
        print("-"*60)
        
        for _, row in df.head(10).iterrows():
            print(f"{row['Protein_Residue']:<25} {row['Peptide_Residue']:<25} {row['Frequency']:.3f}")
        
        print()
        print("OUTPUT FILES:")
        print(f"  Full contact data:   {output_prefix}_full_data.xlsx")
        print(f"  Contact matrix:      {output_prefix}_matrix.xlsx")
        if not args.no_plot:
            print(f"  Heatmap:             {output_prefix}_heatmap.{args.heatmap_format}")
            print(f"  Peptide barplot:     {output_prefix}_peptide_barplot.{args.heatmap_format}")
            print(f"  Protein barplot:     {output_prefix}_protein_barplot.{args.heatmap_format}")
        if args.dlg:
            if histogram_file:
                print(f"  Histogram:           {histogram_file}")
            if cluster_analysis_file:
                print(f"  Cluster analysis:    {cluster_analysis_file}")
        print("="*60)


def create_readme_file(output_dir, analysis_date, args, total_frames, num_contacts):
    """Create a README.txt file."""
    readme_content = f"""================================================================================
                          CRANKPEP ANALYSIS REPORT
                            CrankPep Analysis Tool
================================================================================

ANALYSIS TIMESTAMP: {analysis_date}
OUTPUT DIRECTORY: {output_dir}

================================================================================
                                GENERAL INFORMATION
================================================================================

This directory contains the output files from the CrankPep Contact Analysis Tool,
a comprehensive analysis python script for protein-peptide interactions from AutoDock 
CrankPep docking results.

ANALYSIS PARAMETERS:
  - Protein File: {args.protein}
  - Ligand File: {args.ligand}
  - DLG File: {args.dlg if args.dlg else 'Not provided'}
  - Distance Cutoff: {args.cutoff} Å
  - Total Frames Analyzed: {total_frames}
  - Unique Contact Pairs: {num_contacts}
  - Sort Residues: {'Yes' if args.sort_residues else 'No'}
  - Horizontal Heatmap: {'Yes' if args.horizontal else 'No'}
  - Threshold: {args.threshold}

================================================================================
                            OUTPUT FILE DESCRIPTIONS
================================================================================

CONTACT FREQUENCY DATA FILES (.xlsx format):
───────────────────────────────────────────────────────────────────────────────

1. contacts_full_data.xlsx
   Description: Complete contact frequency data with all protein-peptide residue pairs
   Sheets:
     - Contact_Data: Contains all contact pairs with counts and frequencies
   Columns:
     - Protein_Residue: Identifier of protein residue (ChainID_ResidueNumber_ResidueType)
     - Peptide_Residue: Identifier of peptide residue
     - Contact_Count: Number of frames where contact occurred
     - Frequency: Contact frequency (0-1), normalized by total frames

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

BARPLOT DATA FILES (.xlsx format):
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

DLG ANALYSIS FILES (.xlsx format) - Generated only with --dlg option:
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

VISUALIZATION FILES (Image format):
───────────────────────────────────────────────────────────────────────────────

12. contacts_heatmap.{args.heatmap_format}
    Description: Full contact frequency heatmap
    Content: Visual representation of all protein-peptide contacts
    Color: Yellow (0.0 frequency) → Red (1.0 frequency)

13. contacts_filtered_heatmap.{args.heatmap_format}
    Description: Filtered contact heatmap (contacts above threshold only)
    Content: Shows only significant contacts

14. contacts_peptide_barplot.{args.heatmap_format}
    Description: Bar plot of peptide residue interaction strength
    Content: Total contacts per peptide residue

15. contacts_protein_barplot.{args.heatmap_format}
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

20. contacts_monomer_heatmap.{args.heatmap_format}
    Description: Monomer interaction heatmap
    Content: Summed interactions across all monomers
    Note: Generated only if --num-monomers > 1

MONOMER FILES (.xlsx format) - Generated only with --num-monomers > 1:
───────────────────────────────────────────────────────────────────────────────

21. contacts_monomer_matrix.xlsx
    Description: Contact frequency matrix summed across monomers
    Sheets:
      - Monomer_Matrix: Aggregated contact data per monomer unit

================================================================================
                                HOW THE SCRIPT WORKS
================================================================================

1. STRUCTURE LOADING
   - Loads protein structure from PDBQT, PDB, or MOL2 format
   - Loads ligand/peptide structures with multiple models/conformations

2. ATOM SELECTION
   - Uses MDAnalysis selection strings to identify relevant atoms
   - Supports custom residue ranges (e.g., "resid 31:40")
   - Separates protein and peptide atoms for distance calculations

3. CONTACT ANALYSIS
   - Iterates through all frames/models
   - Calculates pairwise distances between protein and peptide atoms
   - Identifies contacts below the specified cutoff distance
   - Counts unique residue-residue contacts per frame
   - Normalizes by total frames to get contact frequency

4. RESIDUE IDENTIFICATION
   - Generates unique identifiers: ChainID_ResidueNumber_ResidueType
   - Optionally sorts residues from N-terminus to C-terminus
   - Handles chain separations and multiple segments

5. MATRIX GENERATION
   - Creates pivot table with protein residues as rows, peptides as columns
   - Values represent contact frequencies (0.0 to 1.0)
   - Generates row and column-normalized versions
   - Filters data based on user-specified threshold

6. VISUALIZATION
   - Generates heatmaps showing contact patterns
   - Creates bar plots for residue-level analysis
   - Produces histograms for energy distributions
   - Supports multiple image formats (PNG, PDF, SVG, JPG)

7. DLG ANALYSIS (Optional)
   - Parses AutoDock CrankPep DLG (Docking Log) files
   - Extracts ADCP clustering data and affinities
   - Extracts OMM (OpenMM) rescoring results
   - Maps OMM data to ADCP clusters using adcp_rank column
   - Generates sorted energy visualizations
   - Provides comprehensive energy statistics

8. MONOMER ANALYSIS (Optional)
   - For multimeric proteins, aggregates contacts per monomer unit
   - Sums contact frequencies across all monomer copies
   - Generates separate monomer interaction heatmap

================================================================================
                                    DATA FORMAT
================================================================================

All data files are saved in XLSX (Excel) format for better compatibility with:
  - Microsoft Excel
  - LibreOffice Calc
  - Python pandas (direct import)
  - R (readxl package)
  - Other tools for statistical analysis and visualization (like GraphPad Prism, Origin etc.)

Each XLSX file may contain multiple sheets for organized data presentation.

================================================================================
                                USAGE EXAMPLES
================================================================================

Basic contact analysis:
  python crankpep_analysis.py -p protein.pdb -l peptide.pdb -c 4.0 -o results

With DLG file analysis:
  python crankpep_analysis.py --dlg docking.dlg -p protein.pdb \\
    -l ligand_poses.pdb -c 4.0 -o dlg_analysis

With custom selections and filtering:
  python crankpep_analysis.py -p protein.pdb -l peptide.pdb \\
    --prot-sel "protein and resid 31:100" \\
    --pep-sel "resname LIG" \\
    --threshold 0.1 -o filtered_results

With monomer analysis:
  python crankpep_analysis.py -p tetramer.pdb -l peptide.pdb \\
    --num-monomers 4 -o monomer_analysis

Verbose output with custom format:
  python crankpep_analysis.py -p protein.pdb -l peptide.pdb -v \\
    --heatmap-format pdf --dpi 300 -o high_quality

================================================================================
                                CONTACT AND SUPPORT
================================================================================

For questions or issues, please refer to the main script documentation and help:
  python crankpep_analysis.py -h

For bug reports or feature requests, please contact me on github :)
https://github.com/georgeegreat

================================================================================
LICENSE AND DISCLAIMER
================================================================================
This analysis tool is provided for research purposes. Users assume full 
responsibility for the interpretation and use of results.

MIT License

Copyright (c) 2026 Egor Pivovarov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

================================================================================
Generated by: CrankPep Analysis Tool
Analysis Date: {analysis_date}
================================================================================
"""
    
    readme_path = os.path.join(output_dir, "README.txt")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    return readme_path

def main():
    """Main entry point."""
    args = ArgumentParser.parse_arguments()
    
    if not args.protein or not args.ligand:
        print("ERROR: Both --protein and --ligand are required", file=sys.stderr)
        sys.exit(1)
    
    FileValidator.validate_files(args.protein, args.ligand, args.dlg)
    
    prot_sel = ResidueHandler.parse_residue_range(args.prot_sel)
    pep_sel = ResidueHandler.parse_residue_range(args.pep_sel)
    
    output_dir = FileValidator.create_output_directory(args.protein, args.ligand, args.dlg)
    args.output = os.path.join(output_dir, args.output)
    
    # Get current timestamp
    analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if args.verbose:
        print(f"Output directory: {output_dir}")
        print(f"Analysis timestamp: {analysis_date}")
    
    # Parse DLG if provided
    dlg_data = None
    if args.dlg:
        parser = DLGParser(args.dlg, args.verbose)
        dlg_data = parser.parse(args.max_models)
        DLGPlotter.plot_histogram(dlg_data, args.output, args.dpi, args.verbose)
        DLGPlotter.plot_cluster_analysis(dlg_data, args.output, args.dpi, args.verbose)
        DLGPlotter.plot_sorted_omm_interaction_energy(dlg_data, args.output, args.dpi, args.verbose)
        DLGPlotter.plot_sorted_omm_dE_interaction(dlg_data, args.output, args.dpi, args.verbose)
    
    # Analyze contacts
    analyzer = ContactAnalyzer(
        args.protein, args.ligand, args.cutoff, prot_sel, pep_sel,
        args.verbose, args.max_models, args.sort_residues
    )
    contact_counter, prot_atoms, pep_atoms, total_frames, prot_res_set, pep_res_set, protein_res_freq, peptide_res_freq = analyzer.analyze()
    
    # Create matrices
    matrix, df, row_norm, col_norm = ContactMatrixBuilder.create_matrix(
        contact_counter, prot_res_set, pep_res_set, args.output, total_frames, 100
    )
    
    # Generate plots
    if not args.no_plot:
        heatmap_file, filtered_file, filtered_matrix = HeatmapPlotter.plot(
            matrix, args.output, args.heatmap_format, args.dpi, args.cutoff,
            args.horizontal, args.threshold, 12, 8
        )
        
        if filtered_matrix is not None and not filtered_matrix.empty:
            BarplotPlotter.plot_peptide(filtered_matrix, args.output, args.heatmap_format, args.dpi, args.horizontal, total_frames)
            BarplotPlotter.plot_protein(filtered_matrix, args.output, args.heatmap_format, args.dpi, args.horizontal, total_frames)
        
        if args.num_monomers > 1:
            MonomericInteractionPlotter.create_heatmap(
                matrix, args.num_monomers, args.output, args.heatmap_format, args.dpi, 12, 10, args.verbose
            )
    
    # Create README file
    readme_file = create_readme_file(output_dir, analysis_date, args, total_frames, len(df))
    
    # Print summary
    SummaryPrinter.print_summary(df, matrix, args.output, total_frames, args)
    
    if args.verbose:
        print(f"README file created: {readme_file}")
        print("Analysis complete!")


if __name__ == "__main__":
    main()