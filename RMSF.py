#!/usr/bin/env python3
"""
OpenMM Peptide RMSF Analyzer
Calculates Root Mean Square Fluctuation (RMSF) for peptide residues from OpenMM rescored PDB files.
Filters models by dE_interaction < 0 and creates comprehensive RMSF analysis plots and data files.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
import numpy as np
import pandas as pd
import argparse
import sys
import os
from pathlib import Path
import re
from scipy.stats import gaussian_kde
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')
import tempfile
import gc

# Regular expression patterns
_RESIDUE_ID_PATTERN = re.compile(r'^(\S+)_(\d+)_(\S+)$')
_OPENMM_DE_PATTERN = re.compile(r'USER:\s*dE_Interaction\s*=\s*([\d\.\-]+)')


class OpenMMPDBReader:
    """Reads and processes OpenMM rescored PDB files with energy filtering."""
    
    def __init__(self, pdb_file: str, verbose: bool = False, max_models: Optional[int] = None):
        self.pdb_file = pdb_file
        self.verbose = verbose
        self.max_models = max_models
        self.models = []
        self.energy_records = []
        
    def parse_models(self, max_de_interaction: float = 0.0, peptide_chain: str = 'Z') -> List[Dict]:
        """
        Parse OpenMM rescored PDB file, filtering by dE_interaction.
        
        Args:
            max_de_interaction: Maximum dE_Interaction to include (only models with dE < max_de)
            peptide_chain: Chain identifier for peptide
            
        Returns:
            List of dictionaries with model data
        """
        if self.verbose:
            print(f"Parsing OpenMM PDB file: {self.pdb_file}")
            print(f"Filtering by dE_interaction < {max_de_interaction}")
        
        models = []
        current_model = None
        current_lines = []
        current_de = None
        model_count = 0
        kept_count = 0
        
        with open(self.pdb_file, 'r') as f:
            for line in f:
                if line.startswith('MODEL'):
                    # Start new model
                    try:
                        model_num = int(line.split()[1])
                    except (IndexError, ValueError):
                        model_num = len(models) + 1
                    
                    current_model = {
                        'model_num': model_num,
                        'lines': [],
                        'de_interaction': None
                    }
                    current_lines = []
                    current_de = None
                
                elif line.startswith('USER:'):
                    if current_model is not None:
                        # Extract dE_Interaction
                        if 'dE_Interaction' in line:
                            match = _OPENMM_DE_PATTERN.search(line)
                            if match:
                                try:
                                    current_de = float(match.group(1))
                                    current_model['de_interaction'] = current_de
                                except ValueError:
                                    pass
                
                elif line.startswith('ATOM') or line.startswith('HETATM'):
                    if current_model is not None:
                        # Only include peptide atoms from specified chain
                        if len(line) >= 22:
                            atom_chain = line[21]
                            if atom_chain == peptide_chain or atom_chain == peptide_chain.upper():
                                current_lines.append(line)
                
                elif line.startswith('ENDMDL'):
                    if current_model and current_lines and current_de is not None:
                        # Check energy filter - only keep if dE < max_de (typically dE < 0)
                        if current_de < max_de_interaction:
                            current_model['lines'] = current_lines
                            models.append(current_model)
                            kept_count += 1
                            
                            # Store energy record
                            self.energy_records.append({
                                'model_num': current_model['model_num'],
                                'de_interaction': current_de
                            })
                            
                            if self.verbose and kept_count % 50 == 0:
                                print(f"  Kept {kept_count} models...")
                            
                            if self.max_models and kept_count >= self.max_models:
                                break
                    
                    model_count += 1
                    current_model = None
                    current_lines = []
                    current_de = None
        
        self.models = models
        
        if self.verbose:
            print(f"Total models parsed: {model_count}")
            print(f"Models kept (dE < {max_de_interaction}): {len(models)}")
            print(f"Filtering efficiency: {len(models)/max(1, model_count)*100:.1f}%")
        
        return models


class RMSFAnalyzer:
    """Calculates Root Mean Square Fluctuation (RMSF) for peptide residues."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.temp_files = []  # Track temporary files for cleanup
        
    def __del__(self):
        """Clean up temporary files when object is destroyed."""
        self.cleanup_temp_files()
    
    def cleanup_temp_files(self):
        """Clean up all temporary files."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
        self.temp_files = []
    
    def create_multi_model_pdb(self, models: List[Dict]) -> str:
        """
        Create a multi-model PDB file from model data.
        
        Args:
            models: List of model dictionaries
            
        Returns:
            Path to created temporary file
        """
        # Create temporary file
        fd, tmp_path = tempfile.mkstemp(suffix='.pdb')
        os.close(fd)  # Close the file descriptor
        
        try:
            with open(tmp_path, 'w') as f:
                for i, model in enumerate(models):
                    f.write(f"MODEL    {i+1}\n")
                    f.writelines(model['lines'])
                    f.write("ENDMDL\n")
                f.write("END\n")
            
            # Track for cleanup
            self.temp_files.append(tmp_path)
            
            if self.verbose:
                print(f"Created multi-model PDB file: {tmp_path}")
                print(f"  Models: {len(models)}")
                
            return tmp_path
            
        except Exception as e:
            # Clean up if there's an error
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise RuntimeError(f"Failed to create multi-model PDB: {e}")
    
    def load_models_from_pdb(self, pdb_file: str, max_de_interaction: float = 0.0,
                           max_models: Optional[int] = None, peptide_chain: str = 'Z') -> Tuple[mda.Universe, List[int]]:
        """
        Load peptide models from OpenMM PDB file and create a trajectory Universe.
        
        Args:
            pdb_file: Path to OpenMM multi-model PDB file
            max_de_interaction: Maximum dE_Interaction to include
            max_models: Maximum number of models to analyze
            peptide_chain: Chain identifier for peptide
            
        Returns:
            Tuple of (MDAnalysis Universe with multiple frames, list of model numbers)
        """
        reader = OpenMMPDBReader(pdb_file, self.verbose, max_models)
        models = reader.parse_models(max_de_interaction, peptide_chain)
        
        if not models:
            raise ValueError(f"No models found with dE_interaction < {max_de_interaction}")
        
        # Create multi-model PDB file
        tmp_file = self.create_multi_model_pdb(models)
        
        try:
            # Load the multi-model PDB as a trajectory
            universe = mda.Universe(tmp_file)
            model_numbers = [model['model_num'] for model in models]
            
            if self.verbose:
                print(f"Loaded {universe.trajectory.n_frames} frames from {len(models)} models")
                print(f"Number of atoms: {len(universe.atoms)}")
                print(f"Number of residues: {len(universe.residues)}")
            
            return universe, model_numbers
            
        except Exception as e:
            # Clean up temporary file
            if os.path.exists(tmp_file):
                os.unlink(tmp_file)
            raise RuntimeError(f"Failed to load trajectory: {e}")
    
    def calculate_rmsf(self, universe: mda.Universe, 
                      selection: str = "all",
                      align_selection: str = "backbone",
                      reference_frame: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Calculate RMSF for selected atoms after alignment.
        
        Args:
            universe: MDAnalysis Universe with multiple frames
            selection: Atom selection for RMSF calculation
            align_selection: Atom selection for structural alignment
            reference_frame: Frame index to use as reference for alignment
            
        Returns:
            Tuple of (RMSF values array, metadata dictionary)
        """
        if universe.trajectory.n_frames < 2:
            raise ValueError("Need at least 2 frames for RMSF calculation")
        
        if self.verbose:
            print(f"Calculating RMSF for {selection} atoms...")
            print(f"Aligning using {align_selection} atoms")
            print(f"Reference frame: {reference_frame}")
        
        # Select atoms for RMSF calculation
        atoms = universe.select_atoms(selection)
        align_atoms = universe.select_atoms(align_selection)
        
        if len(atoms) == 0:
            raise ValueError(f"No atoms selected with: {selection}")
        
        if len(align_atoms) == 0:
            raise ValueError(f"No atoms selected for alignment with: {align_selection}")
        
        # Store coordinates for all frames
        n_frames = universe.trajectory.n_frames
        n_atoms = len(atoms)
        
        if self.verbose:
            print(f"Processing {n_frames} frames, {n_atoms} atoms...")
        
        # Initialize coordinate array
        coords = np.zeros((n_frames, n_atoms, 3))
        
        # Get reference coordinates from reference frame
        universe.trajectory[reference_frame]
        reference_coords = align_atoms.positions.copy()
        
        # Align and store coordinates for each frame
        for i, ts in enumerate(universe.trajectory):
            if self.verbose and i % 10 == 0:
                print(f"  Processing frame {i+1}/{n_frames}...")
            
            # Calculate rotation matrix to align current frame to reference
            # using the Kabsch algorithm
            current_coords = align_atoms.positions
            
            # Center the coordinates
            current_centered = current_coords - np.mean(current_coords, axis=0)
            ref_centered = reference_coords - np.mean(reference_coords, axis=0)
            
            # Calculate covariance matrix
            H = current_centered.T @ ref_centered
            
            # Singular Value Decomposition
            U, S, Vt = np.linalg.svd(H)
            
            # Calculate rotation matrix
            R = Vt.T @ U.T
            
            # Ensure proper rotation (no reflection)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Apply rotation to all atoms (not just alignment atoms)
            all_current_coords = universe.atoms.positions
            all_centered = all_current_coords - np.mean(current_coords, axis=0)
            all_aligned = all_centered @ R + np.mean(reference_coords, axis=0)
            
            # Store coordinates of selected atoms
            # We need to find the indices of our selected atoms in the universe
            atom_indices = atoms.indices
            coords[i] = all_aligned[atom_indices]
        
        # Calculate average structure
        avg_coords = np.mean(coords, axis=0)
        
        # Calculate RMSF: sqrt(mean((x_i - x_avg)^2))
        squared_diff = np.sum((coords - avg_coords) ** 2, axis=2)  # Sum over x,y,z
        rmsf_values = np.sqrt(np.mean(squared_diff, axis=0))  # Average over frames
        
        # Calculate RMSF per residue
        residue_rmsf = {}
        residue_info = {}
        
        for atom in atoms:
            residue = atom.residue
            res_id = f"{residue.segid}_{residue.resid}_{residue.resname}"
            
            if res_id not in residue_rmsf:
                residue_rmsf[res_id] = []
                residue_info[res_id] = {
                    'residue_number': residue.resid,
                    'residue_name': residue.resname,
                    'chain': residue.segid
                }
            
            # Get RMSF value for this atom
            atom_index = list(atoms.indices).index(atom.index)
            residue_rmsf[res_id].append(rmsf_values[atom_index])
        
        # Average RMSF per residue
        avg_residue_rmsf = {res_id: np.mean(values) for res_id, values in residue_rmsf.items()}
        std_residue_rmsf = {res_id: np.std(values) for res_id, values in residue_rmsf.items()}
        
        # Sort by residue number
        sorted_residues = sorted(avg_residue_rmsf.items(), 
                               key=lambda x: int(x[0].split('_')[1]) if x[0].split('_')[1].isdigit() else 0)
        
        sorted_residue_ids = [res_id for res_id, _ in sorted_residues]
        sorted_rmsf_values = [avg_residue_rmsf[res_id] for res_id in sorted_residue_ids]
        sorted_rmsf_std = [std_residue_rmsf[res_id] for res_id in sorted_residue_ids]
        
        # Create comprehensive results dictionary
        results = {
            'atom_rmsf': rmsf_values,
            'residue_rmsf': avg_residue_rmsf,
            'residue_rmsf_std': std_residue_rmsf,
            'sorted_residue_ids': sorted_residue_ids,
            'sorted_rmsf_values': sorted_rmsf_values,
            'sorted_rmsf_std': sorted_rmsf_std,
            'residue_info': residue_info,
            'n_frames': n_frames,
            'n_atoms': n_atoms,
            'n_residues': len(avg_residue_rmsf),
            'selection': selection,
            'align_selection': align_selection,
            'reference_frame': reference_frame
        }
        
        if self.verbose:
            print(f"RMSF calculation complete:")
            print(f"  Min RMSF: {np.min(rmsf_values):.3f} Å")
            print(f"  Max RMSF: {np.max(rmsf_values):.3f} Å")
            print(f"  Mean RMSF: {np.mean(rmsf_values):.3f} Å")
            print(f"  Number of residues: {len(avg_residue_rmsf)}")
        
        return rmsf_values, results
    
    def calculate_per_atom_type_rmsf(self, universe: mda.Universe, 
                                   reference_frame: int = 0) -> Dict[str, np.ndarray]:
        """
        Calculate RMSF for different atom types (backbone vs sidechain).
        
        Args:
            universe: MDAnalysis Universe with multiple frames
            reference_frame: Frame index to use as reference
            
        Returns:
            Dictionary with RMSF for different atom types
        """
        atom_type_rmsf = {}
        
        # Define atom type selections
        atom_types = {
            'all': 'all',
            'backbone': 'backbone',
            'sidechain': 'sidechain',
            'CA': 'name CA',
            'C': 'name C',
            'N': 'name N',
            'O': 'name O'
        }
        
        for atom_type, selection in atom_types.items():
            try:
                rmsf_values, _ = self.calculate_rmsf(
                    universe, selection=selection, 
                    align_selection='backbone',
                    reference_frame=reference_frame
                )
                atom_type_rmsf[atom_type] = rmsf_values
                
                if self.verbose:
                    print(f"  {atom_type}: {len(rmsf_values)} atoms, "
                          f"mean RMSF = {np.mean(rmsf_values):.3f} Å")
                    
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Could not calculate RMSF for {atom_type}: {e}")
        
        return atom_type_rmsf


class RMSFPlotter:
    """Creates RMSF plots and visualizations."""
    
    @staticmethod
    def create_rmsf_bar_plot(results: Dict[str, Any], output_file: str,
                            dpi: int = 300, figsize: Tuple[int, int] = (16, 8),
                            color_by: str = 'residue_name', verbose: bool = False) -> str:
        """
        Create bar plot of RMSF values per residue.
        
        Args:
            results: RMSF calculation results dictionary
            output_file: Output file path
            dpi: Image resolution
            figsize: Figure size
            color_by: Color scheme ('residue_name', 'rmsf_value', 'uniform')
            verbose: Print progress information
            
        Returns:
            Path to saved plot file
        """
        if verbose:
            print(f"Creating RMSF bar plot...")
        
        # Extract data
        residue_ids = results['sorted_residue_ids']
        rmsf_values = results['sorted_rmsf_values']
        rmsf_std = results['sorted_rmsf_std']
        residue_info = results['residue_info']
        
        # Create labels (compact format)
        labels = []
        for res_id in residue_ids:
            match = _RESIDUE_ID_PATTERN.match(res_id)
            if match:
                chain, num, name = match.groups()
                labels.append(f"{name}{num}")
            else:
                labels.append(res_id)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # X positions
        x_pos = np.arange(len(labels))
        
        # Set colors
        if color_by == 'residue_name':
            # Color by amino acid type
            aa_colors = {
                'ALA': 'lightblue', 'ARG': 'blue', 'ASN': 'cyan', 'ASP': 'darkblue',
                'CYS': 'yellow', 'GLN': 'green', 'GLU': 'darkgreen', 'GLY': 'gray',
                'HIS': 'pink', 'ILE': 'orange', 'LEU': 'darkorange', 'LYS': 'purple',
                'MET': 'brown', 'PHE': 'darkred', 'PRO': 'magenta', 'SER': 'lightgreen',
                'THR': 'lime', 'TRP': 'indigo', 'TYR': 'red', 'VAL': 'gold'
            }
            
            colors = []
            for res_id in residue_ids:
                res_name = residue_info[res_id]['residue_name']
                colors.append(aa_colors.get(res_name, 'gray'))
        elif color_by == 'rmsf_value':
            # Color gradient based on RMSF value
            cmap = plt.cm.YlOrRd
            norm = plt.Normalize(min(rmsf_values), max(rmsf_values))
            colors = cmap(norm(rmsf_values))
        else:
            # Uniform color
            colors = 'steelblue'
        
        # Create bars with error bars
        bars = ax.bar(x_pos, rmsf_values, color=colors, edgecolor='black', 
                     linewidth=0.5, alpha=0.8, yerr=rmsf_std, capsize=3,
                     error_kw={'elinewidth': 1, 'ecolor': 'black', 'capsize': 5})
        
        # Customize plot
        ax.set_xlabel('Residue', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSF (Å)', fontsize=12, fontweight='bold')
        ax.set_title(f'Peptide Residue RMSF\n'
                    f'{results["n_frames"]} models, dE_interaction < 0',
                    fontsize=14, fontweight='bold')
        
        # Set x-ticks
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add horizontal lines for reference
        mean_rmsf = np.mean(rmsf_values)
        ax.axhline(y=mean_rmsf, color='red', linestyle='--', 
                  linewidth=1.5, alpha=0.7, label=f'Mean: {mean_rmsf:.3f} Å')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add text box with statistics
        stats_text = (f"Min: {np.min(rmsf_values):.3f} Å\n"
                     f"Max: {np.max(rmsf_values):.3f} Å\n"
                     f"Mean: {mean_rmsf:.3f} Å\n"
                     f"Std: {np.std(rmsf_values):.3f} Å")
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"Saved RMSF bar plot to: {output_file}")
        
        return output_file
    
    @staticmethod
    def create_rmsf_line_plot(results: Dict[str, Any], output_file: str,
                             dpi: int = 300, figsize: Tuple[int, int] = (16, 8),
                             with_confidence: bool = True, verbose: bool = False) -> str:
        """
        Create line plot of RMSF values with optional confidence intervals.
        
        Args:
            results: RMSF calculation results dictionary
            output_file: Output file path
            dpi: Image resolution
            figsize: Figure size
            with_confidence: Add confidence interval (mean ± std)
            verbose: Print progress information
            
        Returns:
            Path to saved plot file
        """
        if verbose:
            print(f"Creating RMSF line plot...")
        
        # Extract data
        residue_ids = results['sorted_residue_ids']
        rmsf_values = results['sorted_rmsf_values']
        rmsf_std = results['sorted_rmsf_std']
        
        # Create labels (compact format)
        labels = []
        for res_id in residue_ids:
            match = _RESIDUE_ID_PATTERN.match(res_id)
            if match:
                chain, num, name = match.groups()
                labels.append(f"{name}{num}")
            else:
                labels.append(res_id)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # X positions
        x_pos = np.arange(len(labels))
        
        # Plot line with markers
        line = ax.plot(x_pos, rmsf_values, marker='o', markersize=6, 
                      linewidth=2, color='darkblue', alpha=0.8,
                      label='RMSF')
        
        # Add confidence interval if requested
        if with_confidence and rmsf_std is not None:
            ax.fill_between(x_pos, 
                           np.array(rmsf_values) - np.array(rmsf_std),
                           np.array(rmsf_values) + np.array(rmsf_std),
                           alpha=0.3, color='steelblue',
                           label='Mean ± Std')
        
        # Customize plot
        ax.set_xlabel('Residue Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSF (Å)', fontsize=12, fontweight='bold')
        ax.set_title(f'Peptide Residue RMSF Profile\n'
                    f'{results["n_frames"]} models, dE_interaction < 0',
                    fontsize=14, fontweight='bold')
        ax.set_ylim(5, 14)
        # Set x-ticks
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        
        # Set y-limits
        y_min = max(0, np.min(rmsf_values) - 0.5)
        y_max = np.max(rmsf_values) + 0.5
        ax.set_ylim(y_min, y_max)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add horizontal lines for reference
        mean_rmsf = np.mean(rmsf_values)
        ax.axhline(y=mean_rmsf, color='red', linestyle='--', 
                  linewidth=1.5, alpha=0.7, label=f'Mean: {mean_rmsf:.3f} Å')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Highlight regions with high flexibility
        threshold = mean_rmsf + np.std(rmsf_values)
        high_flex_indices = [i for i, val in enumerate(rmsf_values) if val > threshold]
        
        if high_flex_indices:
            for idx in high_flex_indices:
                ax.axvspan(idx-0.4, idx+0.4, alpha=0.2, color='red')
            
            # Add text annotation
            ax.text(0.02, 0.15, f'High flexibility (> mean+std)\n{len(high_flex_indices)} residues',
                   transform=ax.transAxes, fontsize=10, color='red',
                   bbox=dict(boxstyle='round', facecolor='pink', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"Saved RMSF line plot to: {output_file}")
        
        return output_file
    
    @staticmethod
    def create_rmsf_density_plot(atom_type_rmsf: Dict[str, np.ndarray], 
                                output_file: str, dpi: int = 300,
                                figsize: Tuple[int, int] = (14, 10), 
                                vmax: float = 0.32, verbose: bool = False) -> str:
        """
        Create density plots for RMSF distributions by atom type with vmax constraint.
        
        Args:
            atom_type_rmsf: Dictionary with RMSF values for different atom types
            output_file: Output file path
            dpi: Image resolution
            figsize: Figure size
            vmax: Maximum value for KDE density (default: 0.32)
            verbose: Print progress information
            
        Returns:
            Path to saved plot file
        """
        if verbose:
            print(f"Creating RMSF density plots with vmax={vmax}...")
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes_flat = axes.flatten()
        
        # Plot each atom type
        atom_types = list(atom_type_rmsf.keys())[:4]  # Plot first 4 types
        
        for idx, atom_type in enumerate(atom_types):
            if idx >= len(axes_flat):
                break
            
            ax = axes_flat[idx]
            rmsf_data = atom_type_rmsf[atom_type]
            
            if len(rmsf_data) > 1:
                try:
                    # Create KDE plot with vmax constraint
                    kde = gaussian_kde(rmsf_data)
                    x_range = np.linspace(min(rmsf_data), max(rmsf_data), 1000)
                    y_range = kde(x_range)
                    
                    # Apply vmax constraint
                    y_range = np.clip(y_range, 0, vmax)
                    
                    # Plot KDE
                    ax.plot(x_range, y_range, 'b-', linewidth=2, alpha=0.8)
                    ax.fill_between(x_range, 0, y_range, alpha=0.3, color='blue')
                    
                    # Add mean line
                    mean_val = np.mean(rmsf_data)
                    ax.axvline(mean_val, color='red', linestyle='--', 
                              linewidth=1.5, alpha=0.7, label=f'Mean: {mean_val:.3f} Å')
                    
                except Exception as e:
                    if verbose:
                        print(f"  Warning: KDE failed for {atom_type}: {e}")
                    # Fallback to histogram
                    ax.hist(rmsf_data, bins=20, edgecolor='black', alpha=0.7, 
                           color='skyblue', density=True)
            elif len(rmsf_data) == 1:
                # Single data point - just mark it
                ax.axvline(rmsf_data[0], color='blue', linewidth=2)
                ax.text(0.5, 0.5, f'Single value: {rmsf_data[0]:.3f} Å',
                       ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes)
            
            ax.set_xlabel('RMSF (Å)', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'{atom_type} Atoms\n(n={len(rmsf_data)})', 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            if len(rmsf_data) > 0:
                ax.legend(fontsize=9)
            
            # Add statistics text
            if len(rmsf_data) > 0:
                stats_text = f"Min: {np.min(rmsf_data):.3f}\nMax: {np.max(rmsf_data):.3f}"
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                       fontsize=9, verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for idx in range(len(atom_types), len(axes_flat)):
            axes_flat[idx].axis('off')
        
        plt.suptitle(f'RMSF Distribution by Atom Type\n(KDE with vmax={vmax})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"Saved RMSF density plots to: {output_file}")
        
        return output_file


class RMSFDataExporter:
    """Exports RMSF data to various file formats."""
    
    @staticmethod
    def export_to_excel(results: Dict[str, Any], output_file: str, verbose: bool = False) -> str:
        """
        Export RMSF data to Excel file with multiple sheets.
        
        Args:
            results: RMSF calculation results dictionary
            output_file: Output file path
            verbose: Print progress information
            
        Returns:
            Path to saved Excel file
        """
        if verbose:
            print(f"Exporting RMSF data to Excel...")
        
        # Create Excel writer
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 1. Residue RMSF data
            residue_data = []
            for res_id in results['sorted_residue_ids']:
                rmsf_val = results['residue_rmsf'][res_id]
                rmsf_std = results['residue_rmsf_std'].get(res_id, 0)
                match = _RESIDUE_ID_PATTERN.match(res_id)
                if match:
                    chain, num, name = match.groups()
                    residue_data.append({
                        'Residue_ID': res_id,
                        'Chain': chain,
                        'Residue_Number': int(num) if num.isdigit() else num,
                        'Residue_Name': name,
                        'RMSF_Angstrom': rmsf_val,
                        'RMSF_Std': rmsf_std
                    })
            
            residue_df = pd.DataFrame(residue_data)
            residue_df = residue_df.sort_values('Residue_Number')
            residue_df.to_excel(writer, sheet_name='Residue_RMSF', index=False)
            
            # 2. Detailed statistics
            stats_data = {
                'Statistic': [
                    'Number of Models', 'Number of Residues', 'Number of Atoms',
                    'Overall Mean RMSF (Å)', 'Overall Std RMSF (Å)',
                    'Minimum RMSF (Å)', 'Maximum RMSF (Å)',
                    'Selection Criteria', 'Alignment Selection',
                    'Reference Frame'
                ],
                'Value': [
                    results['n_frames'], results['n_residues'], results['n_atoms'],
                    np.mean(results['sorted_rmsf_values']), 
                    np.std(results['sorted_rmsf_values']),
                    np.min(results['sorted_rmsf_values']),
                    np.max(results['sorted_rmsf_values']),
                    results['selection'], results['align_selection'],
                    results['reference_frame']
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='Analysis_Statistics', index=False)
            
            # 3. Raw atom RMSF values
            if 'atom_rmsf' in results:
                atom_df = pd.DataFrame({
                    'Atom_Index': range(len(results['atom_rmsf'])),
                    'RMSF_Angstrom': results['atom_rmsf']
                })
                atom_df.to_excel(writer, sheet_name='Atom_RMSF', index=False)
        
        if verbose:
            print(f"Saved RMSF data to Excel file: {output_file}")
        
        return output_file


def main():
    """Main RMSF analysis pipeline."""
    parser = argparse.ArgumentParser(
        description='Calculate RMSF for peptide residues from OpenMM PDB files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rmsf_analyzer.py --input openmm_results.pdb --output rmsf_analysis
  python rmsf_analyzer.py --input relaxations.pdb --max-models 500 --chain A
  python rmsf_analyzer.py --input docked.pdb --verbose --dpi 600
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Input OpenMM multi-model PDB file')
    parser.add_argument('--output', '-o', default='rmsf_analysis',
                       help='Output directory and file prefix (default: rmsf_analysis)')
    parser.add_argument('--chain', '-c', default='Z',
                       help='Chain identifier for peptide (default: Z)')
    parser.add_argument('--max-models', type=int, default=None,
                       help='Maximum number of models to analyze')
    parser.add_argument('--max-de', type=float, default=0.0,
                       help='Maximum dE_Interaction to include (default: 0.0, only favorable)')
    parser.add_argument('--selection', default='all',
                       help='Atom selection for RMSF calculation (default: all)')
    parser.add_argument('--align-selection', default='backbone',
                       help='Atom selection for alignment (default: backbone)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for output images (default: 300)')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png',
                       help='Output image format (default: png)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = f"rmsf_{args.output}"
    os.makedirs(output_dir, exist_ok=True)
    output_prefix = os.path.join(output_dir, args.output)
    
    print(f"\n{'='*70}")
    print("OPENMM PEPTIDE RMSF ANALYSIS")
    print(f"{'='*70}")
    print(f"Input file:          {args.input}")
    print(f"Output directory:    {output_dir}")
    print(f"Peptide chain:       {args.chain}")
    print(f"Max dE_interaction:  {args.max_de} (only models with dE < {args.max_de})")
    print(f"Max models:          {args.max_models or 'All'}")
    print(f"RMSF selection:      {args.selection}")
    print(f"Alignment selection: {args.align_selection}")
    print(f"Image DPI:           {args.dpi}")
    print()
    
    try:
        # Step 1: Load models and calculate RMSF
        if args.verbose:
            print("Step 1: Loading models and calculating RMSF...")
        
        analyzer = RMSFAnalyzer(verbose=args.verbose)
        
        # Load peptide models
        universe, model_numbers = analyzer.load_models_from_pdb(
            pdb_file=args.input,
            max_de_interaction=args.max_de,
            max_models=args.max_models,
            peptide_chain=args.chain
        )
        
        # Calculate RMSF
        rmsf_values, results = analyzer.calculate_rmsf(
            universe=universe,
            selection=args.selection,
            align_selection=args.align_selection,
            reference_frame=0
        )
        
        # Clean up temporary files
        analyzer.cleanup_temp_files()
        
        # Step 2: Export data to files
        if args.verbose:
            print("\nStep 2: Exporting RMSF data...")
        
        # Export to Excel
        excel_file = f"{output_prefix}_rmsf_data.xlsx"
        RMSFDataExporter.export_to_excel(results, excel_file, args.verbose)
        print(f"✓ RMSF data exported to Excel: {excel_file}")
        
        # Export to CSV
        residue_data = []
        for res_id in results['sorted_residue_ids']:
            rmsf_val = results['residue_rmsf'][res_id]
            rmsf_std = results['residue_rmsf_std'].get(res_id, 0)
            match = _RESIDUE_ID_PATTERN.match(res_id)
            if match:
                chain, num, name = match.groups()
                residue_data.append({
                    'residue_id': res_id,
                    'chain': chain,
                    'residue_number': int(num) if num.isdigit() else num,
                    'residue_name': name,
                    'rmsf_angstrom': rmsf_val,
                    'rmsf_std': rmsf_std
                })
        
        residue_df = pd.DataFrame(residue_data)
        residue_csv = f"{output_prefix}_residue_rmsf.csv"
        residue_df.to_csv(residue_csv, index=False)
        print(f"✓ Residue RMSF data exported to CSV: {residue_csv}")
        
        # Calculate atom type RMSF for plotting
        if args.verbose:
            print("\nCalculating atom type RMSF for plotting...")
        
        analyzer2 = RMSFAnalyzer(verbose=False)
        universe2, _ = analyzer2.load_models_from_pdb(
            pdb_file=args.input,
            max_de_interaction=args.max_de,
            max_models=args.max_models,
            peptide_chain=args.chain
        )
        
        atom_type_rmsf = analyzer2.calculate_per_atom_type_rmsf(universe2)
        analyzer2.cleanup_temp_files()
        
        # Step 3: Generate plots
        if args.verbose:
            print("\nStep 3: Generating plots...")
        
        plotter = RMSFPlotter()
        
        # Bar plot
        bar_plot_file = f"{output_prefix}_bar_plot.{args.format}"
        plotter.create_rmsf_bar_plot(
            results=results,
            output_file=bar_plot_file,
            dpi=args.dpi,
            verbose=args.verbose
        )
        print(f"✓ Bar plot saved to: {bar_plot_file}")
        
        # Line plot
        line_plot_file = f"{output_prefix}_line_plot.{args.format}"
        plotter.create_rmsf_line_plot(
            results=results,
            output_file=line_plot_file,
            dpi=args.dpi,
            verbose=args.verbose
        )
        print(f"✓ Line plot saved to: {line_plot_file}")
        
        # Density plot with vmax=0.32
        density_plot_file = f"{output_prefix}_density_plot.{args.format}"
        plotter.create_rmsf_density_plot(
            atom_type_rmsf=atom_type_rmsf,
            output_file=density_plot_file,
            dpi=args.dpi,
            vmax=0.32,
            verbose=args.verbose
        )
        print(f"✓ Density plots saved to: {density_plot_file}")
        
        # Step 4: Create summary report
        if args.verbose:
            print("\nStep 4: Creating summary report...")
        
        summary_file = f"{output_prefix}_analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("OPENMM PEPTIDE RMSF ANALYSIS SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("ANALYSIS PARAMETERS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Input file:          {args.input}\n")
            f.write(f"Peptide chain:       {args.chain}\n")
            f.write(f"Max dE_interaction:  {args.max_de}\n")
            f.write(f"Models analyzed:     {results['n_frames']}\n")
            f.write(f"RMSF selection:      {results['selection']}\n")
            f.write(f"Alignment selection: {results['align_selection']}\n")
            f.write(f"Reference frame:     {results['reference_frame']}\n\n")
            
            f.write("RESULTS SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Number of residues:  {results['n_residues']}\n")
            f.write(f"Number of atoms:     {results['n_atoms']}\n")
            overall_mean = np.mean(results['sorted_rmsf_values'])
            overall_std = np.std(results['sorted_rmsf_values'])
            f.write(f"Overall mean RMSF:   {overall_mean:.3f} Å\n")
            f.write(f"Overall std RMSF:    {overall_std:.3f} Å\n")
            f.write(f"Minimum RMSF:        {np.min(results['sorted_rmsf_values']):.3f} Å\n")
            f.write(f"Maximum RMSF:        {np.max(results['sorted_rmsf_values']):.3f} Å\n\n")
            
            # Top 5 most flexible residues
            f.write("MOST FLEXIBLE RESIDUES:\n")
            f.write("-" * 40 + "\n")
            
            # Create list of (residue_id, rmsf) tuples
            residue_rmsf_list = list(results['residue_rmsf'].items())
            # Sort by RMSF in descending order
            residue_rmsf_list.sort(key=lambda x: x[1], reverse=True)
            
            for i, (res_id, rmsf) in enumerate(residue_rmsf_list[:5]):
                match = _RESIDUE_ID_PATTERN.match(res_id)
                if match:
                    chain, num, name = match.groups()
                    f.write(f"{i+1}. {name}{num} (Chain {chain}): {rmsf:.3f} Å\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("OUTPUT FILES:\n")
            f.write("-" * 40 + "\n")
            f.write(f"1. {os.path.basename(excel_file)} - Complete RMSF data (Excel)\n")
            f.write(f"2. {os.path.basename(residue_csv)} - Residue RMSF data (CSV)\n")
            f.write(f"3. {os.path.basename(bar_plot_file)} - RMSF bar plot\n")
            f.write(f"4. {os.path.basename(line_plot_file)} - RMSF line plot\n")
            f.write(f"5. {os.path.basename(density_plot_file)} - Density plots (KDE vmax=0.32)\n")
            f.write(f"6. {os.path.basename(summary_file)} - This summary file\n")
            f.write("=" * 70 + "\n")
        
        print(f"✓ Summary report saved to: {summary_file}")
        
        # Final summary
        print(f"\n{'='*70}")
        print("RMSF ANALYSIS COMPLETE!")
        print(f"{'='*70}")
        print(f"Output directory: {output_dir}/")
        print(f"\nGenerated files:")
        print(f"  {os.path.basename(excel_file)}")
        print(f"  {os.path.basename(residue_csv)}")
        print(f"  {os.path.basename(bar_plot_file)}")
        print(f"  {os.path.basename(line_plot_file)}")
        print(f"  {os.path.basename(density_plot_file)}")
        print(f"  {os.path.basename(summary_file)}")
        
        print(f"\nKey findings:")
        print(f"  Models analyzed: {results['n_frames']}")
        print(f"  Mean RMSF: {overall_mean:.3f} Å")
        if residue_rmsf_list:
            print(f"  Most flexible residue: {residue_rmsf_list[0][0]} ({residue_rmsf_list[0][1]:.3f} Å)")
            print(f"  Least flexible residue: {residue_rmsf_list[-1][0]} ({residue_rmsf_list[-1][1]:.3f} Å)")
        
        print(f"\nUse the Excel file for further statistical analysis in:")
        print(f"  • GraphPad Prism")
        print(f"  • Origin")
        print(f"  • SPSS")
        print(f"  • R")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()