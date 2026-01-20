#!/usr/bin/env python3
"""
OpenMM Peptide Ramachandran Analyzer
Calculates φ/ψ angles for peptide residues from OpenMM rescored PDB files.
Filters models by dE_interaction < 0 and creates comprehensive Ramachandran plots.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import MDAnalysis as mda
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
                            current_model['lines'] = current_lines + ['END\n']
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


class PhiPsiCalculator:
    """Calculates φ and ψ backbone dihedral angles for peptide residues."""
    
    @staticmethod
    def calculate_dihedral(p1: np.ndarray, p2: np.ndarray, 
                          p3: np.ndarray, p4: np.ndarray) -> float:
        """
        Calculate dihedral angle between four points.
        
        Returns:
            Dihedral angle in degrees (-180 to 180)
        """
        # Calculate vectors
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        
        # Calculate normals
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        
        # Normalize
        n1_norm = n1 / np.linalg.norm(n1)
        n2_norm = n2 / np.linalg.norm(n2)
        
        # Calculate angle
        m1 = np.cross(n1_norm, b2 / np.linalg.norm(b2))
        
        x = np.dot(n1_norm, n2_norm)
        y = np.dot(m1, n2_norm)
        
        # Return angle in degrees
        angle = np.degrees(np.arctan2(y, x))
        
        # Ensure angle is between -180 and 180
        if angle < -180:
            angle += 360
        elif angle > 180:
            angle -= 360
            
        return angle
    
    @classmethod
    def calculate_phi_psi(cls, universe: mda.Universe) -> pd.DataFrame:
        """
        Calculate φ and ψ angles for all peptide residues.
        
        Args:
            universe: MDAnalysis Universe containing peptide structure
            
        Returns:
            DataFrame with residue information and angles
        """
        # Select all peptide atoms
        peptide = universe.select_atoms("all")
        
        if len(peptide.residues) < 2:
            raise ValueError("Peptide must have at least 2 residues for dihedral calculation")
        
        results = []
        residues = list(peptide.residues)
        
        for i, res in enumerate(residues):
            res_data = {
                'residue_id': f"{res.segid}_{res.resid}_{res.resname}",
                'residue_number': res.resid,
                'residue_name': res.resname,
                'phi': None,
                'psi': None
            }
            
            # Calculate PHI angle (for residues after the first)
            if i > 0:
                try:
                    # Get atoms: C(i-1), N(i), CA(i), C(i)
                    prev_res = residues[i-1]
                    
                    C_prev = prev_res.atoms.select_atoms("name C")
                    N_curr = res.atoms.select_atoms("name N")
                    CA_curr = res.atoms.select_atoms("name CA")
                    C_curr = res.atoms.select_atoms("name C")
                    
                    if (len(C_prev) > 0 and len(N_curr) > 0 and 
                        len(CA_curr) > 0 and len(C_curr) > 0):
                        phi = cls.calculate_dihedral(
                            C_prev.positions[0],
                            N_curr.positions[0],
                            CA_curr.positions[0],
                            C_curr.positions[0]
                        )
                        res_data['phi'] = phi
                except Exception:
                    pass
            
            # Calculate PSI angle (for residues before the last)
            if i < len(residues) - 1:
                try:
                    # Get atoms: N(i), CA(i), C(i), N(i+1)
                    next_res = residues[i+1]
                    
                    N_curr = res.atoms.select_atoms("name N")
                    CA_curr = res.atoms.select_atoms("name CA")
                    C_curr = res.atoms.select_atoms("name C")
                    N_next = next_res.atoms.select_atoms("name N")
                    
                    if (len(N_curr) > 0 and len(CA_curr) > 0 and 
                        len(C_curr) > 0 and len(N_next) > 0):
                        psi = cls.calculate_dihedral(
                            N_curr.positions[0],
                            CA_curr.positions[0],
                            C_curr.positions[0],
                            N_next.positions[0]
                        )
                        res_data['psi'] = psi
                except Exception:
                    pass
            
            results.append(res_data)
        
        return pd.DataFrame(results)
    
    @classmethod
    def analyze_openmm_models(cls, pdb_file: str, max_de_interaction: float = 0.0,
                            max_models: Optional[int] = None, peptide_chain: str = 'Z',
                            verbose: bool = False) -> pd.DataFrame:
        """
        Analyze φ/ψ angles across OpenMM models with energy filtering.
        
        Args:
            pdb_file: Path to OpenMM multi-model PDB file
            max_de_interaction: Maximum dE_Interaction to include
            max_models: Maximum number of models to analyze
            peptide_chain: Chain identifier for peptide
            verbose: Print progress information
            
        Returns:
            DataFrame with angles from all filtered models
        """
        reader = OpenMMPDBReader(pdb_file, verbose, max_models)
        models = reader.parse_models(max_de_interaction, peptide_chain)
        
        if not models:
            raise ValueError(f"No models found with dE_interaction < {max_de_interaction}")
        
        all_angles = []
        
        for i, model in enumerate(models):
            if verbose and i % 10 == 0:
                print(f"  Processing model {i+1}/{len(models)}...")
            
            # Create temporary file for this model
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
                tmp.writelines(model['lines'])
                tmp_file = tmp.name
            
            try:
                # Load universe and calculate angles
                u = mda.Universe(tmp_file)
                df_angles = cls.calculate_phi_psi(u)
                
                # Add model information
                df_angles['model'] = model['model_num']
                df_angles['model_index'] = i
                df_angles['de_interaction'] = model['de_interaction']
                
                all_angles.append(df_angles)
                
                # Clean up
                del u
                os.unlink(tmp_file)
                
            except Exception as e:
                if verbose:
                    print(f"  Warning: Failed to analyze model {model['model_num']}: {e}")
                try:
                    os.unlink(tmp_file)
                except:
                    pass
                continue
        
        if not all_angles:
            raise ValueError("No valid models could be processed")
        
        # Combine all results
        combined_df = pd.concat(all_angles, ignore_index=True)
        
        if verbose:
            print(f"\nAnalysis Summary:")
            print(f"  Total angle measurements: {len(combined_df)}")
            print(f"  Unique residues: {combined_df['residue_id'].nunique()}")
            print(f"  Models analyzed: {combined_df['model'].nunique()}")
            print(f"  Complete φ/ψ pairs: {len(combined_df.dropna(subset=['phi', 'psi']))}")
        
        # Add energy records to combined dataframe
        energy_df = pd.DataFrame(reader.energy_records)
        return combined_df, energy_df


class EnhancedRamachandranPlotter:
    """Creates enhanced Ramachandran plots with optimized layouts."""
    
    # Region boundaries for common secondary structures
    REGIONS = {
        'beta_sheet': {
            'phi_range': (-180, 0),
            'psi_range': (90, 180),
            'color': 'red',
            'alpha': 0.1,
            'label': 'β-sheet'
        },
        'alpha_helix': {
            'phi_range': (-120, -30),
            'psi_range': (-60, 30),
            'color': 'blue',
            'alpha': 0.1,
            'label': 'α-helix'
        },
        'left_handed_helix': {
            'phi_range': (30, 120),
            'psi_range': (30, 120),
            'color': 'green',
            'alpha': 0.1,
            'label': 'Left-handed'
        }
    }
    
    @staticmethod
    def _add_region_boxes(ax):
        """Add boxes for common secondary structure regions."""
        regions = EnhancedRamachandranPlotter.REGIONS
        
        for region_name, region in regions.items():
            phi_min, phi_max = region['phi_range']
            psi_min, psi_max = region['psi_range']
            
            rect = plt.Rectangle((phi_min, psi_min), 
                               phi_max - phi_min, 
                               psi_max - psi_min,
                               facecolor=region['color'], 
                               alpha=region['alpha'],
                               edgecolor=region['color'], 
                               linewidth=1.5)
            ax.add_patch(rect)
            
            # Add label
            ax.text((phi_min + phi_max) / 2, (psi_min + psi_max) / 2,
                   region['label'], ha='center', va='center',
                   fontweight='bold', color=region['color'], fontsize=8)
    
    @staticmethod
    def create_comprehensive_ramachandran(df: pd.DataFrame, output_file: str, 
                                         dpi: int = 300, verbose: bool = False) -> str:
        """
        Create comprehensive Ramachandran plot with all data.
        
        Args:
            df: DataFrame with phi and psi angles
            output_file: Output file path
            dpi: Image resolution
            verbose: Print progress information
            
        Returns:
            Path to saved plot file
        """
        # Filter out residues with missing angles
        plot_df = df.dropna(subset=['phi', 'psi']).copy()
        
        if plot_df.empty:
            raise ValueError("No complete φ/ψ angle pairs found")
        
        if verbose:
            print(f"Creating comprehensive Ramachandran plot from {len(plot_df)} angle pairs...")
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # 1. Scatter plot with residue coloring
        scatter = axes[0, 0].scatter(plot_df['phi'], plot_df['psi'], 
                                     c=plot_df['residue_number'], 
                                     cmap='viridis', alpha=0.7, s=20, 
                                     edgecolor='black', linewidth=0.3)
        axes[0, 0].set_xlabel('φ (degrees)', fontsize=11)
        axes[0, 0].set_ylabel('ψ (degrees)', fontsize=11)
        axes[0, 0].set_title(f'Ramachandran Plot by Residue Number\n(n={len(plot_df)} points)', 
                           fontsize=12, fontweight='bold')
        axes[0, 0].set_xlim(-180, 180)
        axes[0, 0].set_ylim(-180, 180)
        axes[0, 0].grid(True, alpha=0.2)
        axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        axes[0, 0].axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Add region boxes
        EnhancedRamachandranPlotter._add_region_boxes(axes[0, 0])
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[0, 0])
        cbar.set_label('Residue Number', fontsize=10)
        
        # 2. Hexbin density plot (INCREASED WIDTH)
        hb = axes[0, 1].hexbin(plot_df['phi'], plot_df['psi'], gridsize=40, 
                              cmap='YlOrRd', mincnt=1)
        axes[0, 1].set_xlabel('φ (degrees)', fontsize=11)
        axes[0, 1].set_ylabel('ψ (degrees)', fontsize=11)
        axes[0, 1].set_title('2D Density Distribution\n(Hexbin)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlim(-180, 180)
        axes[0, 1].set_ylim(-180, 180)
        axes[0, 1].grid(True, alpha=0.2)
        axes[0, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        axes[0, 1].axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Add region boxes
        EnhancedRamachandranPlotter._add_region_boxes(axes[0, 1])
        
        # Add colorbar
        cb1 = plt.colorbar(hb, ax=axes[0, 1])
        cb1.set_label('Count', fontsize=10)
        
        # 3. Kernel Density Estimate (INCREASED WIDTH)
        x = plot_df['phi'].values
        y = plot_df['psi'].values
        
        if len(x) > 1:
            try:
                from scipy.stats import gaussian_kde
                k = gaussian_kde(np.vstack([x, y]), bw_method=0.1)
                
                # Create grid
                xi, yi = np.mgrid[-180:180:200j, -180:180:200j]
                zi = k(np.vstack([xi.flatten(), yi.flatten()]))
                
                # Reshape and plot
                zi = zi.reshape(xi.shape)
                
                contour = axes[1, 0].contourf(xi, yi, zi, levels=30, cmap='RdYlBu_r', alpha=0.9)
                axes[1, 0].set_xlabel('φ (degrees)', fontsize=11)
                axes[1, 0].set_ylabel('ψ (degrees)', fontsize=11)
                axes[1, 0].set_title('Kernel Density Estimate\n(Contour Plot)', 
                                    fontsize=12, fontweight='bold')
                axes[1, 0].set_xlim(-180, 180)
                axes[1, 0].set_ylim(-180, 180)
                axes[1, 0].grid(True, alpha=0.2)
                axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
                axes[1, 0].axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
                
                # Add region boxes
                EnhancedRamachandranPlotter._add_region_boxes(axes[1, 0])
                
                # Add colorbar
                cb2 = plt.colorbar(contour, ax=axes[1, 0])
                cb2.set_label('Density', fontsize=10)
                
            except Exception as e:
                if verbose:
                    print(f"  Warning: KDE calculation failed: {e}")
                axes[1, 0].text(0.5, 0.5, 'KDE calculation failed', 
                              ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Kernel Density Estimate (failed)')
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient data for KDE', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Kernel Density Estimate')
        
        # 4. Residue-type colored scatter
        if 'residue_name' in plot_df.columns:
            # Create color map for residue types
            unique_res = plot_df['residue_name'].unique()
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_res)))
            color_map = dict(zip(unique_res, colors))
            
            # Plot each residue type
            for res_name in unique_res:
                res_data = plot_df[plot_df['residue_name'] == res_name]
                axes[1, 1].scatter(res_data['phi'], res_data['psi'], 
                                  color=color_map[res_name], alpha=0.7, s=15,
                                  edgecolor='black', linewidth=0.3, label=res_name)
            
            axes[1, 1].set_xlabel('φ (degrees)', fontsize=11)
            axes[1, 1].set_ylabel('ψ (degrees)', fontsize=11)
            axes[1, 1].set_title('Ramachandran by Residue Type', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlim(-180, 180)
            axes[1, 1].set_ylim(-180, 180)
            axes[1, 1].grid(True, alpha=0.2)
            axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
            axes[1, 1].axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Add region boxes
            EnhancedRamachandranPlotter._add_region_boxes(axes[1, 1])
            
            # Add legend (limit to 10 entries)
            if len(unique_res) <= 10:
                axes[1, 1].legend(loc='upper right', fontsize=8, ncol=2)
            else:
                # Create separate legend figure
                fig_legend, ax_legend = plt.subplots(figsize=(8, 0.5 * len(unique_res)))
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor=color_map[r], label=r) 
                                 for r in unique_res]
                ax_legend.legend(handles=legend_elements, loc='center', ncol=4)
                ax_legend.axis('off')
                legend_file = output_file.replace('.png', '_legend.png')
                fig_legend.savefig(legend_file, dpi=dpi, bbox_inches='tight')
                plt.close(fig_legend)
        
        else:
            axes[1, 1].text(0.5, 0.5, 'No residue name information', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Ramachandran by Residue Type')
        
        # Add overall title
        plt.suptitle(f'OpenMM Peptide Ramachandran Analysis\n'
                    f'Models with dE_interaction < 0 (n={plot_df["model"].nunique()} models, '
                    f'{len(plot_df)} angle pairs)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"Saved comprehensive Ramachandran plot to: {output_file}")
        
        return output_file
    
    @staticmethod
    def create_angle_distributions_optimized(df: pd.DataFrame, output_file: str, 
                                           dpi: int = 300, verbose: bool = False) -> str:
        """
        Create optimized angle distribution plots (5 subplots per row, no circles).
        
        Args:
            df: DataFrame with phi and psi angles
            output_file: Output file path
            dpi: Image resolution
            verbose: Print progress information
            
        Returns:
            Path to saved plot file
        """
        # Filter out terminal residues (those with missing phi or psi)
        plot_df = df.dropna(subset=['phi', 'psi']).copy()
        
        # Identify internal residues (have both phi and psi)
        internal_residues = plot_df['residue_id'].unique()
        
        # Group by residue identifier for internal residues only
        residue_groups = []
        for residue_id, group in plot_df.groupby('residue_id'):
            if residue_id in internal_residues:
                residue_groups.append((residue_id, group))
        
        if not residue_groups:
            raise ValueError("No internal residues with complete φ/ψ angles found")
        
        # Sort by residue number
        residue_groups.sort(key=lambda x: int(x[0].split('_')[1]) if x[0].split('_')[1].isdigit() else 0)
        
        if verbose:
            print(f"Creating distribution plots for {len(residue_groups)} internal residues...")
        
        # Set up subplots: 5 per row
        n_residues = len(residue_groups)
        cols = 5
        rows = (n_residues + cols - 1) // cols
        
        # Adjust figure size based on number of rows
        fig_height = max(3, rows * 2.5)
        fig, axes = plt.subplots(rows, cols, figsize=(16, fig_height))
        
        # Flatten axes array for easy indexing
        if rows > 1:
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes] if cols == 1 else axes
        
        for idx, (residue_id, res_df) in enumerate(residue_groups):
            if idx >= len(axes_flat):
                break
            
            ax = axes_flat[idx]
            
            # Create simple scatter plot without circles/ellipses
            if not res_df.empty:
                # Use small, semi-transparent points
                ax.scatter(res_df['phi'], res_df['psi'], 
                          c='blue', alpha=0.6, s=15, 
                          edgecolor='none')
                
                # Calculate and plot mean position as a cross
                mean_phi = res_df['phi'].mean()
                mean_psi = res_df['psi'].mean()
                ax.scatter([mean_phi], [mean_psi], 
                          c='red', s=80, marker='x', linewidth=2,
                          label='Mean')
            
            # Extract residue info for title
            match = _RESIDUE_ID_PATTERN.match(residue_id)
            if match:
                chain, num, name = match.groups()
                # Use compact title format
                title = f"{name}{num}"
            else:
                title = f"Res{idx+1}"
            
            ax.set_title(title, fontsize=9, pad=2)
            ax.set_xlabel('φ', fontsize=8)
            ax.set_ylabel('ψ', fontsize=8)
            ax.set_xlim(-180, 180)
            ax.set_ylim(-180, 180)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            
            # Add thin zero lines
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(residue_groups), len(axes_flat)):
            axes_flat[idx].axis('off')
        
        plt.suptitle('Ramachandran Plots by Residue Position\n(Internal residues only, 5 per row)', 
                    fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"Saved optimized residue distribution plots to: {output_file}")
        
        return output_file
    
    @staticmethod
    def create_density_heatmap(df: pd.DataFrame, output_file: str, 
                              dpi: int = 300, verbose: bool = False) -> str:
        """
        Create wide density heatmap for Ramachandran angles.
        
        Args:
            df: DataFrame with phi and psi angles
            output_file: Output file path
            dpi: Image resolution
            verbose: Print progress information
            
        Returns:
            Path to saved plot file
        """
        # Filter out residues with missing angles
        plot_df = df.dropna(subset=['phi', 'psi']).copy()
        
        if plot_df.empty:
            raise ValueError("No complete φ/ψ angle pairs found")
        
        if verbose:
            print(f"Creating wide density heatmap from {len(plot_df)} angle pairs...")
        
        # Create WIDE figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 1. Hexbin plot (wider)
        hb = ax1.hexbin(plot_df['phi'], plot_df['psi'], gridsize=50, 
                       cmap='YlOrRd', mincnt=1)
        ax1.set_xlabel('φ (degrees)', fontsize=12)
        ax1.set_ylabel('ψ (degrees)', fontsize=12)
        ax1.set_title('2D Density Distribution (Hexbin)', fontsize=13, fontweight='bold')
        ax1.set_xlim(-180, 180)
        ax1.set_ylim(-180, 180)
        ax1.grid(True, alpha=0.2)
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Add region boxes
        EnhancedRamachandranPlotter._add_region_boxes(ax1)
        
        # Add colorbar
        cb1 = plt.colorbar(hb, ax=ax1)
        cb1.set_label('Count', fontsize=11)
        
        # 2. 2D Histogram (wider)
        hist, xedges, yedges = np.histogram2d(plot_df['phi'], plot_df['psi'], 
                                              bins=50, range=[[-180, 180], [-180, 180]])
        
        # Plot histogram
        im = ax2.imshow(hist.T, origin='lower', cmap='hot', 
                       extent=[-180, 180, -180, 180], aspect='auto')
        ax2.set_xlabel('φ (degrees)', fontsize=12)
        ax2.set_ylabel('ψ (degrees)', fontsize=12)
        ax2.set_title('2D Histogram', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.2)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Add region boxes
        EnhancedRamachandranPlotter._add_region_boxes(ax2)
        
        # Add colorbar
        cb2 = plt.colorbar(im, ax=ax2)
        cb2.set_label('Count', fontsize=11)
        
        plt.suptitle(f'OpenMM Peptide Ramachandran Density Analysis\n'
                    f'Models with dE_interaction < 0 (n={plot_df["model"].nunique()} models)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"Saved wide density heatmap to: {output_file}")
        
        return output_file


class StatisticsGenerator:
    """Generates comprehensive statistics and reports."""
    
    @staticmethod
    def generate_summary_stats(df: pd.DataFrame, energy_df: pd.DataFrame, 
                              output_prefix: str, verbose: bool = False):
        """
        Generate comprehensive statistics and save to files.
        
        Args:
            df: DataFrame with angle data
            energy_df: DataFrame with energy data
            output_prefix: Output file prefix
            verbose: Print progress information
        """
        if verbose:
            print("\nGenerating summary statistics...")
        
        # 1. Basic angle statistics
        stats_data = []
        
        # Overall statistics
        phi_data = df['phi'].dropna()
        psi_data = df['psi'].dropna()
        
        if not phi_data.empty:
            stats_data.append({
                'measure': 'phi_all',
                'n': len(phi_data),
                'mean': phi_data.mean(),
                'std': phi_data.std(),
                'min': phi_data.min(),
                'max': phi_data.max(),
                'median': phi_data.median(),
                'q1': phi_data.quantile(0.25),
                'q3': phi_data.quantile(0.75)
            })
        
        if not psi_data.empty:
            stats_data.append({
                'measure': 'psi_all',
                'n': len(psi_data),
                'mean': psi_data.mean(),
                'std': psi_data.std(),
                'min': psi_data.min(),
                'max': psi_data.max(),
                'median': psi_data.median(),
                'q1': psi_data.quantile(0.25),
                'q3': psi_data.quantile(0.75)
            })
        
        # Statistics by residue type
        if 'residue_name' in df.columns:
            for res_name, group in df.groupby('residue_name'):
                phi_res = group['phi'].dropna()
                psi_res = group['psi'].dropna()
                
                if not phi_res.empty:
                    stats_data.append({
                        'measure': f'phi_{res_name}',
                        'n': len(phi_res),
                        'mean': phi_res.mean(),
                        'std': phi_res.std(),
                        'min': phi_res.min(),
                        'max': phi_res.max(),
                        'median': phi_res.median()
                    })
                
                if not psi_res.empty:
                    stats_data.append({
                        'measure': f'psi_{res_name}',
                        'n': len(psi_res),
                        'mean': psi_res.mean(),
                        'std': psi_res.std(),
                        'min': psi_res.min(),
                        'max': psi_res.max(),
                        'median': psi_res.median()
                    })
        
        # Save statistics
        stats_df = pd.DataFrame(stats_data)
        stats_file = f"{output_prefix}_angle_statistics.xlsx"
        stats_df.to_excel(stats_file, index=False, engine='openpyxl')
        
        # 2. Secondary structure analysis
        if not df.empty:
            # Identify points in different regions
            region_counts = {'beta_sheet': 0, 'alpha_helix': 0, 'left_handed_helix': 0, 'other': 0}
            
            for _, row in df.dropna(subset=['phi', 'psi']).iterrows():
                phi = row['phi']
                psi = row['psi']
                
                in_region = False
                for region_name, region in EnhancedRamachandranPlotter.REGIONS.items():
                    phi_min, phi_max = region['phi_range']
                    psi_min, psi_max = region['psi_range']
                    
                    if (phi_min <= phi <= phi_max) and (psi_min <= psi <= psi_max):
                        region_counts[region_name] += 1
                        in_region = True
                        break
                
                if not in_region:
                    region_counts['other'] += 1
            
            total_points = sum(region_counts.values())
            ss_data = []
            for region, count in region_counts.items():
                percentage = (count / total_points * 100) if total_points > 0 else 0
                ss_data.append({
                    'region': region,
                    'count': count,
                    'percentage': percentage
                })
            
            ss_df = pd.DataFrame(ss_data)
            ss_file = f"{output_prefix}_secondary_structure.xlsx"
            ss_df.to_excel(ss_file, index=False, engine='openpyxl')
        
        # 3. Energy statistics
        if not energy_df.empty:
            energy_stats = {
                'n_models': len(energy_df),
                'mean_de': energy_df['de_interaction'].mean(),
                'std_de': energy_df['de_interaction'].std(),
                'min_de': energy_df['de_interaction'].min(),
                'max_de': energy_df['de_interaction'].max(),
                'median_de': energy_df['de_interaction'].median()
            }
            
            energy_stats_df = pd.DataFrame([energy_stats])
            energy_file = f"{output_prefix}_energy_statistics.xlsx"
            energy_stats_df.to_excel(energy_file, index=False, engine='openpyxl')
        
        # 4. Create summary report
        summary_file = f"{output_prefix}_analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("OPENMM PEPTIDE RAMACHANDRAN ANALYSIS SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("DATA OVERVIEW:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total angle measurements: {len(df)}\n")
            f.write(f"Unique residues: {df['residue_id'].nunique()}\n")
            f.write(f"Models analyzed: {df['model'].nunique()}\n")
            f.write(f"Complete φ/ψ pairs: {len(df.dropna(subset=['phi', 'psi']))}\n\n")
            
            if not energy_df.empty:
                f.write("ENERGY FILTERING:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Models with dE_interaction < 0: {len(energy_df)}\n")
                f.write(f"Mean dE_interaction: {energy_df['de_interaction'].mean():.2f} kcal/mol\n")
                f.write(f"Range: {energy_df['de_interaction'].min():.2f} to {energy_df['de_interaction'].max():.2f} kcal/mol\n\n")
            
            if not stats_df.empty:
                f.write("ANGLE STATISTICS (overall):\n")
                f.write("-" * 40 + "\n")
                phi_stats = stats_df[stats_df['measure'] == 'phi_all']
                psi_stats = stats_df[stats_df['measure'] == 'psi_all']
                
                if not phi_stats.empty:
                    f.write("φ angles:\n")
                    f.write(f"  Mean: {phi_stats['mean'].values[0]:.1f} ± {phi_stats['std'].values[0]:.1f}°\n")
                    f.write(f"  Range: {phi_stats['min'].values[0]:.1f} to {phi_stats['max'].values[0]:.1f}°\n")
                
                if not psi_stats.empty:
                    f.write("ψ angles:\n")
                    f.write(f"  Mean: {psi_stats['mean'].values[0]:.1f} ± {psi_stats['std'].values[0]:.1f}°\n")
                    f.write(f"  Range: {psi_stats['min'].values[0]:.1f} to {psi_stats['max'].values[0]:.1f}°\n\n")
            
            if 'ss_df' in locals():
                f.write("SECONDARY STRUCTURE DISTRIBUTION:\n")
                f.write("-" * 40 + "\n")
                for _, row in ss_df.iterrows():
                    f.write(f"{row['region']}: {row['count']} points ({row['percentage']:.1f}%)\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("OUTPUT FILES:\n")
            f.write("-" * 40 + "\n")
            f.write(f"1. {output_prefix}_angles.csv - Raw angle data\n")
            f.write(f"2. {output_prefix}_angle_statistics.xlsx - Statistical summary\n")
            f.write(f"3. {output_prefix}_secondary_structure.xlsx - Secondary structure analysis\n")
            f.write(f"4. {output_prefix}_energy_statistics.xlsx - Energy statistics\n")
            f.write(f"5. {output_prefix}_comprehensive.png - Comprehensive Ramachandran plot\n")
            f.write(f"6. {output_prefix}_density_heatmap.png - Wide density heatmap\n")
            f.write(f"7. {output_prefix}_distributions.png - Residue distribution plots (5 per row)\n")
            f.write(f"8. {output_prefix}_analysis_summary.txt - This summary file\n")
            f.write("=" * 70 + "\n")
        
        if verbose:
            print(f"✓ Statistics saved to: {stats_file}")
            print(f"✓ Summary report saved to: {summary_file}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate φ/ψ angles for peptide residues from OpenMM PDB files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python openmm_ramachandran.py --input openmm_results.pdb --output my_analysis
  python openmm_ramachandran.py --input relaxations.pdb --max-models 500 --chain A
  python openmm_ramachandran.py --input docked.pdb --verbose --dpi 600
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Input OpenMM multi-model PDB file')
    parser.add_argument('--output', '-o', default='angles_analysis',
                       help='Output directory and file prefix (default: angles_analysis)')
    parser.add_argument('--chain', '-c', default='Z',
                       help='Chain identifier for peptide (default: Z)')
    parser.add_argument('--max-models', type=int, default=None,
                       help='Maximum number of models to analyze')
    parser.add_argument('--max-de', type=float, default=0.0,
                       help='Maximum dE_Interaction to include (default: 0.0, only favorable)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for output images (default: 300)')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png',
                       help='Output image format (default: png)')
    parser.add_argument('--all-plots', action='store_true',
                       help='Generate all plot types')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Generate comprehensive Ramachandran plot')
    parser.add_argument('--density', action='store_true',
                       help='Generate wide density heatmap')
    parser.add_argument('--distributions', action='store_true',
                       help='Generate residue distribution plots (5 per row)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()


def main():
    """Main analysis pipeline."""
    args = parse_arguments()
    
    # Create output directory
    output_dir = f"angles_{args.output}"
    os.makedirs(output_dir, exist_ok=True)
    output_prefix = os.path.join(output_dir, args.output)
    
    print(f"\n{'='*70}")
    print("OPENMM PEPTIDE RAMACHANDRAN ANALYSIS")
    print(f"{'='*70}")
    print(f"Input file:          {args.input}")
    print(f"Output directory:    {output_dir}")
    print(f"Peptide chain:       {args.chain}")
    print(f"Max dE_interaction:  {args.max_de} (only models with dE < {args.max_de})")
    print(f"Max models:          {args.max_models or 'All'}")
    print(f"Image DPI:           {args.dpi}")
    print()
    
    # Determine which plots to generate
    if args.all_plots or not any([args.comprehensive, args.density, args.distributions]):
        args.comprehensive = args.density = args.distributions = True
    
    try:
        # Step 1: Calculate φ/ψ angles with energy filtering
        if args.verbose:
            print("Step 1: Calculating φ/ψ angles with energy filtering...")
        
        df_angles, energy_df = PhiPsiCalculator.analyze_openmm_models(
            pdb_file=args.input,
            max_de_interaction=args.max_de,
            max_models=args.max_models,
            peptide_chain=args.chain,
            verbose=args.verbose
        )
        
        # Save raw angle data
        csv_file = f"{output_prefix}_angles.csv"
        df_angles.to_csv(csv_file, index=False)
        print(f"✓ Angle data saved to: {csv_file}")
        
        # Save energy data
        if not energy_df.empty:
            energy_csv = f"{output_prefix}_energies.csv"
            energy_df.to_csv(energy_csv, index=False)
            print(f"✓ Energy data saved to: {energy_csv}")
        
        # Step 2: Generate statistics
        StatisticsGenerator.generate_summary_stats(df_angles, energy_df, output_prefix, args.verbose)
        
        # Step 3: Generate plots
        if args.verbose:
            print("\nStep 3: Generating plots...")
        
        plotter = EnhancedRamachandranPlotter()
        generated_plots = []
        
        # Comprehensive Ramachandran plot
        if args.comprehensive:
            comprehensive_file = f"{output_prefix}_comprehensive.{args.format}"
            try:
                plotter.create_comprehensive_ramachandran(
                    df=df_angles,
                    output_file=comprehensive_file,
                    dpi=args.dpi,
                    verbose=args.verbose
                )
                generated_plots.append(comprehensive_file)
                print(f"✓ Comprehensive plot saved to: {comprehensive_file}")
            except Exception as e:
                print(f"✗ Failed to create comprehensive plot: {e}")
        
        # Wide density heatmap
        if args.density:
            density_file = f"{output_prefix}_density_heatmap.{args.format}"
            try:
                plotter.create_density_heatmap(
                    df=df_angles,
                    output_file=density_file,
                    dpi=args.dpi,
                    verbose=args.verbose
                )
                generated_plots.append(density_file)
                print(f"✓ Density heatmap saved to: {density_file}")
            except Exception as e:
                print(f"✗ Failed to create density heatmap: {e}")
        
        # Optimized residue distribution plots (5 per row, no circles)
        if args.distributions:
            distributions_file = f"{output_prefix}_distributions.{args.format}"
            try:
                plotter.create_angle_distributions_optimized(
                    df=df_angles,
                    output_file=distributions_file,
                    dpi=args.dpi,
                    verbose=args.verbose
                )
                generated_plots.append(distributions_file)
                print(f"✓ Residue distribution plots saved to: {distributions_file}")
            except Exception as e:
                print(f"✗ Failed to create distribution plots: {e}")
        
        # Final summary
        print(f"\n{'='*70}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*70}")
        print(f"Output directory: {output_dir}/")
        print(f"\nGenerated files:")
        print(f"  {os.path.basename(csv_file)}")
        print(f"  {os.path.basename(output_prefix)}_angle_statistics.xlsx")
        print(f"  {os.path.basename(output_prefix)}_secondary_structure.xlsx")
        print(f"  {os.path.basename(output_prefix)}_energy_statistics.xlsx")
        print(f"  {os.path.basename(output_prefix)}_analysis_summary.txt")
        for plot in generated_plots:
            print(f"  {os.path.basename(plot)}")
        
        print(f"\nAnalysis parameters:")
        print(f"  Models analyzed: {df_angles['model'].nunique()}")
        print(f"  Residues analyzed: {df_angles['residue_id'].nunique()}")
        print(f"  Angle pairs: {len(df_angles.dropna(subset=['phi', 'psi']))}")
        if not energy_df.empty:
            print(f"  Mean dE_interaction: {energy_df['de_interaction'].mean():.2f} kcal/mol")
        
        print(f"\nUse the Excel files for further statistical analysis in programs like:")
        print(f"  • GraphPad Prism")
        print(f"  • Origin")
        print(f"  • SPSS")
        print(f"  • R")
        print(f"\nThe Ramachandran plots show:")
        print(f"  • Blue region: α-helix conformations")
        print(f"  • Red region: β-sheet conformations")
        print(f"  • Green region: Left-handed helix conformations")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()