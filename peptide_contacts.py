#!/usr/bin/env python3
"""
Generate residue-residue contact probability maps from multi-model PDB files
containing a receptor and a peptide ligand. The peptide is automatically
identified as the chain with the smallest number of residues. For each frame,
a contact is counted if the minimum distance between any atoms of two peptide
residues is < cutoff. The contact probability is the fraction of frames in
which the contact exists. The diagonal is always 1 (self-contact).
Subtraction maps (file_i - file_j) are created for all pairs i < j.
"""

import os
import argparse
import warnings

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
from MDAnalysis.exceptions import NoDataError
import matplotlib.pyplot as plt


class ContactMapCalculator:
    """
    Compute a contact probability matrix from a multi‑model PDB file,
    automatically selecting the smallest chain as the peptide.
    """

    def __init__(self, pdb_file, cutoff=4.0):
        self.pdb_file = pdb_file
        self.cutoff = cutoff
        self.universe = mda.Universe(pdb_file)

        # Automatically identify the peptide chain
        self.peptide_identifier, self.use_chainid = self._find_peptide_chain()
        self.peptide_atoms = self._select_peptide_atoms()

        if len(self.peptide_atoms) == 0:
            raise ValueError(f"No peptide atoms found in {pdb_file}")

        # Get unique residues, sorted by residue number
        self.residues = sorted(
            self.peptide_atoms.residues, key=lambda r: r.resid
        )
        self.n_residues = len(self.residues)

        # Store residue names and numbers separately (for mutation handling)
        self.residue_names = [res.resname for res in self.residues]
        self.residue_numbers = [res.resnum for res in self.residues]

        self.residue_labels = [
            f"{res.resname}{res.resnum}" for res in self.residues
        ]

    def _find_peptide_chain(self):
        """
        Determine the chain with the fewest residues.
        Returns:
            identifier (str): chain ID or segid of the peptide.
            use_chainid (bool): True if identifier is a chainID, False if segid.
        """
        # Try to use chainID first
        try:
            chain_ids = np.unique(self.universe.atoms.chainIDs)
            use_chainid = True
            identifiers = chain_ids
        except (AttributeError, NoDataError):
            # Fall back to segid
            try:
                segids = np.unique(self.universe.atoms.segids)
                use_chainid = False
                identifiers = segids
            except (AttributeError, NoDataError):
                raise RuntimeError(
                    "Universe has neither chainID nor segid information. Cannot identify chains."
                )

        if len(identifiers) == 0:
            raise RuntimeError("No chains found in the PDB file.")

        # Count residues per chain
        residue_counts = {}
        for ident in identifiers:
            if use_chainid:
                atoms = self.universe.select_atoms(f"chainID {ident}")
            else:
                atoms = self.universe.select_atoms(f"segid {ident}")
            residue_counts[ident] = len(np.unique(atoms.resids))

        peptide_ident = min(residue_counts, key=residue_counts.get)
        print(
            f"Detected peptide chain: {peptide_ident} "
            f"(residues: {residue_counts[peptide_ident]})"
        )
        return peptide_ident, use_chainid

    def _select_peptide_atoms(self):
        """Select atoms belonging to the identified peptide chain."""
        if self.use_chainid:
            return self.universe.select_atoms(f"chainID {self.peptide_identifier}")
        else:
            return self.universe.select_atoms(f"segid {self.peptide_identifier}")

    def compute(self):
        """
        Returns:
            np.ndarray: symmetric matrix of shape (n_residues, n_residues)
                        with contact probabilities (0-1). Diagonal is 1.
        """
        contact_sum = np.zeros((self.n_residues, self.n_residues))
        n_frames = len(self.universe.trajectory)

        for ts in self.universe.trajectory:
            for i in range(self.n_residues):
                res_i = self.residues[i]
                atoms_i = res_i.atoms
                for j in range(i + 1, self.n_residues):
                    res_j = self.residues[j]
                    atoms_j = res_j.atoms

                    # Minimum distance between any atom of residue i and any atom of residue j
                    dist = np.min(
                        distance_array(atoms_i.positions, atoms_j.positions)
                    )
                    if dist < self.cutoff:
                        contact_sum[i, j] += 1
                        contact_sum[j, i] += 1

        # Set diagonal to number of frames (so final probability = 1)
        for i in range(self.n_residues):
            contact_sum[i, i] = n_frames

        contact_prob = contact_sum / n_frames
        return contact_prob


class MapPlotter:
    """Static methods for plotting contact and subtraction maps."""

    @staticmethod
    def plot_contact_map(
        matrix,
        title,
        filename,
        residue_labels=None,
        vmin=0,
        vmax=1,
        cmap="RdBu"):
        """
        Save a heatmap of the contact matrix.
        """
        fig, ax = plt.subplots(figsize=(18, 15))
        im = ax.imshow(
            matrix, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper"
        )
        _ = 20
        ax.set_title(title, fontsize=_)
        if residue_labels is not None:
            ax.set_xticks(range(len(residue_labels)))
            ax.set_xticklabels(residue_labels, rotation=72, fontsize=_)
            ax.set_yticks(range(len(residue_labels)))
            ax.set_yticklabels(residue_labels, fontsize=_)
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=_)
        plt.tight_layout()
        plt.savefig(filename, dpi=600)
        plt.close()

    @staticmethod
    def plot_subtraction_map(
        matrix,
        title,
        filename,
        residue_labels=None,
        vmin=-0.5,
        vmax=0.5,
        cmap="RdBu",
        ):
        """
        Save a heatmap of a subtraction matrix (diverging colormap).
        """
        MapPlotter.plot_contact_map(
            matrix, title, filename, residue_labels, vmin, vmax, cmap
        )


class AnalysisManager:
    """
    Orchestrates the analysis of multiple PDB files:
    - Computes contact maps for each file.
    - Saves individual contact maps.
    - Computes and saves subtraction maps for all pairs i < j.
    """

    def __init__(self, pdb_files, cutoff, output_dir):
        self.pdb_files = pdb_files
        self.cutoff = cutoff
        self.output_dir = output_dir
        # Each element: (basename, matrix, labels, names, numbers)
        self.contact_maps = []

    def run(self):
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        indiv_dir = os.path.join(self.output_dir, "individual")
        sub_dir = os.path.join(self.output_dir, "subtraction")
        os.makedirs(indiv_dir, exist_ok=True)
        os.makedirs(sub_dir, exist_ok=True)

        # Compute and save individual contact maps
        for pdb in self.pdb_files:
            print(f"Processing {pdb} ...")
            calculator = ContactMapCalculator(pdb, self.cutoff)
            matrix = calculator.compute()
            basename = os.path.splitext(os.path.basename(pdb))[0]
            self.contact_maps.append(
                (
                    basename,
                    matrix,
                    calculator.residue_labels,
                    calculator.residue_names,
                    calculator.residue_numbers,
                )
            )

            outfile = os.path.join(indiv_dir, f"contact_map_{basename}.png")
            MapPlotter.plot_contact_map(
                matrix,
                f"Inter-Residual Contact Map",
                outfile,
                calculator.residue_labels,
            )

        # Generate subtraction maps for all i < j
        n = len(self.contact_maps)
        for i in range(n):
            for j in range(i + 1, n):
                name_i, mat_i, labels_i, names_i, numbers_i = self.contact_maps[i]
                name_j, mat_j, _, names_j, numbers_j = self.contact_maps[j]

                # Ensure matrices have the same shape
                if mat_i.shape != mat_j.shape:
                    warnings.warn(
                        f"{name_i} and {name_j} have different shapes "
                        f"({mat_i.shape} vs {mat_j.shape}). Skipping subtraction."
                    )
                    continue

                # Build combined residue labels for the subtraction map
                combined_labels = []
                for idx in range(len(names_i)):
                    if names_i[idx] == names_j[idx] and numbers_i[idx] == numbers_j[idx]:
                        combined_labels.append(f"{names_i[idx]}{numbers_i[idx]}")
                    else:
                        combined_labels.append(
                            f"{names_i[idx]}{numbers_i[idx]} / {names_j[idx]}{numbers_j[idx]}"
                        )

                sub_matrix = mat_i - mat_j
                outfile = os.path.join(
                    sub_dir, f"subtraction_{name_i}_minus_{name_j}.png"
                )
                MapPlotter.plot_subtraction_map(
                    sub_matrix,
                    f"Inter-Residual Subtraction Contact Map",
                    outfile,
                    combined_labels,
                )

        print(f"All maps saved under {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate contact probability maps from multi‑model PDB files "
        "(peptide automatically identified as smallest chain)."
    )
    parser.add_argument(
        "pdb_files",
        nargs="+",
        help="One or more multi‑model PDB files (each model contains receptor and peptide).",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=4.0,
        help="Distance cutoff in Å for a contact (default: 4.0).",
    )
    parser.add_argument(
        "--outdir",
        default="./contact_maps",
        help="Output directory (default: ./contact_maps).",
    )
    args = parser.parse_args()

    manager = AnalysisManager(args.pdb_files, args.cutoff, args.outdir)
    manager.run()


if __name__ == "__main__":
    main()