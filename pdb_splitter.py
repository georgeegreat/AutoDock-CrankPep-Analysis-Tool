#!/usr/bin/env python3
"""
split_pdb_models.py – Split a multi‑model PDB file into single‑model files,
optionally keeping only selected chains.

Usage:
    python split_pdb_models.py input.pdb output_dir [--chains A B ...]
"""

import argparse
import os
import re
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Split a multi-model PDB file into single-model PDB files."
    )
    parser.add_argument("input", help="Path to the multi-model PDB file.")
    parser.add_argument("output_dir", help="Directory where single-model files will be written.")
    parser.add_argument(
        "--chains", nargs="+", default=None,
        help="Space-separated list of chain IDs to keep (e.g., A B). "
             "Use quotes and a space for a blank chain ID: ' '."
    )
    return parser.parse_args()

def chain_id_from_line(line: str) -> str:
    """Extract the chain ID from a PDB ATOM/HETATM/ANISOU line (column 22)."""
    if len(line) >= 22:
        return line[21]   # 0-based indexing, column 22
    return " "

def should_keep_line(line: str, inside_model: bool, chain_ids: set) -> bool:
    """
    Decide whether to keep a line based on chain filtering.
    - ATOM, HETATM, ANISOU : kept if chain_id is in chain_ids.
    - TER : skipped when chain filtering is active.
    - All other lines (including non‑coordinate records) are kept.
    """
    if not inside_model or chain_ids is None:
        return True

    record = line[:6].strip()
    if record in ("ATOM", "HETATM", "ANISOU"):
        return chain_id_from_line(line) in chain_ids
    if record == "TER":
        # When filtering chains, TER records become meaningless.
        return False
    # HEADER, REMARK, etc. inside MODEL – keep them.
    return True

def split_pdb(input_path: str, output_dir: str, chain_ids: set = None):
    """Read the multi‑model PDB and write single‑model files."""
    with open(input_path, 'r') as fh:
        lines = fh.readlines()

    # Collect header lines (everything before the first MODEL record)
    header_lines = []
    model_blocks = []          # each element: (model_number, list of lines)
    current_model_lines = None
    current_model_num = 0
    inside_model = False

    for line in lines:
        # Detect MODEL / ENDMDL boundaries
        if line.startswith("MODEL"):
            inside_model = True
            # Extract model number from the line, fallback to sequential
            m = re.match(r"MODEL\s+(\d+)", line)
            current_model_num = int(m.group(1)) if m else current_model_num + 1
            current_model_lines = [line]   # keep the MODEL line itself
            continue
        if line.startswith("ENDMDL") and inside_model:
            current_model_lines.append(line)
            model_blocks.append((current_model_num, current_model_lines))
            inside_model = False
            continue

        if not inside_model:
            # Outside any model: treat as header/footer
            header_lines.append(line)
        else:
            current_model_lines.append(line)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Determine base name for output files from the input filename
    stem = Path(input_path).stem   # e.g. "1abc" from "1abc.pdb"

    for model_num, block_lines in model_blocks:
        # Build the content for this single‑model PDB file
        # 1. Copy original header
        output_lines = list(header_lines)

        # 2. Model block – we rename the model to 1 for a true single‑model file
        rewritten_model = False
        for line in block_lines:
            # Keep the line only if it passes the chain filter
            if not should_keep_line(line, inside_model=True, chain_ids=chain_ids):
                continue

            # Replace the MODEL line to set model number = 1
            if line.startswith("MODEL") and not rewritten_model:
                output_lines.append("MODEL        1\n")
                rewritten_model = True
                continue
            output_lines.append(line)

        # 3. Add a final END record if not present
        if output_lines and not output_lines[-1].startswith("END"):
            output_lines.append("END\n")

        # Write the file
        out_name = f"{stem}_model_{model_num:04d}.pdb"
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, 'w') as out_fh:
            out_fh.writelines(output_lines)

    print(f"Written {len(model_blocks)} models to '{output_dir}'.")
    if chain_ids:
        print(f"Chains retained: {sorted(chain_ids)}")

def main():
    args = parse_arguments()

    # Prepare chain filter set
    chain_set = None
    if args.chains:
        # Allow a single space to represent a blank chain ID
        chain_set = {ch if ch != "" else " " for ch in args.chains}

    split_pdb(args.input, args.output_dir, chain_set)

if __name__ == "__main__":
    main()