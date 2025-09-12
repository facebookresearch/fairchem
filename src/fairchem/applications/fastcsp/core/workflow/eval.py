"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Crystal Structure Evaluation Module

This module provides functionality for evaluating predicted crystal structures against
experimental references using two methods:
1. Cambridge Structural Database (CSD) and their Python API for packing similarity
2. Pymatgen StructureMatcher for crystallographic comparison

The evaluation uses:
- CSD: packing similarity metrics with increasing shell sizes (RMSD15, RMSD20, RMSD30)
- Pymatgen: crystallographic structure matching with lattice and site tolerances

The evaluation workflow:
1. Parse predicted crystal structures from CIF format
2. Load experimental reference structures from CSD or CIF files
3. Compute similarity using either CSD packing similarity or pymatgen StructureMatcher
4. Filter and rank matches based on similarity thresholds
5. Generate comprehensive evaluation reports

Requires:
- CSD Python API license for CSD-based comparisons (optional)
- Pymatgen for crystallographic structure matching
- Target crystal structures in CIF format
- Swifter for parallel DataFrame processing
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import swifter
from p_tqdm import p_map
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure

# Default directory containing experimental crystal structures
TARGET_XTALS_DIR = Path(
    "/private/home/anuroops/workspace/chemistry/fairchem-cmu-csp/data/CSP/target_xtals"
)

# Configure swifter for parallel processing of structure comparisons
swifter.set_defaults(
    npartitions=1000,  # Number of data partitions for parallel processing
    dask_threshold=1,  # Minimum data size to trigger Dask usage
    scheduler="processes",  # Use multiprocessing for CPU-intensive comparisons
    progress_bar=True,  # Show progress during evaluation
    progress_bar_desc="Evaluating",
    allow_dask_on_strings=False,  # Disable Dask for string operations
    force_parallel=False,  # Allow automatic parallel/serial decision
)


def ccdc_match_settings(shell_size=30, ignore_H=True, mol_diff=False):
    """
    Configure CCDC PackingSimilarity settings for crystal structure comparison.

    Sets up the parameters for comparing crystal structures using the CCDC
    packing similarity algorithm, which evaluates structural similarity based
    on molecular packing arrangements within a specified coordination shell.

    Args:
        shell_size: Number of molecules to include in the packing shell analysis.
                   Larger values provide more comprehensive but slower comparisons.
        ignore_H: Whether to ignore hydrogen atom positions in the comparison.
        mol_diff: Whether to allow molecular differences during comparison.
                 False enforces exact molecular matching.

    Returns:
        PackingSimilarity: Configured CCDC PackingSimilarity object ready for
                          structure comparisons.

    Note:
        Distance and angle tolerances are automatically scaled based on shell_size
        to balance sensitivity with computational efficiency.
    """
    from ccdc.crystal import PackingSimilarity

    se = PackingSimilarity()
    # Configure packing shell parameters
    se.settings.packing_shell_size = shell_size
    se.settings.distance_tolerance = (shell_size + 5) / 100  # Scale with shell size
    se.settings.angle_tolerance = shell_size + 5  # Degrees
    se.settings.ignore_hydrogen_positions = ignore_H
    se.settings.allow_molecular_differences = mol_diff
    return se


def match(row, target_xtals, shell_size=30):
    """
    Compare a single predicted crystal structure against experimental references.

    Evaluates whether a predicted crystal structure matches any of the provided
    experimental reference structures using CCDC packing similarity analysis.
    Returns the best match if similarity criteria are met.

    Args:
        row: DataFrame row containing structure data with 'relaxed_cif' column
             containing the CIF string of the predicted structure
        target_xtals: Dictionary mapping reference codes to CCDC Crystal objects
                     containing experimental structures for comparison
        shell_size: Size of the molecular coordination shell for packing analysis

    Returns:
        tuple: (refcode, rmsd) where:
               - refcode: Reference code of the best matching experimental structure,
                         or None if no match found
               - rmsd: Root mean square deviation of the match in Angstroms,
                      or None if no match found

    Note:
        A match is considered valid only if the number of matched molecules
        equals or exceeds the specified shell_size, ensuring structural
        similarity across the entire coordination environment.
    """
    from ccdc.crystal import Crystal

    # Parse the predicted crystal structure from CIF format
    try:
        gen_xtal = Crystal.from_string(row.relaxed_cif, "cif")
    except Exception as e:
        print(f"Error parsing {row.relaxed_cif}: {e}")
        return None, None

    # Configure packing similarity evaluation
    se = ccdc_match_settings(shell_size=shell_size)

    # Compare against all experimental reference structures
    for refcode, target_xtal in target_xtals.items():
        results = se.compare(gen_xtal, target_xtal)
        # Check if match meets minimum shell size requirement
        if results is not None and results.nmatched_molecules >= shell_size:
            print(
                f"Matched[{shell_size}] {row.structure_id} | {refcode}: {results.rmsd}",
                flush=True,
            )
            return refcode, results.rmsd
    return None, None


def ccdc_match_file(
    generated_xtals_path: Path,
    refcodes: list[str],
    output_path: Path,
    target_xtals_dir: Path = TARGET_XTALS_DIR,
):
    """
    Evaluate all predicted crystal structures in a file against experimental references.

    Performs comprehensive structure matching for all predicted structures in a
    Parquet file using a hierarchical approach with increasing shell sizes
    (RMSD15 → RMSD20 → RMSD30). Only structures passing lower thresholds are
    evaluated at higher levels for computational efficiency.

    Args:
        generated_xtals_path: Path to Parquet file containing predicted structures
                             with 'relaxed_cif' and energy columns
        refcodes: List of CSD reference codes for experimental structures to match against
        output_path: Directory to save evaluation results
        target_xtals_dir: Directory containing experimental CIF files

    Workflow:
        1. Load experimental reference structures from CIF files
        2. Sort predicted structures by energy (lowest first)
        3. Evaluate RMSD15 matches for all structures
        4. For RMSD15 matches, compute RMSD20
        5. For RMSD20 matches, compute RMSD30
        6. Save comprehensive results with match information

    Output:
        Parquet file with original structure data plus match columns:
        - match15, rmsd15: RMSD15 evaluation results
        - match20, rmsd20: RMSD20 evaluation results (if RMSD15 matched)
        - match30, rmsd30: RMSD30 evaluation results (if RMSD20 matched)

    Note:
        Hierarchical evaluation significantly reduces computational time by only
        performing expensive high-shell comparisons on promising candidates.
    """
    from ccdc.crystal import Crystal

    # Check if results already exist to avoid recomputation
    outfile = output_path / f"{generated_xtals_path.stem}.parquet"
    if outfile.exists():
        print(f"Skipping {generated_xtals_path} because it already exists")
        return

    print(f"Evaluating {generated_xtals_path}")

    # Load experimental reference structures
    target_xtals = {
        refcode: Crystal.from_string(
            (target_xtals_dir / f"{refcode}.cif").read_text(), "cif"
        )
        for refcode in refcodes.split(",")
    }
    print("Target xtals:", target_xtals.keys())

    # Load and sort predicted structures by energy
    df = pd.read_parquet(generated_xtals_path, engine="pyarrow")
    df.sort_values(by="energy_relaxed_per_molecule", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("Number of structures:", df.shape[0])

    # Level 1: RMSD15 evaluation for all structures
    results15 = df.swifter.apply(
        lambda row: match(row, target_xtals, shell_size=15),
        axis=1,
        result_type="expand",
    )
    df[["match15", "rmsd15"]] = results15

    # Level 2: RMSD20 evaluation only for RMSD15 matches
    df15 = df[df["match15"].notna()]
    print("df15 shape:", df15.shape)
    if df15.shape[0] > 0:
        results20 = df15.swifter.apply(
            lambda row: match(row, target_xtals, shell_size=20),
            axis=1,
            result_type="expand",
        )
        results20.columns = ["match20", "rmsd20"]
        df.loc[df15.index, ["match20", "rmsd20"]] = results20
    else:
        df[["match20", "rmsd20"]] = None

    # Level 3: RMSD30 evaluation only for RMSD20 matches
    df20 = df[df["match20"].notna()]
    print("df20 shape:", df20.shape)
    if df20.shape[0] > 0:
        results30 = df20.swifter.apply(
            lambda row: match(row, target_xtals, shell_size=30),
            axis=1,
            result_type="expand",
        )
        results30.columns = ["match30", "rmsd30"]
        df.loc[df20.index, ["match30", "rmsd30"]] = results30
    else:
        df[["match30", "rmsd30"]] = None

    # Display summary of matches found
    print(df[["match20", "rmsd20", "match30", "rmsd30"]])

    # Save comprehensive evaluation results
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        outfile,
        engine="pyarrow",
        compression="zstd",
    )
    print(f"Saved file: {outfile}")


def pymatgen_match(
    row,
    target_structures: dict[str, Structure],
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5,
):
    """
    Compare a single predicted crystal structure against experimental references using pymatgen.

    Uses pymatgen StructureMatcher to evaluate whether a predicted crystal structure
    matches any of the provided experimental reference structures based on
    crystallographic comparison of lattice parameters and atomic positions.

    Args:
        row: DataFrame row containing structure data with 'relaxed_cif' column
             containing the CIF string of the predicted structure
        target_structures: Dictionary mapping reference codes to pymatgen Structure objects
                          containing experimental structures for comparison
        ltol: Lattice parameter tolerance (fractional difference)
        stol: Site tolerance (Ångström) for atomic position matching
        angle_tol: Angle tolerance (degrees) for lattice angle matching

    Returns:
        tuple: (refcode, rms_dist) where:
               - refcode: Reference code of the best matching experimental structure,
                         or None if no match found
               - rms_dist: RMS distance metric from StructureMatcher,
                          or None if no match found

    Note:
        Unlike CSD-based matching, pymatgen matching provides boolean results
        with RMS distance metrics for matched structures.
    """
    try:
        # Parse the predicted crystal structure from CIF format
        pred_structure = Structure.from_str(row.relaxed_cif, fmt="cif")
    except Exception as e:
        print(f"Error parsing structure {row.structure_id}: {e}")
        return None, None

    # Configure StructureMatcher with specified tolerances
    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)

    # Compare against all experimental reference structures
    best_match = None
    best_rms_dist = float("inf")

    for refcode, target_structure in target_structures.items():
        if matcher.fit(pred_structure, target_structure):
            # Get RMS distance for the match
            rms_dist = matcher.get_rms_dist(pred_structure, target_structure)[0]
            if rms_dist < best_rms_dist:
                best_match = refcode
                best_rms_dist = rms_dist

    if best_match is not None:
        print(
            f"Pymatgen Match: {row.structure_id} | {best_match}: {best_rms_dist:.4f}",
            flush=True,
        )
        return best_match, best_rms_dist

    return None, None


def load_target_structures_from_cifs(
    target_xtals_dir: Path, refcodes: list[str]
) -> dict[str, Structure]:
    """
    Load experimental reference structures from CIF files for pymatgen comparison.

    Loads crystal structures from CIF files in the target directory and converts them
    to pymatgen Structure objects for crystallographic comparison.

    Args:
        target_xtals_dir: Directory containing CIF files with experimental structures
        refcodes: List of reference codes to load (should match CIF filenames)

    Returns:
        Dictionary mapping reference codes to pymatgen Structure objects

    Note:
        - CIF files should be named as "{refcode}.cif"
        - Structures that fail to load are skipped with warning messages
    """
    target_structures = {}

    for refcode in refcodes:
        cif_file = target_xtals_dir / f"{refcode}.cif"
        if cif_file.exists():
            try:
                structure = Structure.from_file(str(cif_file))
                target_structures[refcode] = structure
                print(f"Loaded reference structure: {refcode}")
            except Exception as e:
                print(f"Warning: Could not load {refcode}.cif: {e}")
        else:
            print(f"Warning: CIF file not found: {cif_file}")

    return target_structures


def pymatgen_match_file(
    generated_xtals_path: Path,
    refcodes: list[str],
    output_path: Path,
    match_params: dict[str, float],
    target_xtals_dir: Path = TARGET_XTALS_DIR,
):
    """
    Evaluate all predicted crystal structures in a file using pymatgen StructureMatcher.

    Performs crystallographic structure matching for all predicted structures in a
    Parquet file using pymatgen's StructureMatcher with specified tolerances.

    Args:
        generated_xtals_path: Path to Parquet file containing predicted structures
        refcodes: List of experimental reference codes for comparison
        output_path: Directory where evaluation results will be saved
        match_params: Dictionary containing matching tolerances:
                     - ltol: Lattice parameter tolerance
                     - stol: Site tolerance
                     - angle_tol: Angle tolerance
        target_xtals_dir: Directory containing experimental CIF files

    Output:
        Creates a Parquet file with original structure data plus:
        - pymatgen_match: Reference code of matched structure or None
        - pymatgen_rms_dist: RMS distance of the match or None

    Note:
        Uses swifter for parallel processing to efficiently handle large structure sets.
    """
    # Check if output already exists
    output_file = output_path / generated_xtals_path.name
    if output_file.exists():
        print(f"Output file already exists, skipping: {output_file}")
        return

    # Load predicted structures
    df = pd.read_parquet(generated_xtals_path, engine="pyarrow")

    # Load experimental reference structures
    target_structures = load_target_structures_from_cifs(target_xtals_dir, refcodes)

    if not target_structures:
        print(f"No reference structures loaded for {generated_xtals_path.name}")
        # Save original dataframe with null match columns
        df["pymatgen_match"] = None
        df["pymatgen_rms_dist"] = None
        output_path.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_file, engine="pyarrow")
        return

    # Perform pymatgen-based structure matching
    print(f"Running pymatgen matching for {generated_xtals_path.name}")
    results = df.swifter.apply(
        lambda row: pymatgen_match(
            row,
            target_structures,
            ltol=match_params["ltol"],
            stol=match_params["stol"],
            angle_tol=match_params["angle_tol"],
        ),
        axis=1,
    )

    # Extract match results
    df["pymatgen_match"] = results.apply(lambda x: x[0])
    df["pymatgen_rms_dist"] = results.apply(lambda x: x[1])

    # Save results
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, engine="pyarrow")
    print(f"Saved pymatgen evaluation results: {output_file}")


def compute_structure_matches(
    input_dir: Path,
    output_dir: Path,
    molecules_file: Path,
    config: Optional[dict[str, Any]] = None,
):
    """
    Orchestrate structure matching evaluation for all predicted crystal structures.

    Main function that coordinates the evaluation of predicted crystal structures
    against experimental references. Supports both CSD-based packing similarity
    and pymatgen-based crystallographic matching based on configuration.

    Args:
        root: Directory containing predicted structure files (Parquet format)
              Each file should contain structures for a single molecule
        output_path: Directory to save evaluation results
        molecules_file: CSV file mapping molecule names to CSD reference codes
                      Expected columns: 'name', 'refcode'
        config: Optional configuration dictionary containing:
               - evaluation_method: 'csd' (default) or 'pymatgen'
               - pymatgen_match_params: tolerances for pymatgen matching
                 (ltol, stol, angle_tol)

    Workflow:
        1. Discover all structure files in the root directory
        2. Load molecule name to reference code mapping
        3. Determine evaluation method from config (CSD vs pymatgen)
        4. Launch parallel evaluation jobs for all molecule files
        5. Save consolidated evaluation results

    Output:
        For each input file, creates a corresponding Parquet file with:
        - Original structure data (energies, coordinates, etc.)
        - Match results based on selected method:
          * CSD: match15/20/30 with RMSD values
          * Pymatgen: pymatgen_match with RMS distance

    Note:
        - Uses parallel processing (p_map) to efficiently handle large datasets
        - Files with "bkp" in the name are automatically excluded from processing
        - Default method is CSD-based if no config provided or method not specified
    """
    # Discover all structure files to evaluate
    paths = list(input_dir.iterdir())

    # Randomize processing order for better load balancing
    import random

    random.shuffle(paths)

    # Load molecule name to reference code mapping
    df = pd.read_csv(molecules_file)
    name_to_refcodes = dict(zip(df["name"], df["refcode"]))

    # Import parallel processing utility

    # Filter out backup files and prepare parallel evaluation
    paths = [path for path in paths if "bkp" not in path.name]
    refcodes = [name_to_refcodes[Path(path).stem] for path in paths]

    # Determine evaluation method from config
    evaluation_method = "csd"  # Default to CSD method
    if config and "evaluate" in config and "method" in config["evaluate"]:
        evaluation_method = config["evaluate"]["method"].lower()
    elif config and "evaluation_method" in config:
        # Backward compatibility
        evaluation_method = config["evaluation_method"].lower()

    if evaluation_method == "pymatgen":
        # Use pymatgen-based structure matching
        print("Using pymatgen StructureMatcher for structure evaluation")

        # Get matching parameters from config or use defaults
        match_params = {"ltol": 0.2, "stol": 0.3, "angle_tol": 5}
        if (
            config
            and "evaluate" in config
            and "pymatgen_match_params" in config["evaluate"]
        ):
            match_params.update(config["evaluate"]["pymatgen_match_params"])
        elif config and "pymatgen_match_params" in config:
            # Backward compatibility
            match_params.update(config["pymatgen_match_params"])

        print(f"Pymatgen matching parameters: {match_params}")

        # Execute parallel pymatgen evaluation
        p_map(
            pymatgen_match_file,
            paths,
            refcodes,
            [output_dir] * len(paths),
            [match_params] * len(paths),
            num_cpus=30,
        )
    else:
        # Use CSD-based packing similarity (default method)
        print("Using CSD packing similarity for structure evaluation")

        # Execute parallel CSD evaluation
        p_map(ccdc_match_file, paths, refcodes, [output_dir] * len(paths), num_cpus=30)


if __name__ == "__main__":
    """
    Example usage for structure evaluation.

    This example demonstrates how to run structure matching evaluation
    on a specific dataset with filtered structures from a relaxation run.
    """
    # Define paths for a specific evaluation run
    root = Path("./").resolve()

    # Execute structure matching evaluation
    compute_structure_matches(
        root=root / "filtered_structures",  # Directory with predicted structures
        output_path=root
        / "matched_structures",  # Output directory for evaluation results
        molecules_file=Path(
            "configs/example_systems.csv"
        ),  # Molecule to refcode mapping
    )
