"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Crystal Structure Evaluation Module

Evaluate predicted crystal structures against experimental references using:
- CSD Python API for packing similarity (local CPU execution)
- Pymatgen StructureMatcher (SLURM distributed execution)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import swifter
from fairchem.applications.fastcsp.core.utils.logging import get_central_logger
from fairchem.applications.fastcsp.core.utils.slurm import submit_slurm_jobs
from p_tqdm import p_map
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure


def get_eval_config(config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Extract evaluation configuration from config dictionary."""
    logger = get_central_logger()
    eval_config = config.get("evaluate", {})
    eval_method = eval_config.get("method", "csd").lower()

    if eval_method not in ["csd", "pymatgen"]:
        logger.error(f"Invalid evaluation method '{eval_method}' specified.")
        raise ValueError("Evaluation method must be 'csd' or 'pymatgen'.")

    eval_config["method"] = eval_method

    if eval_method == "csd":
        csd_config = eval_config.get("csd", {})
        eval_config["csd_python_cmd"] = csd_config.get("python_cmd", "python")
        eval_config["num_cpus"] = csd_config.get("num_cpus", 1)
    elif eval_method == "pymatgen":
        pmg_params = eval_config.get("pymatgen_match_params", {})
        eval_config["pymatgen_match_params"] = {
            "ltol": pmg_params.get("ltol", 0.1),
            "stol": pmg_params.get("stol", 0.1),
            "angle_tol": pmg_params.get("angle_tol", 5.0),
        }

    return eval_config


def ccdc_match_settings(shell_size=30, ignore_H=True, mol_diff=False):
    """
    Configure CCDC settings for crystal structure comparison.

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
        Distance and angle tolerances are automatically scaled based on shell_size.
    """
    try:
        from ccdc.crystal import PackingSimilarity
    except ImportError as e:
        raise ImportError("CSD Python API required for CCDC matching.") from e

    se = PackingSimilarity()
    # Configure packing shell parameters
    se.settings.packing_shell_size = shell_size
    se.settings.distance_tolerance = (shell_size + 5) / 100
    se.settings.angle_tolerance = shell_size + 5
    se.settings.ignore_hydrogen_positions = ignore_H
    se.settings.allow_molecular_differences = mol_diff
    return se


def match_structures(row, target_structures, method="csd", **kwargs):
    """
    Compare a single predicted crystal structure against experimental references.

    Evaluates whether a predicted crystal structure matches any of the provided
    experimental reference structures using either CCDC packing similarity or
    pymatgen StructureMatcher.

    Args:
        row: DataFrame row containing structure data with 'relaxed_cif' column
        target_structures: Dictionary mapping reference codes to target structures
                          (CCDC Crystal objects for CSD, pymatgen Structure for pymatgen)
        method: Evaluation method ('csd' or 'pymatgen')
        **kwargs: Method-specific parameters
                 For CSD: shell_size (default 30)
                 For pymatgen: ltol, stol, angle_tol

    Returns:
        tuple: (refcode, metric) where:
               - refcode: Reference code of the best matching structure, or None
               - metric: RMSD for CSD or RMS distance for pymatgen, or None
    """
    logger = get_central_logger()

    if method == "csd":
        return _match_csd(row, target_structures, logger, **kwargs)
    elif method == "pymatgen":
        return _match_pymatgen(row, target_structures, logger, **kwargs)
    else:
        logger.error(f"Unknown matching method: {method}")
        return None, None


def _match_csd(row, target_xtals, logger, shell_size=30):
    """CSD-specific matching logic."""
    try:
        from ccdc.crystal import Crystal
    except ImportError as e:
        raise ImportError("CSD Python API required for CCDC matching.") from e

    try:
        gen_xtal = Crystal.from_string(row.relaxed_cif, "cif")
    except Exception as e:
        logger.error(f"Error parsing CSD structure {row.structure_id}: {e}")
        return None, None

    se = ccdc_match_settings(shell_size=shell_size)

    for refcode, target_xtal in target_xtals.items():
        results = se.compare(gen_xtal, target_xtal)
        if results is not None and results.nmatched_molecules >= shell_size:
            logger.info(
                f"CSD Match[{shell_size}] {row.structure_id} | {refcode}: {results.rmsd}"
            )
            return refcode, results.rmsd
    return None, None


def _match_pymatgen(row, target_structures, logger, ltol=0.2, stol=0.3, angle_tol=5):
    """Pymatgen-specific matching logic."""
    try:
        pred_structure = Structure.from_str(row.relaxed_cif, fmt="cif")
    except Exception as e:
        logger.error(f"Error parsing pymatgen structure {row.structure_id}: {e}")
        return None, None

    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    best_match = None
    best_rms_dist = float("inf")

    for refcode, target_structure in target_structures.items():
        if matcher.fit(pred_structure, target_structure):
            rms_dist = matcher.get_rms_dist(pred_structure, target_structure)[0]
            if rms_dist < best_rms_dist:
                best_match = refcode
                best_rms_dist = rms_dist

    if best_match is not None:
        logger.info(
            f"Pymatgen Match: {row.structure_id} | {best_match}: {best_rms_dist:.4f}"
        )
        return best_match, best_rms_dist

    return None, None


def load_target_structures(target_xtals_dir: Path, refcodes: list[str], method: str):
    """Load experimental reference structures from CIF files."""
    logger = get_central_logger()
    target_structures = {}

    if method == "csd":
        try:
            from ccdc.crystal import Crystal
        except ImportError as e:
            raise ImportError("CSD Python API required for CCDC matching.") from e

    for refcode in refcodes:
        cif_file = target_xtals_dir / f"{refcode}.cif"
        if not cif_file.exists():
            logger.warning(f"CIF file not found: {cif_file}")
            continue
        try:
            if method == "csd":
                structure = Crystal.from_string(cif_file.read_text(), "cif")
            elif method == "pymatgen":
                structure = Structure.from_file(str(cif_file))
            target_structures[refcode] = structure
        except Exception as e:
            logger.warning(f"Could not load {method} {refcode}.cif: {e}")

    return target_structures


def evaluate_structures_file(
    generated_xtals_path: Path,
    refcodes: list[str],
    output_path: Path,
    target_xtals_dir: Path,
    method: str = "csd",
    **method_params,
):
    """
    Unified function to evaluate predicted crystal structures against experimental references.

    Args:
        generated_xtals_path: Path to Parquet file containing predicted structures
        refcodes: List of experimental reference codes for comparison
        output_path: Directory to save evaluation results
        target_xtals_dir: Directory containing experimental CIF files
        method: Evaluation method ('csd' or 'pymatgen')
        **method_params: Method-specific parameters
    """
    logger = get_central_logger()

    swifter.set_defaults(
        npartitions=1000,
        dask_threshold=1,
        scheduler="processes",
        progress_bar=True,
        progress_bar_desc="Evaluating",
        allow_dask_on_strings=False,
        force_parallel=False,
    )

    if method == "csd":
        outfile = output_path / f"{generated_xtals_path.stem}.parquet"
    else:
        outfile = output_path / generated_xtals_path.name

    if outfile.exists():
        logger.info(f"Output file already exists, skipping: {outfile}")
        return

    logger.info(f"Evaluating {generated_xtals_path.name}")

    target_structures = load_target_structures(target_xtals_dir, refcodes, method)
    if not target_structures:
        logger.warning(
            f"No reference structures loaded for {generated_xtals_path.name}"
        )
        return

    filtered_df = pd.read_parquet(generated_xtals_path, engine="pyarrow")

    if method == "csd":
        # CSD hierarchical evaluation (RMSD15 → RMSD20 → RMSD30)
        filtered_df = filtered_df.sort_values(by="energy_relaxed_per_molecule")
        filtered_df = filtered_df.reset_index(drop=True)
        logger.info(f"Number of structures: {filtered_df.shape[0]}")

        # Level 1: RMSD15 evaluation
        results15 = filtered_df.swifter.apply(
            lambda row: match_structures(
                row, target_structures, method="csd", shell_size=15
            ),
            axis=1,
            result_type="expand",
        )
        filtered_df[["match15", "rmsd15"]] = results15

        # Level 2: RMSD20 evaluation only for RMSD15 matches
        df15 = filtered_df[filtered_df["match15"].notna()]
        if df15.shape[0] > 0:
            results20 = df15.swifter.apply(
                lambda row: match_structures(
                    row, target_structures, method="csd", shell_size=20
                ),
                axis=1,
                result_type="expand",
            )
            results20.columns = ["match20", "rmsd20"]
            filtered_df.loc[df15.index, ["match20", "rmsd20"]] = results20
        else:
            filtered_df[["match20", "rmsd20"]] = None

        # Level 3: RMSD30 evaluation only for RMSD20 matches
        df20 = filtered_df[filtered_df["match20"].notna()]
        if df20.shape[0] > 0:
            results30 = df20.swifter.apply(
                lambda row: match_structures(
                    row, target_structures, method="csd", shell_size=30
                ),
                axis=1,
                result_type="expand",
            )
            results30.columns = ["match30", "rmsd30"]
            filtered_df.loc[df20.index, ["match30", "rmsd30"]] = results30
        else:
            filtered_df[["match30", "rmsd30"]] = None

        matches_summary = filtered_df[
            ["match20", "rmsd20", "match30", "rmsd30"]
        ].dropna()
        if not matches_summary.empty:
            logger.info(f"Found {len(matches_summary)} structures with CSD matches")

    elif method == "pymatgen":
        results = filtered_df.swifter.apply(
            lambda row: match_structures(
                row, target_structures, method="pymatgen", **method_params
            ),
            axis=1,
        )
        filtered_df["pymatgen_match"] = results.apply(lambda x: x[0])
        filtered_df["pymatgen_rmsd"] = results.apply(lambda x: x[1])

    output_path.mkdir(parents=True, exist_ok=True)
    filtered_df.to_parquet(outfile, engine="pyarrow", compression="zstd")

    logger.info(f"Saved {method.upper()} evaluation results: {outfile}")


def compute_structure_matches(
    input_dir: Path,
    output_dir: Path,
    eval_config: dict[str, Any],
    molecules_file: str | Path,
    target_xtals_dir: Path,
):
    """
    Structure matching evaluation for all predicted crystal structures.

    Args:
        input_dir: Directory containing predicted structure files (Parquet format)
        output_dir: Directory to save evaluation results
        eval_config: Evaluation configuration dictionary
        molecules_file: CSV file mapping molecule names to CSD reference codes
        target_xtals_dir: Directory containing experimental CIF files

    Note:
        Uses the unified evaluate_structures_file function for both CSD and pymatgen methods.
    """
    logger = get_central_logger()

    # Discover all structure files to evaluate
    parquet_files = list(input_dir.iterdir())
    random.shuffle(parquet_files)

    # Load molecule name to CSD reference code mapping
    molecules_df = pd.read_csv(molecules_file)
    name_to_refcodes = dict(zip(molecules_df["name"], molecules_df["refcode"]))

    # Filter out backup files and prepare evaluation
    parquet_files = [path for path in parquet_files if "bkp" not in path.name]
    refcodes_list = [
        name_to_refcodes[Path(path).stem].split(",") for path in parquet_files
    ]

    # Determine evaluation method and parameters
    method = eval_config["method"]
    method_params = {}

    if eval_config["method"] == "pymatgen":
        logger.info("Using pymatgen StructureMatcher for structure evaluation")
        method_params = {
            "ltol": eval_config.get("pymatgen_match_params", {}).get("ltol", 0.2),
            "stol": eval_config.get("pymatgen_match_params", {}).get("stol", 0.3),
            "angle_tol": eval_config.get("pymatgen_match_params", {}).get(
                "angle_tol", 5
            ),
        }
        logger.info(f"Pymatgen matching parameters: {method_params}")
    elif eval_config["method"] == "csd":
        logger.info("Using CSD Python API for structure evaluation")

    if method == "csd":
        # CSD: Use p_map for local CPU execution
        logger.info("Using CSD Python API for structure evaluation (p_map)")

        args_list = []
        for i, parquet_file in enumerate(parquet_files):
            args_list.append(
                (parquet_file, refcodes_list[i], output_dir, target_xtals_dir, method)
            )

        p_map(
            lambda args: evaluate_structures_file(*args),
            args_list,
            num_cpus=eval_config.get("num_cpus", 1),
        )

    elif method == "pymatgen":
        # Pymatgen: Use SLURM distributed execution
        logger.info("Using pymatgen StructureMatcher for structure evaluation (SLURM)")

        # Get SLURM configuration
        slurm_params = eval_config.get("slurm", {})

        # Prepare job arguments for submitit
        job_args = []
        for i, parquet_file in enumerate(parquet_files):
            job_args.append(
                (
                    evaluate_structures_file,
                    (
                        parquet_file,
                        refcodes_list[i],
                        output_dir,
                        target_xtals_dir,
                        method,
                    ),
                    method_params,
                )
            )

        submit_slurm_jobs(
            job_args,
            output_dir=output_dir / "slurm",
            job_name="fastcsp_eval_pymatgen",
            **slurm_params,
        )


if __name__ == "__main__":
    """Example usage for structure evaluation."""
    import yaml

    root = Path("./").resolve()
    config_path = Path("configs/example_config.yaml")

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    eval_config = get_eval_config(config)

    compute_structure_matches(
        input_dir=root / "filtered_structures",
        output_dir=root / "matched_structures",
        eval_config=eval_config,
        molecules_file=Path("configs/example_systems.csv"),
        target_xtals_dir=root / "target_structures",
    )

    logger = get_central_logger()
    logger.info("Structure evaluation completed")
