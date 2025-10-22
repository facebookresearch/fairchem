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
from fairchem.applications.fastcsp.core.utils.slurm import (
    get_eval_slurm_config,
    submit_slurm_jobs,
)
from p_tqdm import p_map
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure


def get_eval_config_and_method(
    config: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], str]:
    """Extract evaluation configuration from config dictionary."""
    logger = get_central_logger()
    eval_config = config.get("evaluate", {})
    eval_method = eval_config.get("method", "csd").lower()

    if eval_method not in ["csd", "pymatgen"]:
        logger.error(f"Invalid evaluation method '{eval_method}' specified.")
        raise ValueError("Evaluation method must be 'csd' or 'pymatgen'.")

    if eval_method == "csd":
        csd_config = eval_config.get("csd", {})
        # TODO: Use CSD_PYTHON_CMD executable with subprocess
        eval_config["csd_python_cmd"] = csd_config.get("python_cmd", "python")
        eval_config["num_cpus"] = csd_config.get("num_cpus", 1)
    elif eval_method == "pymatgen":
        pmg_config = eval_config.get("pymatgen", {}).get("match_params", {})
        eval_config["pymatgen_match_params"] = {
            "ltol": pmg_config.get("ltol", 0.1),
            "stol": pmg_config.get("stol", 0.1),
            "angle_tol": pmg_config.get("angle_tol", 5.0),
        }

    return eval_config, eval_method


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


def match_structures(row, target_structures, eval_method="csd", **kwargs):
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

    if eval_method == "csd":
        return _match_csd(row, target_structures, logger, **kwargs)
    elif eval_method == "pymatgen":
        return _match_pymatgen(row, target_structures, logger, **kwargs)
    else:
        logger.error(f"Invalid evaluation method '{eval_method}' specified.")
        raise ValueError("Evaluation method must be 'csd' or 'pymatgen'.")


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

    matcher = ccdc_match_settings(shell_size=shell_size)
    best_match_refcode = None
    best_rmsd = float("inf")

    for refcode, target_xtal in target_xtals.items():
        try:
            results = matcher.compare(gen_xtal, target_xtal)
            if (
                results is not None
                and results.nmatched_molecules >= shell_size
                and results.rmsd < best_rmsd
            ):
                best_match_refcode = refcode
                best_rmsd = results.rmsd
        except Exception as e:
            logger.warning(f"Error matching {row.structure_id} against {refcode}: {e}")
            continue
    if best_match_refcode is not None:
        logger.info(
            f"CSD Best Match[{shell_size}] {row.structure_id} | {best_match_refcode}: {best_rmsd}"
        )
        return best_match_refcode, best_rmsd
    return None, None


def _match_pymatgen(row, target_xtals, logger, ltol=0.2, stol=0.3, angle_tol=5):
    """Pymatgen-specific matching logic."""
    try:
        pred_structure = Structure.from_str(row.relaxed_cif, fmt="cif")
    except Exception as e:
        logger.error(f"Error parsing pymatgen structure {row.structure_id}: {e}")
        return None, None

    matcher = StructureMatcher(
        ltol=ltol, stol=stol, angle_tol=angle_tol, ignored_species=["H"]
    )
    best_match_refcode = None
    best_rmsd = float("inf")

    for refcode, target_xtal in target_xtals.items():
        try:
            if matcher.fit(pred_structure, target_xtal):
                rms_dist = matcher.get_rms_dist(pred_structure, target_xtal)[0]
                if rms_dist < best_rmsd:
                    best_match_refcode = refcode
                    best_rmsd = rms_dist
        except Exception as e:
            logger.warning(f"Error matching {row.structure_id} against {refcode}: {e}")
            continue

    if best_match_refcode is not None:
        logger.info(
            f"Pymatgen Match: {row.structure_id} | {best_match_refcode}: {best_rmsd:.4f}"
        )
        return best_match_refcode, best_rmsd

    return None, None


def load_target_structures(
    molecules_file: str | Path,
    target_xtals_dir: Path | str | None = None,
    eval_method: str = "csd",
) -> tuple[dict[str, Any], list[list[str]]]:
    """
    Load experimental reference structures from CIF files.

    Supports two modes:
    1. cif_path column in molecules_file: Load from paths specified in the DataFrame
    2. target_xtals_dir: Load from a directory containing {refcode}.cif files

    Args:
        molecules_file: CSV file path with name, refcode columns, and optionally cif_path column
        target_xtals_dir: Directory containing experimental CIF files (optional if cif_path column exists)
        eval_method: Evaluation method ('csd' or 'pymatgen')

    Returns:
        tuple: (target_structures_dict, refcodes_list_per_molecule)
    """
    logger = get_central_logger()
    target_structures = {}

    # Load molecules DataFrame
    molecules_df = pd.read_csv(molecules_file)

    # Check if cif_path column exists for direct CIF path resolution
    use_cif_paths = (
        "cif_path" in molecules_df.columns and not molecules_df["cif_path"].isna().all()
    )

    if use_cif_paths:
        logger.info("Loading target structures from cif_path column of molecule data")
        refcodes_list = []

        for _, row in molecules_df.iterrows():
            refcodes = [r.strip() for r in row["refcode"].split(",")]
            cif_path = Path(row["cif_path"])

            refcodes_list.append(refcodes)

            if len(refcodes) == 1:
                refcode = refcodes[0]
                if cif_path.suffix != ".cif":
                    cif_path = cif_path / f"{refcode}.cif"
                    if not cif_path.exists():
                        logger.error(f"CIF file not found for {refcode}: {cif_path}")
                        continue

                try:
                    structure = _load_single_structure(cif_path, eval_method)
                    target_structures[refcode] = structure
                    logger.debug(f"Loaded structure for {refcode} from {cif_path}")
                except Exception as e:
                    logger.warning(
                        f"Could not load {eval_method} structure for {refcode} from {cif_path}: {e}"
                    )

            else:
                # Multiple refcodes: cif_path should be a directory with {refcode}.cif files
                if not cif_path.is_dir():
                    logger.error(
                        f"For multiple refcodes {refcodes}, cif_path should be a directory: {cif_path}"
                    )
                    continue

                for refcode in refcodes:
                    cif_file = cif_path / f"{refcode}.cif"
                    if not cif_file.exists():
                        logger.warning(f"CIF file not found: {cif_file}")
                        continue

                    try:
                        structure = _load_single_structure(cif_file, eval_method)
                        target_structures[refcode] = structure
                        logger.debug(f"Loaded structure for {refcode} from {cif_file}")
                    except Exception as e:
                        logger.warning(
                            f"Could not load {eval_method} structure for {refcode} from {cif_file}: {e}"
                        )

    else:
        # Use target_xtals_dir approach
        if target_xtals_dir is None:
            raise ValueError(
                "target_xtals_dir is required when cif_path column is not available"
            )

        logger.info(f"Loading target structures from directory: {target_xtals_dir}")
        target_xtals_path = Path(target_xtals_dir)

        # Get all unique refcodes from molecules_df
        all_refcodes = set()
        refcodes_list = []
        for _, row in molecules_df.iterrows():
            refcodes = [rc.strip() for rc in row["refcode"].split(",")]
            refcodes_list.append(refcodes)
            all_refcodes.update(refcodes)

        # Load structures from directory
        for refcode in all_refcodes:
            cif_filename = target_xtals_path / f"{refcode}.cif"
            if not cif_filename.exists():
                logger.warning(f"CIF file not found: {cif_filename}")
                continue

            try:
                structure = _load_single_structure(cif_filename, eval_method)
                target_structures[refcode] = structure
                logger.info(f"Loaded structure for {refcode} from {cif_filename}")
            except Exception as e:
                logger.warning(
                    f"Could not load {eval_method} structure for {refcode}: {e}"
                )

    return target_structures, refcodes_list


def _load_single_structure(cif_path: Path, eval_method: str):
    if eval_method == "csd":
        try:
            from ccdc.crystal import Crystal
        except ImportError as e:
            raise ImportError("CSD Python API required for CCDC matching.") from e
        return Crystal.from_string(cif_path.read_text(), "cif")
    elif eval_method == "pymatgen":
        return Structure.from_file(str(cif_path))
    else:
        raise ValueError("Evaluation method must be 'csd' or 'pymatgen'.")


def evaluate_structures_file(
    generated_xtals_path: Path,
    refcodes: list[str],
    output_dir: Path,
    target_structures: dict[str, Any],
    eval_method: str = "csd",
    **method_params,
):
    """
    Unified function to evaluate predicted crystal structures against experimental references.

    Args:
        generated_xtals_path: Path to Parquet file containing predicted structures
        refcodes: List of experimental reference codes for comparison
        output_dir: Directory to save evaluation results
        target_structures: Dictionary mapping refcodes to loaded structure objects
        eval_method: Evaluation method ('csd' or 'pymatgen')
        **method_params: Method-specific parameters
    """
    logger = get_central_logger()

    outfile = output_dir / generated_xtals_path.name

    if outfile.exists():
        logger.info(f"Output file already exists, skipping: {outfile}")
        return

    logger.info(f"Evaluating {generated_xtals_path.name}")

    # Filter target structures to only include the ones we need for this file
    filtered_target_structures = {
        refcode: target_structures[refcode]
        for refcode in refcodes
        if refcode in target_structures
    }

    if not filtered_target_structures:
        logger.warning(
            f"No reference structures available for {generated_xtals_path.name} with refcodes {refcodes}"
        )
        return

    filtered_df = pd.read_parquet(generated_xtals_path, engine="pyarrow")

    # Set up swifter for parallel processing
    swifter.set_defaults(
        npartitions=min(len(filtered_df), method_params.get("num_cpus", 1)),
        dask_threshold=1,
        scheduler="processes",
        progress_bar=True,
        progress_bar_desc="Evaluating",
        allow_dask_on_strings=False,
        force_parallel=False,
    )

    if eval_method == "csd":
        # CSD hierarchical evaluation (RMSD15 → RMSD20 → RMSD30)

        filtered_df = filtered_df.sort_values(by="energy_relaxed_per_molecule")
        filtered_df = filtered_df.reset_index(drop=True)
        logger.info(f"Number of structures: {filtered_df.shape[0]}")

        # Level 1: RMSD15 evaluation
        results15 = filtered_df.swifter.apply(
            lambda row: match_structures(
                row, filtered_target_structures, eval_method="csd", shell_size=15
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
                    row, filtered_target_structures, eval_method="csd", shell_size=20
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
                    row, filtered_target_structures, eval_method="csd", shell_size=30
                ),
                axis=1,
                result_type="expand",
            )
            results30.columns = ["match30", "rmsd30"]
            filtered_df.loc[df20.index, ["match30", "rmsd30"]] = results30
        else:
            filtered_df[["match30", "rmsd30"]] = None

        matches_summary = filtered_df[
            ["match15", "rmsd15", "match20", "rmsd20", "match30", "rmsd30"]
        ].dropna()
        if not matches_summary.empty:
            logger.info(f"Found {len(matches_summary)} structures with CSD matches")

    elif eval_method == "pymatgen":
        results = filtered_df.swifter.apply(
            lambda row: match_structures(
                row, filtered_target_structures, eval_method="pymatgen", **method_params
            ),
            axis=1,
        )
        filtered_df["pymatgen_match"] = results.apply(
            lambda x: x[0] if x is not None else None
        )
        filtered_df["pymatgen_rmsd"] = results.apply(
            lambda x: x[1] if x is not None else None
        )

        # Log summary of pymatgen matches
        matches_summary = filtered_df[["pymatgen_match", "pymatgen_rmsd"]].dropna()
        if not matches_summary.empty:
            logger.info(
                f"Found {len(matches_summary)} structures with pymatgen matches"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_df.to_parquet(outfile, engine="pyarrow", compression="zstd")

    logger.info(f"Saved {eval_method.upper()} evaluation results: {outfile}")


def compute_structure_matches(
    input_dir: Path,
    output_dir: Path,
    eval_method: str,
    eval_config: dict[str, Any],
    molecules_file: str | Path,
):
    """
    Structure matching evaluation for all predicted crystal structures.

    Args:
        input_dir: Directory containing predicted structure files (Parquet format)
        output_dir: Directory to save evaluation results
        eval_method: Evaluation method ('csd' or 'pymatgen')
        eval_config: Evaluation configuration dictionary (includes target_xtals_dir if needed)
        molecules_file: CSV file mapping molecule names to CSD reference codes

    Note:
        Uses the unified evaluate_structures_file function for both CSD and pymatgen methods.
        SLURM configuration uses default parameters for pymatgen execution.
        target_xtals_dir is retrieved from eval_config and is optional when cif_path column exists.
    """
    logger = get_central_logger()

    # Discover all structure files to evaluate
    parquet_files = [
        path
        for path in list(input_dir.iterdir())
        if "bkp" not in path.name or Path(path).suffix == ".parquet"
    ]
    random.shuffle(parquet_files)

    # Load target structures
    target_structures, refcodes_list = load_target_structures(
        molecules_file, eval_config.get("target_xtals_dir"), eval_method
    )

    # Create mapping from molecule name to refcodes for parquet file processing
    molecules_df = pd.read_csv(molecules_file)
    name_to_refcodes_list = dict(zip(molecules_df["name"], refcodes_list))
    refcodes_list = [name_to_refcodes_list[Path(path).stem] for path in parquet_files]

    if not target_structures:
        logger.error("No reference structures loaded - skipping evaluation")
        return None

    if eval_method == "csd":
        # CSD: Local CPU execution
        logger.info("Using CSD Python API for structure evaluation (local CPU)")

        args_list = []
        for i, parquet_file in enumerate(parquet_files):
            args_list.append(
                (
                    parquet_file,
                    refcodes_list[i],
                    output_dir,
                    target_structures,
                    eval_method,
                )
            )

        return p_map(
            lambda args: evaluate_structures_file(*args),
            args_list,
            num_cpus=eval_config["num_cpus"],
        )

    elif eval_method == "pymatgen":
        # Pymatgen: SLURM distributed execution
        logger.info("Using pymatgen StructureMatcher for structure evaluation")
        method_params = eval_config.get("pymatgen_match_params", {})
        logger.info(f"Pymatgen matching parameters: {method_params}")

        # Get SLURM configuration from eval_config
        slurm_params = get_eval_slurm_config(eval_config)

        job_args = []
        for i, parquet_file in enumerate(parquet_files):
            job_args.append(
                (
                    evaluate_structures_file,
                    (
                        parquet_file,
                        refcodes_list[i],
                        output_dir,
                        target_structures,
                        eval_method,
                    ),
                    method_params,
                )
            )

        return submit_slurm_jobs(
            job_args,
            output_dir=output_dir.parent / "slurm",
            **slurm_params,
        )
    else:
        logger.error(f"Invalid evaluation method '{eval_method}' specified.")
        raise ValueError("Evaluation method must be 'csd' or 'pymatgen'.")


if __name__ == "__main__":
    """Example usage for standalone structure evaluation."""
    import yaml

    root = Path("./").resolve()
    config_path = Path("configs/example_config.yaml")

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    eval_config, eval_method = get_eval_config_and_method(config)

    compute_structure_matches(
        input_dir=root / "filtered_structures",
        output_dir=root / "matched_structures",
        eval_method=eval_method,
        eval_config=eval_config,
        molecules_file=Path("configs/example_systems.csv"),
    )

    logger = get_central_logger()
    logger.info("Structure evaluation completed")
