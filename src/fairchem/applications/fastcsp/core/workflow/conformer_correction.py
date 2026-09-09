"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Conformer-fragment energy corrections for ML-relaxed molecular crystals.

For each row we extract the ``z`` molecules from ``cif_relaxed``, run a
single-point energy with the ``original`` calculator (used for relax) and the
``corrector`` calculator, then apply::

    energy_corrected = energy_relaxed - sum_z E_original + sum_z E_corrector
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import submitit
from ase.units import eV, kJ, mol
from fairchem.applications.fastcsp.core.utils.logging import get_central_logger
from fairchem.applications.fastcsp.core.utils.slurm import (
    get_conformer_corrections_slurm_config,
)
from fairchem.applications.fastcsp.core.utils.structure import (
    cif_to_structure,
    extract_molecules,
)
from fairchem.applications.fastcsp.core.workflow.relax import (
    CHECKPOINTS,
    create_calculator,
    get_relax_config_and_dir,
)
from tqdm import tqdm

if TYPE_CHECKING:
    from pathlib import Path

    from ase import Atoms

EV_TO_KJ_PER_MOL = eV / (kJ / mol)

CIF_COL = "cif_relaxed"
Z_COL = "z"
ENERGY_COL = "energy_relaxed"
CONNECTIVITY_COL = "validity.connectivity_unchanged"
APPLIED_COL = "validity.conformer_corrections.applied"


def _is_already_corrected(parquet: Path) -> bool:
    """True iff ``parquet`` already carries the ``applied`` column."""
    try:
        return APPLIED_COL in pq.read_schema(parquet).names
    except Exception:  # - corrupt/empty parquet
        return False


def get_conformer_corrections_config_and_dirs(
    config: dict[str, Any], verbose: bool = False
) -> tuple[dict[str, Any], Path, Path | None]:
    """Return ``(cc_config, input_dir, output_dir)``; ``output_dir=None`` = in-place."""
    relax_params, relax_output_dir = get_relax_config_and_dir(config)
    cc = {
        k.replace("-", "_"): v
        for k, v in (config.get("conformer_corrections") or {}).items()
    }
    corrector = cc.get("corrector_calculator")
    if not corrector:
        raise ValueError("conformer_corrections.corrector_calculator is required")

    input_dir = relax_output_dir / "raw_structures"
    output_dir = (
        relax_output_dir / "raw_conformer_corrected_structures"
        if cc.get("separate_output", False)
        else None
    )
    cc_config = {
        "original_calculator": cc.get(
            "original_calculator", relax_params["calculator"]
        ),
        "corrector_calculator": corrector,
        "slurm": cc.get("slurm") or {},
    }
    if verbose:
        logger = get_central_logger()
        logger.info("Conformer-corrections configuration:")
        logger.info(f"  Original  calculator: {cc_config['original_calculator']}")
        logger.info(f"  Corrector calculator: {cc_config['corrector_calculator']}")
        logger.info(f"  Input  directory: {input_dir}")
        logger.info(f"  Output directory: {output_dir or '(in-place)'}")
    return cc_config, input_dir, output_dir


def _fragment_energies(atoms_list: Sequence[Atoms], calc, task_name: str) -> np.ndarray:
    """Per-fragment single-point energies (kJ/mol); failures return NaN."""
    logger = get_central_logger()
    out = np.full(len(atoms_list), np.nan, dtype=float)
    for k, atoms in enumerate(atoms_list):
        a = atoms.copy()
        if task_name == "omol":
            a.info.setdefault("spin", 1)
            a.info.setdefault("charge", 0)
        a.calc = calc
        try:
            out[k] = float(a.get_potential_energy()) * EV_TO_KJ_PER_MOL
        except Exception as exc:
            logger.warning(f"Single-point failed for fragment {k}: {exc}")
    return out


def apply_conformer_corrections(
    input_files: Sequence[Path],
    input_dir: Path,
    output_dir: Path | None,
    original_calc: str,
    corrector_calc: str,
) -> None:
    """Rewrite every parquet in ``input_files`` with correction columns.

    Loads each calculator once and iterates. When ``output_dir`` is ``None``,
    rewrites in place (and skips already-corrected files); otherwise mirrors
    the input tree under ``output_dir`` and skips files already written.
    """
    logger = get_central_logger()
    if not input_files:
        logger.info("No files to process for this rank")
        return

    logger.info(f"Loading original calculator: {original_calc}")
    orig = create_calculator({"calculator": original_calc})
    orig_task = CHECKPOINTS[original_calc]["task_name"]
    logger.info(f"Loading corrector calculator: {corrector_calc}")
    corr = create_calculator({"calculator": corrector_calc})
    corr_task = CHECKPOINTS[corrector_calc]["task_name"]

    n_skipped = 0
    for input_file in tqdm(input_files):
        if output_dir is None:
            if _is_already_corrected(input_file):
                n_skipped += 1
                continue
            output_file = input_file
        else:
            output_file = output_dir / input_file.relative_to(input_dir)
            if output_file.exists():
                n_skipped += 1
                continue

        input_df = pd.read_parquet(input_file)

        # Per-row fragment extraction. connectivity_unchanged from relax
        # already anchors the bond graph to the pre-relax topology, so we
        # only need to check that we recovered exactly z fragments.
        fragments: list[list[Atoms] | None] = []
        for _, row in input_df.iterrows():
            z = int(row.get(Z_COL, 0) or 0)
            cif = row.get(CIF_COL)
            if z <= 0 or not cif or not bool(row.get(CONNECTIVITY_COL, True)):
                fragments.append(None)
                continue
            try:
                atoms = extract_molecules(cif_to_structure(cif))
            except Exception as exc:
                logger.warning(
                    f"Extraction failed for {row.get('structure_id', '?')}: {exc}"
                )
                fragments.append(None)
                continue
            fragments.append(atoms if len(atoms) == z else None)

        input_df["correction.n_fragments"] = [
            0 if frags is None else len(frags) for frags in fragments
        ]

        # Flatten fragments across rows, run both single-point sweeps once,
        # then scatter-accumulate back to per-row totals.
        flat_atoms: list[Atoms] = []
        flat_owner: list[int] = []
        for ri, frags in enumerate(fragments):
            if frags is None:
                continue
            for a in frags:
                flat_atoms.append(a)
                flat_owner.append(ri)
        e_orig = _fragment_energies(flat_atoms, orig, orig_task)
        e_corr = _fragment_energies(flat_atoms, corr, corr_task)

        sum_orig = np.full(len(input_df), np.nan, dtype=float)
        sum_corr = np.full(len(input_df), np.nan, dtype=float)
        for k, ri in enumerate(flat_owner):
            if np.isnan(sum_orig[ri]):
                sum_orig[ri] = 0.0
                sum_corr[ri] = 0.0
            sum_orig[ri] += e_orig[k]
            sum_corr[ri] += e_corr[k]

        z_arr = input_df[Z_COL].astype(float).to_numpy()
        e_relaxed = (
            input_df[ENERGY_COL].astype(float).to_numpy()
            if ENERGY_COL in input_df.columns
            else np.full(len(input_df), np.nan)
        )
        input_df["correction.e_fragments_original"] = sum_orig
        input_df["correction.e_fragments_original_per_molecule"] = sum_orig / z_arr
        input_df["correction.e_fragments_corrector"] = sum_corr
        input_df["correction.e_fragments_corrector_per_molecule"] = sum_corr / z_arr
        input_df["energy_corrected"] = e_relaxed - sum_orig + sum_corr
        input_df["energy_corrected_per_molecule"] = input_df["energy_corrected"] / z_arr
        input_df[APPLIED_COL] = np.isfinite(sum_orig) & np.isfinite(sum_corr)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        input_df.to_parquet(output_file, compression="zstd")
        logger.debug(
            f"Wrote {len(input_df)} rows ({int(input_df[APPLIED_COL].sum())} corrected) "
            f"-> {output_file}"
        )

    if n_skipped:
        logger.info(f"Skipped {n_skipped}/{len(input_files)} already-corrected files")


def run_conformer_corrections_jobs(
    input_dir: Path,
    output_dir: Path | None,
    cc_config: dict[str, Any],
) -> list[submitit.Job]:
    """Submit per-rank conformer-corrections array jobs."""
    logger = get_central_logger()
    slurm_config, executor_params = get_conformer_corrections_slurm_config(cc_config)

    slurm_log_dir = (output_dir or input_dir).parent / "slurm_conformer_corrections"
    slurm_log_dir.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=slurm_log_dir)
    executor.update_parameters(**executor_params)

    all_files = sorted(input_dir.glob("**/*.parquet"))
    logger.info(f"Found {len(all_files)} input parquet files in {input_dir}")
    if output_dir is None:
        pending = [f for f in all_files if not _is_already_corrected(f)]
    else:
        pending = [
            f for f in all_files if not (output_dir / f.relative_to(input_dir)).exists()
        ]
    logger.info(f"Files needing correction: {len(pending)} / {len(all_files)}")
    if not pending:
        return []

    num_ranks = int(slurm_config.get("num_ranks", 1))
    jobs = []
    with executor.batch():
        jobs = [
            executor.submit(
                apply_conformer_corrections,
                pending[rank::num_ranks],
                input_dir,
                output_dir,
                cc_config["original_calculator"],
                cc_config["corrector_calculator"],
            )
            for rank in range(min(num_ranks, len(pending)))
        ]
    logger.info(
        f"Submitted {len(jobs)} conformer-corrections array jobs with job-id: "
        f"{jobs[0].job_id.split('_')[0] if jobs else ''}"
    )
    return jobs
