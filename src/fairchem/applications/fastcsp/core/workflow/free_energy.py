"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import pandas as pd
from fairchem.applications.fastcsp.core.utils.logging import get_central_logger
from fairchem.applications.fastcsp.core.utils.slurm import (
    get_slurm_config,
    submit_slurm_jobs,
)
from fairchem.applications.fastcsp.core.utils.structure import cif_to_atoms
from fairchem.applications.fastcsp.core.workflow.relax import create_calculator

from fairchem.core.components.calculate.recipes.phonons import (
    calculate_vibrational_thermo,
)

# Subdirectory of the free-energy output directory where per-row partial
# results are persisted. Writing each row's result immediately after it is
# computed allows the workflow to resume from where it left off if some batch
# jobs fail, without recomputing structures that have already been processed.
_TEMP_RESULTS_SUBDIR = ".tmp_results"


def _row_result_path(temp_dir: Path, source_file: Path | str, row_index: int) -> Path:
    """Return the temp parquet path for a single (source_file, row_index) result."""
    source_file = Path(source_file)
    return temp_dir / source_file.stem / f"row_{row_index}.parquet"


def _save_row_result(
    temp_dir: Path,
    source_file: Path | str,
    row_index: int,
    result: dict[str, Any],
) -> None:
    """Atomically write a per-row free energy result to its temp parquet file.

    The result dict (with array-valued thermo entries) is written as a
    single-row DataFrame; the schema of the file therefore reflects exactly
    which keys were populated, so failed rows produce a parquet with only
    the ``source_file`` and ``row_index`` columns.
    """
    path = _row_result_path(temp_dir, source_file, row_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame([result]).to_parquet(tmp, engine="pyarrow", compression="zstd")
    tmp.replace(path)


def _load_row_result(path: Path) -> dict[str, Any]:
    """Load a single per-row temp parquet back into a result dict."""
    _df = pd.read_parquet(path, engine="pyarrow")
    return _df.iloc[0].to_dict()


def get_free_energy_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract free energy configuration from the workflow config.

    Args:
        config: Full workflow configuration dictionary

    Returns:
        Free energy parameters dictionary
    """
    fe_config = config.get("free_energy", {})
    return {
        "calculator": fe_config.get("calculator", "uma_sm_1p1_omc"),
        "quasiharmonic": fe_config.get("quasiharmonic", True),
        "atom_disp": fe_config.get("atom_disp", 0.01),
        "min_lengths": fe_config.get("min_lengths", 15.0),
        "t_step": fe_config.get("t_step", 10),
        "t_max": fe_config.get("t_max", 500),
        "t_min": fe_config.get("t_min", 0),
        "structures_per_job": fe_config.get("structures_per_job", 10),
        "match_only": fe_config.get("match_only", True),
        "energy_cutoff": fe_config.get("energy_cutoff", None),
        "max_structures": fe_config.get("max_structures", None),
        "compute_dos": fe_config.get("compute_dos", False),
        "input_directory": fe_config.get("input_directory", "filtered_structures"),
    }


def get_free_energy_slurm_config(
    fe_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Get SLURM configuration for free energy jobs.

    Args:
        fe_config: Free energy configuration dictionary.
            If None, returns default parameters.

    Returns:
        SLURM parameters for submit_slurm_jobs function
    """
    if fe_config is None:
        fe_config = {}

    full_config = {"free_energy": fe_config}
    return get_slurm_config(full_config, "free_energy", "submit_slurm_jobs")


def compute_free_energy_single(
    input_file: Path,
    row_index: int,
    fe_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute vibrational free energy for a single polymorph.

    Args:
        input_file: Path to input parquet file with relaxed structures
        row_index: Integer index of the row to process
        fe_config: Free energy configuration parameters

    Returns:
        Dictionary with free energy results, or empty dict on failure.
        Always includes 'source_file' and 'row_index' keys for reassembly.
    """
    logger = get_central_logger()

    structures_df = pd.read_parquet(input_file, engine="pyarrow")
    row = structures_df.iloc[row_index]

    result = {"source_file": str(input_file), "row_index": row_index}

    cif_col = "cif_relaxed" if "cif_relaxed" in structures_df.columns else "relaxed_cif"
    atoms = cif_to_atoms(row[cif_col])
    if atoms is None:
        logger.warning(
            f"Skipping structure at index {row_index} in {input_file}: "
            "could not parse relaxed cif"
        )
        return result

    calc = create_calculator(fe_config)
    atoms.calc = calc
    try:
        thermo = calculate_vibrational_thermo(
            atoms,
            quasiharmonic=fe_config.get("quasiharmonic", False),
            atom_disp=fe_config.get("atom_disp", 0.01),
            min_lengths=fe_config.get("min_lengths", 15.0),
            t_step=fe_config.get("t_step", 10),
            t_max=fe_config.get("t_max", 500),
            t_min=fe_config.get("t_min", 0),
            compute_dos=fe_config.get("compute_dos", False),
        )
        result.update(thermo)
    except Exception:
        logger.exception(
            f"Free energy calculation failed for index {row_index} in {input_file}"
        )

    return result


def compute_free_energy_batch(
    work_items: list[tuple[str, int]],
    fe_config: dict[str, Any],
    temp_dir: Path,
) -> list[dict[str, Any]]:
    """
    Compute vibrational free energies for a batch of structures in one job.

    The calculator is created once and reused across all structures in the
    batch to avoid repeated model loading overhead. Each row's result is
    persisted to ``temp_dir`` immediately after it is computed so that
    partial progress survives job-level failures and can be picked up on
    workflow restart.

    Args:
        work_items: List of (input_file_path, row_index) tuples
        fe_config: Free energy configuration parameters
        temp_dir: Directory where per-row partial results are written

    Returns:
        List of result dictionaries, one per work item.
    """
    logger = get_central_logger()
    calc = create_calculator(fe_config)

    results = []
    for input_file_str, row_index in work_items:
        input_file = Path(input_file_str)
        result = {"source_file": input_file_str, "row_index": row_index}

        structures_df = pd.read_parquet(input_file, engine="pyarrow")
        row = structures_df.iloc[row_index]

        cif_col = (
            "cif_relaxed" if "cif_relaxed" in structures_df.columns else "relaxed_cif"
        )
        atoms = cif_to_atoms(row[cif_col])
        if atoms is None:
            logger.warning(
                f"Skipping structure at index {row_index} in {input_file}: "
                "could not parse relaxed cif"
            )
        else:
            atoms.calc = calc
            try:
                thermo = calculate_vibrational_thermo(
                    atoms,
                    quasiharmonic=fe_config.get("quasiharmonic", False),
                    atom_disp=fe_config.get("atom_disp", 0.01),
                    min_lengths=fe_config.get("min_lengths", 15.0),
                    t_step=fe_config.get("t_step", 10),
                    t_max=fe_config.get("t_max", 500),
                    t_min=fe_config.get("t_min", 0),
                    compute_dos=fe_config.get("compute_dos", False),
                )
                result.update(thermo)
            except Exception:
                logger.exception(
                    f"Free energy calculation failed for index {row_index} "
                    f"in {input_file}"
                )

        _save_row_result(temp_dir, input_file, row_index, result)
        results.append(result)

    return results


def collect_free_energy_results(
    jobs: list,
    input_dir: Path,
    output_dir: Path,
    fe_config: dict[str, Any],
) -> None:
    """
    Aggregate per-row partial results into final per-source parquet files.

    Results are read from the temp parquet files written incrementally by
    :func:`compute_free_energy_batch`, not from ``job.result()``. This means
    that partial progress is preserved when individual jobs fail: rows that
    did complete still appear under ``output_dir/.tmp_results`` and a
    workflow restart will only resubmit the missing rows.

    A source's final parquet is only written (and its temp files cleaned up)
    once all of its expected rows have a temp result. Sources with missing
    rows are left in the partial state so the next run can finish them.
    Job-level failures are logged but do not abort collection.

    The final parquets contain only the rows that passed
    :func:`filter_structure_indices` (i.e. the same subset that was scheduled
    for free-energy computation), with a fresh ``RangeIndex``. Rows that were
    filtered out at submission time are not carried through to the output.

    Args:
        jobs: List of completed submitit Job objects (used only to surface
            any job-level failures in the logs)
        input_dir: Directory containing the input parquet files; used to
            determine the expected set of row indices per source
        output_dir: Directory for output parquet files
        fe_config: Free energy configuration parameters; the same filtering
            options used by :func:`compute_free_energies` are applied here
            to determine the expected row indices per source
    """
    logger = get_central_logger()
    temp_dir = output_dir / _TEMP_RESULTS_SUBDIR

    # Surface job-level failures, but rely on temp files for the actual data
    # so a single failed job does not block aggregation of the rest.
    failed_jobs = 0
    for job in jobs:
        try:
            job.result()
        except Exception:
            failed_jobs += 1
            logger.exception(f"Free energy job {job.job_id} failed")
    if failed_jobs:
        logger.warning(
            f"{failed_jobs}/{len(jobs)} free energy jobs failed; "
            "aggregating results from temp files for completed structures"
        )

    incomplete_sources: list[str] = []

    for parquet_file in sorted(input_dir.iterdir()):
        if not parquet_file.name.endswith(".parquet"):
            continue

        output_file = output_dir / parquet_file.name
        if output_file.exists():
            continue

        structures_df = pd.read_parquet(parquet_file, engine="pyarrow")
        expected_indices = filter_structure_indices(structures_df, fe_config)

        # Check that every expected row has a temp result before finalizing
        missing = [
            idx
            for idx in expected_indices
            if not _row_result_path(temp_dir, parquet_file, idx).exists()
        ]
        if missing:
            incomplete_sources.append(parquet_file.name)
            logger.warning(
                f"{parquet_file.name}: {len(missing)}/{len(expected_indices)} "
                "rows missing temp results; leaving partial state for next run"
            )
            continue

        results: list[dict[str, Any]] = [
            _load_row_result(_row_result_path(temp_dir, parquet_file, idx))
            for idx in expected_indices
        ]

        thermo_keys = {
            k for r in results for k in r if k not in ("row_index", "source_file")
        }
        for key in sorted(thermo_keys):
            structures_df[key] = None

        for r in results:
            idx = r["row_index"]
            for key in thermo_keys:
                if key in r:
                    # ``.at`` (not ``.loc``) is required when assigning an
                    # array-valued result to a single cell; ``.loc`` interprets
                    # an iterable RHS as a broadcast and raises
                    # ``ValueError: Must have equal len keys and value when
                    # setting with an iterable``.
                    structures_df.at[structures_df.index[idx], key] = (  # noqa: PD008
                        r[key]
                    )

        filtered_df = structures_df.iloc[expected_indices].reset_index(drop=True)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        filtered_df.to_parquet(output_file, engine="pyarrow", compression="zstd")
        logger.info(
            f"Wrote free energy results for {len(filtered_df)} structures "
            f"to {output_file}"
        )

        source_subdir = temp_dir / parquet_file.stem
        if source_subdir.exists():
            for tf in source_subdir.glob("row_*.parquet"):
                tf.unlink()
            with contextlib.suppress(OSError):
                source_subdir.rmdir()

    if incomplete_sources:
        logger.warning(
            f"{len(incomplete_sources)} source file(s) have incomplete free "
            f"energy results and were not finalized: {incomplete_sources}"
        )

    # Best-effort cleanup of the now-empty temp root
    if temp_dir.exists():
        with contextlib.suppress(OSError):
            temp_dir.rmdir()


def filter_structure_indices(
    structures_df: pd.DataFrame,
    fe_config: dict[str, Any],
) -> list[int]:
    """
    Return row indices to include based on filtering criteria.

    When multiple filters are active, they are intersected — a structure must
    satisfy all active criteria. If ``max_structures`` is set, the filtered
    results are further truncated to the N lowest-energy structures.

    Args:
        structures_df: DataFrame with columns "match" and
            "energy_relaxed_per_molecule"
        fe_config: Free energy config containing optional keys
            ``match_only``, ``energy_cutoff``, and ``max_structures``

    Returns:
        Sorted list of integer row indices to process.
    """
    match_only = fe_config.get("match_only", True)
    energy_cutoff = fe_config.get("energy_cutoff")
    max_structures = fe_config.get("max_structures")

    if not match_only and energy_cutoff is None and max_structures is None:
        return list(range(len(structures_df)))

    all_indices = set(range(len(structures_df)))
    masks = []

    if match_only:
        masks.append(set(structures_df.index[structures_df["match"].notna()].tolist()))

    if energy_cutoff is not None:
        min_energy = structures_df["energy_relaxed_per_molecule"].min()
        masks.append(
            set(
                structures_df.index[
                    structures_df["energy_relaxed_per_molecule"]
                    <= min_energy + energy_cutoff
                ].tolist()
            )
        )

    # Intersection of all active filters
    indices = all_indices
    for mask in masks:
        indices = indices & mask

    indices = sorted(indices)

    if max_structures is not None:
        energies = structures_df.loc[indices, "energy_relaxed_per_molecule"]
        indices = energies.nsmallest(max_structures).index.tolist()
        indices.sort()

    return indices


def compute_free_energies(
    input_dir: Path,
    output_dir: Path,
    fe_config: dict[str, Any],
) -> list:
    """
    Submit batched free energy jobs across all parquet files.

    Structures are grouped into batches of ``structures_per_job`` (default 10)
    so that each SLURM job processes multiple structures sequentially, reducing
    scheduling overhead for short-running calculations.

    Each row's result is persisted incrementally to a temp file so that work
    completed by previous (potentially failed) runs is not redone: rows whose
    temp result already exists under ``output_dir/.tmp_results`` are skipped
    when assembling the work list.

    Args:
        input_dir: Directory containing input parquet files
        output_dir: Directory for output parquet files
        fe_config: Free energy configuration parameters

    Returns:
        List of submitted SLURM jobs
    """
    logger = get_central_logger()
    slurm_params = get_free_energy_slurm_config(fe_config)
    structures_per_job = fe_config.get("structures_per_job", 10)
    temp_dir = output_dir / _TEMP_RESULTS_SUBDIR

    work_items: list[tuple[str, int]] = []
    skipped_existing = 0
    for parquet_file in sorted(input_dir.iterdir()):
        if not parquet_file.name.endswith(".parquet"):
            continue
        output_file = output_dir / parquet_file.name
        if output_file.exists():
            logger.info(f"Skipping {parquet_file.name}, already processed")
            continue

        structures_df = pd.read_parquet(parquet_file, engine="pyarrow")
        indices = filter_structure_indices(structures_df, fe_config)
        for row_index in indices:
            if _row_result_path(temp_dir, parquet_file, row_index).exists():
                skipped_existing += 1
                continue
            work_items.append((str(parquet_file), row_index))

    if skipped_existing:
        logger.info(
            f"Skipping {skipped_existing} structures with existing partial "
            f"results under {temp_dir}"
        )

    batches = [
        work_items[i : i + structures_per_job]
        for i in range(0, len(work_items), structures_per_job)
    ]

    job_args = [
        (compute_free_energy_batch, (batch, fe_config, temp_dir), {})
        for batch in batches
    ]

    logger.info(
        f"Submitting {len(job_args)} free energy jobs "
        f"({len(work_items)} structures, {structures_per_job} per job)"
    )
    return submit_slurm_jobs(
        job_args,
        output_dir=output_dir.parent / "slurm",
        **slurm_params,
    )
