"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Structure Deduplication Utilities for FastCSP
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd  # Noqa: TCH002
from fairchem.applications.fastcsp.core.utils.logging import get_central_logger
from fairchem.applications.fastcsp.core.utils.structure import get_structure_group
from p_tqdm import p_map
from pymatgen.analysis.structure_matcher import StructureMatcher
from tqdm import tqdm


def process_structure_group(
    group_data,
    ltol=0.2,
    stol=0.3,
    angle_tol=5,
    ignored_species: list[str] | None = ["H"],  # Noqa: B006
    pool=None,
    pbar=None,
    density_tol=None,
    energy_tol=None,
    apply_niggli_filter=False,
    scale: bool = True,
    primitive_cell: bool = True,
):
    # group_data = (indices, structures, energies). ``energies`` may be None
    # (only consulted when ``energy_tol`` is set).
    indices, structures, energies = group_data
    n = len(structures)
    if n <= 1:
        return [(indices[0], 0)] if n == 1 else []
    sm = StructureMatcher(
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
        ignored_species=list(ignored_species) if ignored_species else [],
        scale=scale,
        primitive_cell=primitive_cell,
    )
    # Cheap prefilters (computed once per bucket): if two structures disagree
    # on Niggli (a,b,c,alpha,beta,gamma) by more than (ltol, angle_tol), on
    # density by more than density_tol, or on energy by more than
    # energy_tol, sm.fit is essentially guaranteed False.
    if apply_niggli_filter:
        niggli = [s.lattice.get_niggli_reduced_lattice() for s in structures]
        abc = np.array([np.sort([L.a, L.b, L.c]) for L in niggli])
        ang = np.array([np.sort([L.alpha, L.beta, L.gamma]) for L in niggli])
    dens = np.array([s.density for s in structures])
    en = (
        np.asarray(energies, dtype=float)
        if (energy_tol is not None and energies is not None)
        else None
    )
    # Reorder so seeds are drawn closest-to-median-density first. This places
    # each seed's +-density_tol window over the dense region of any real
    # cluster, maximizing prefilter hit-rate.
    order = np.argsort(np.abs(dens - np.median(dens)))
    indices = [indices[k] for k in order]
    structures = [structures[k] for k in order]
    dens = dens[order]
    if apply_niggli_filter:
        abc = abc[order]
        ang = ang[order]
    if en is not None:
        en = en[order]
    used = [False] * n
    out = []
    subgroup = 0
    for i in range(n):
        if used[i]:
            continue
        used[i] = True
        ref = structures[i]
        out.append((indices[i], subgroup))
        # Heartbeat for the outer dedup bar (respects its mininterval).
        if pbar is not None:
            pbar.set_postfix_str(f"bkt={n} seed={subgroup} used={used.count(True)}/{n}")
        # Build the candidate list via cheap prefilters; only survivors get
        # the expensive sm.fit call.
        remaining = []
        for j in range(i + 1, n):
            if used[j]:
                continue
            if density_tol is not None and abs(dens[j] - dens[i]) > density_tol:
                continue
            if en is not None and abs(en[j] - en[i]) > energy_tol:
                continue
            if apply_niggli_filter:
                if not np.all(np.abs(abc[j] - abc[i]) / abc[i] < ltol):
                    continue
                if not np.all(np.abs(ang[j] - ang[i]) < angle_tol):
                    continue
            remaining.append(j)
        if pool is not None and len(remaining) > 10:
            # chunksize: amortize ~ms-scale pickle/dispatch over multiple
            # sm.fit calls per worker. Cap at 8 so the slowest chunk never
            # holds back the wall by too much (a chunk of 8 fits ~= 2s wall).
            is_dup_list = pool.starmap(
                sm.fit,
                [(ref, structures[j]) for j in remaining],
                chunksize=max(1, min(8, len(remaining) // pool._processes + 1)),
            )
            for j, is_dup in zip(remaining, is_dup_list):
                if is_dup:
                    used[j] = True
                    out.append((indices[j], subgroup))
        else:
            for j in remaining:
                if sm.fit(ref, structures[j]):
                    used[j] = True
                    out.append((indices[j], subgroup))
        subgroup += 1
    return out


def deduplicate_structures(
    structures_df: pd.DataFrame,
    structure_col: str = "structure",
    mol_id_col: str = "mol_id",
    conf_col: str | None = None,
    z_col: str | None = None,
    spg_col: str | None = None,
    density_col: str | None = None,
    density_bin_size: float | None = None,
    energy_col: str | None = None,
    energy_bin_size: float | None = None,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5,
    ignored_species: list[str] | None = ["H"],  # Noqa: B006
    density_tol: float | None = None,
    energy_tol: float | None = None,
    apply_niggli_filter: bool = False,
    scale: bool = True,
    primitive_cell: bool = True,
    keep: str | None = None,
    keep_col: str | None = None,
    n_jobs: int | None = None,
):
    """
    Two-stage deduplication: hash blocker + parallel StructureMatcher.

    The hash blocker is always (mol_id). Each optional ``*_col`` argument
    extends the blocker key when set:

    - ``z_col`: z number column
    - ``spg_col``: space-group column (e.g. ``"spg_generated"``)
    - ``conf_col``: conformer id column (e.g. ``"conf_id"``)
    - ``density_col`` + ``density_bin_size``: binned density
    - ``energy_col`` + ``energy_bin_size``: binned energy

    Representative selection (``keep`` + ``keep_col``):

    - ``keep=None``  -> keep all rows, just attach ``group_index``.
    - ``keep="min"``    -> keep row with the smallest ``keep_col`` per group
      (use for post-relax: ``keep_col="energy_relaxed_per_molecule"``).
    - ``keep="median"`` -> keep row whose ``keep_col`` value is closest to the
      group median (use for pre-relax: ``keep_col="density_generated"``).
    - ``keep="first"``  -> keep the first row per group (arbitrary).

    Cheap prefilters (skip ``sm.fit`` for obviously-different pairs):

    - ``density_tol`` (g/cc): drop pairs with ``|Δρ| > density_tol``. None = off.
    - ``energy_tol`` (same units as ``energy_col``): drop pairs with
      ``|ΔE| > energy_tol``. Requires ``energy_col`` to be set; None = off.
    - ``apply_niggli_filter`` (bool): when True, drop pairs whose
      Niggli-reduced (a,b,c,alpha,beta,gamma) disagree by more than
      (``ltol``, ``angle_tol``). Default False (skip the Niggli computation
      and prefilter entirely).

    StructureMatcher knobs (forwarded verbatim):

    - ``scale`` (bool, default True): when False, ``sm.fit`` will not
      rescale lattices to a common volume. Useful post-relax with
      ``bin_by_z=True`` (same Z -> same atom count, and relaxed volumes
      are physically meaningful so two structures at different volumes
      are not the same minimum). Slightly faster.
    - ``primitive_cell`` (bool, default True): when False, ``sm.fit`` will
      not reduce to primitive cells before matching. Safe (and a small
      speed win) when ``bin_by_z=True``, since every structure in a
      bucket already has the same atom count.
    """
    logger = get_central_logger()

    if n_jobs is None:
        n_jobs = max(len(os.sched_getaffinity(0)), 1)

    hash_density = density_col is not None and density_bin_size is not None
    hash_energy = energy_col is not None and energy_bin_size is not None

    logger.debug(
        f"Blocker - mol_id_col={mol_id_col}, z_col={z_col}, conf_col={conf_col}, "
        f"spg_col={spg_col}, density_col={density_col} (bin={density_bin_size}), "
        f"energy_col={energy_col} (bin={energy_bin_size})"
    )
    logger.debug(f"Total structures to process: {len(structures_df)}")

    # Hash keys via get_structure_group
    hash_cols = [mol_id_col]
    for c in (
        conf_col,
        z_col,
        spg_col,
        density_col if hash_density else None,
        energy_col if hash_energy else None,
    ):
        if c is not None:
            hash_cols.append(c)  # Noqa: PERF401

    def _row_to_hash(x):
        return get_structure_group(
            x[mol_id_col],
            conf_id=x[conf_col] if conf_col is not None else None,
            z=x[z_col] if z_col is not None else None,
            spg=x[spg_col] if spg_col is not None else None,
            density=x[density_col] if hash_density else None,
            density_bin_size=density_bin_size if hash_density else None,
            energy=x[energy_col] if hash_energy else None,
            energy_bin_size=energy_bin_size if hash_energy else None,
        )

    hashes = structures_df[hash_cols].apply(_row_to_hash, axis=1)

    # Group structures by hash for efficient pre-filtering
    hash_groups = defaultdict(list)
    for i, h in enumerate(hashes):
        hash_groups[h].append(i)
    # Process largest buckets first so the long-pole bucket starts early.
    hash_groups = sorted(hash_groups.items(), key=lambda kv: len(kv[1]))
    logger.info(f"Number of unique hashes: {len(hash_groups)}")
    string = "Top 10 largest hash buckets:"
    for i, (hash_val, indices) in enumerate(hash_groups[-10:]):
        string += f"  {i+1}. Hash: {hash_val}, Count: {len(indices)}"
    for i, (hash_val, indices) in enumerate(hash_groups[:10]):
        string += f"  {i+1}. Hash: {hash_val}, Count: {len(indices)}"
    logger.info(string)

    # Stage 2: Prepare data for parallel crystallographic comparison.
    # When ``energy_tol`` is set, pre-extract the energy column so the
    # worker can apply the cheap |ΔE| prefilter without touching the DataFrame.
    use_energy_prefilter = energy_tol is not None and energy_col is not None
    if energy_tol is not None and energy_col is None:
        logger.warning(
            "energy_tol is set but energy_col is None; energy prefilter "
            "will be disabled. Pass energy_col to enable it."
        )
    energies_arr = (
        structures_df[energy_col].to_numpy() if use_energy_prefilter else None
    )
    structures_arr = structures_df[structure_col].to_numpy()
    groups_to_process = [
        (
            indices,
            structures_arr[indices],
            energies_arr[indices] if use_energy_prefilter else None,
        )
        for _, indices in hash_groups
    ]

    # Stage 3: Parallel crystallographic deduplication within hash groups.
    num_groups = len(groups_to_process)
    logger.info(f"Processing {num_groups} hash groups in parallel...")
    worker = partial(
        process_structure_group,
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
        ignored_species=ignored_species,
        density_tol=density_tol,
        energy_tol=energy_tol if use_energy_prefilter else None,
        apply_niggli_filter=apply_niggli_filter,
        scale=scale,
        primitive_cell=primitive_cell,
    )
    biggest = max((len(g[0]) for g in groups_to_process), default=0)
    # Path A (inner-pool fan-out) only wins when a single bucket is large
    # enough that the per-seed survivors-after-prefilter (~10-15% of N for
    # pre-relax, much higher post-relax) regularly saturate n_jobs cores.
    # Empirical rule: bucket needs N >~ 8 * n_jobs. Below that, the inner pool
    # leaves most cores idle each seed; p_map across buckets is faster.
    use_inner_pool = biggest > 8 * n_jobs and len(groups_to_process) <= 2 * n_jobs
    if use_inner_pool:
        logger.info(
            f"{len(groups_to_process)} buckets, largest = {biggest}; using "
            f"parallel-greedy inner loop with a shared {n_jobs}-process pool."
        )
        with mp.get_context("fork").Pool(n_jobs) as pool:
            # Process largest buckets first so ETA stabilizes early.
            ordered = groups_to_process[::-1]
            pbar = tqdm(ordered, desc="dedup", mininterval=60)
            results = []
            for g in pbar:
                results.append(worker(g, pool=pool, pbar=pbar))  # Noqa: PERF401
            # Restore ascending-size order for downstream zip with hash_groups.
            results = results[::-1]
    else:
        logger.info(
            f"{len(groups_to_process)} buckets, largest = {biggest}; "
            f"using p_map across {n_jobs} workers (data-parallel buckets)."
        )
        results = p_map(worker, groups_to_process, num_cpus=n_jobs)

    # Stage 4: Combine results and assign global group indices
    all_matches = []
    for (hash_val, _), group_results in zip(hash_groups, results):
        for idx, subgroup in group_results:
            # Create globally unique group identifier
            all_matches.append((idx, f"{hash_val}_{subgroup}"))

    unique_groups = len({match[1] for match in all_matches})
    logger.info(
        f"Deduplication completed: {unique_groups} unique groups from {len(all_matches)} structures"
    )

    # Stage 5: Apply group assignments to DataFrame
    all_matches.sort(key=lambda x: x[0])  # Sort by original DataFrame index
    structures_df["group_index"] = [match[1] for match in all_matches]

    # Stage 6: Optional representative selection (one row per group_index)
    if keep is not None:
        if keep == "first":
            keep_idx = structures_df.drop_duplicates(subset=["group_index"]).index
        else:
            if keep_col is None:
                raise ValueError(f"keep={keep!r} requires keep_col to be set")
            grouped = structures_df.groupby("group_index")[keep_col]
            if keep == "min":
                keep_idx = grouped.idxmin()
            elif keep == "max":
                keep_idx = grouped.idxmax()
            elif keep == "median":
                # Row whose value is closest to its group's median
                median_per_group = grouped.transform("median")
                diff = (structures_df[keep_col] - median_per_group).abs()
                keep_idx = diff.groupby(structures_df["group_index"]).idxmin()
            else:
                raise ValueError(
                    f"keep must be one of None|'first'|'min'|'max'|'median', got {keep!r}"
                )
        logger.info(f"Selecting representatives: keep={keep!r}, keep_col={keep_col!r}")
        structures_df = structures_df.loc[keep_idx].reset_index(drop=True)
        logger.info(f"Structures after representative selection: {len(structures_df)}")
    return structures_df
