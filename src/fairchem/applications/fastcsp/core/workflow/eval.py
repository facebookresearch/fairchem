"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Crystal Structure Evaluation Module

Evaluate predicted crystal structures against experimental references using:
- CSD Python API for packing similarity (local CPU execution, streaming writes)
- Pymatgen StructureMatcher (SLURM distributed execution)
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fairchem.applications.fastcsp.core.utils.logging import get_central_logger
from fairchem.applications.fastcsp.core.utils.slurm import (
    get_eval_slurm_config,
    submit_slurm_jobs,
)
from multiprocess import Pool
from p_tqdm import p_map
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen.*")


def get_eval_config_and_method(
    config: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], str, str]:
    """Extract evaluation configuration from config dictionary."""
    logger = get_central_logger()
    eval_config = config.get("evaluate", {})
    eval_method = eval_config.get("method").lower()

    if eval_method not in ["csd", "ccdc", "pymatgen", "pmg"]:
        logger.error(f"Invalid evaluation method '{eval_method}' specified.")
        raise ValueError(
            "Evaluation method must be 'csd', 'ccdc', 'pymatgen', or 'pmg'."
        )

    if eval_method in ["csd", "ccdc"]:
        eval_method = "csd"
        csd_config = eval_config.get(eval_method, {})
        eval_config["csd_python_cmd"] = csd_config.get("python_cmd", "python")
        eval_config["num_cpus"] = csd_config.get("num_cpus", 1)
        eval_dir_name = "matched_structures_csd"
    else:
        eval_method = "pymatgen"
        pmg_config = eval_config.get(eval_method, {}).get("match_params", {})
        eval_config["pymatgen_match_params"] = {
            "ltol": pmg_config.get("ltol", 0.2),
            "stol": pmg_config.get("stol", 0.3),
            "angle_tol": pmg_config.get("angle_tol", 5),
        }
        p = eval_config["pymatgen_match_params"]
        eval_dir_name = (
            f"matched_structures_pmg_l{p['ltol']}_s{p['stol']}_a{p['angle_tol']}"
        )

    return eval_config, eval_method, eval_dir_name


def load_target_structures(
    molecules_file: str | Path,
    target_xtals_dir: Path | str | None = None,
) -> tuple[dict[str, str], list[list[str]]]:
    """
    Load experimental reference structures from CIF files.

    Supports two modes:
      1. ``cif_path`` column in ``molecules_file``: load from explicit paths.
      2. ``target_xtals_dir``: search directory for ``{refcode}.cif`` files.

    Args:
        molecules_file: CSV with columns ``name``, ``refcode`` (comma-separated),
            and optionally ``cif_path``.
        target_xtals_dir: Directory containing reference CIFs (required if no
            ``cif_path`` column).

    Returns:
        (target_cifs, refcodes_list) where ``target_cifs`` maps refcode -> CIF
        text and ``refcodes_list`` aligns with the order of rows in
        ``molecules_file``.
    """
    logger = get_central_logger()
    molecules_df = pd.read_csv(molecules_file)
    target_cifs: dict[str, str] = {}
    refcodes_list: list[list[str]] = []

    use_cif_paths = (
        "cif_path" in molecules_df.columns and not molecules_df["cif_path"].isna().all()
    )

    if use_cif_paths:
        logger.info("Loading target structures from cif_path column of molecule data")
        for _, row in molecules_df.iterrows():
            refcodes = [r.strip() for r in row["refcode"].split(",")]
            refcodes_list.append(refcodes)
            cif_path = Path(row["cif_path"])
            for rc in refcodes:
                p = cif_path if cif_path.suffix == ".cif" else cif_path / f"{rc}.cif"
                if p.exists():
                    target_cifs[rc] = p.read_text()
                else:
                    logger.error(f"CIF file not found for {rc}: {p}")
    else:
        if target_xtals_dir is None:
            raise ValueError(
                "target_xtals_dir required when cif_path column is missing"
            )
        logger.info(f"Loading target structures from directory {target_xtals_dir}")
        all_cifs = list(Path(target_xtals_dir).rglob("*.cif"))
        for _, row in molecules_df.iterrows():
            refcodes = [r.strip() for r in row["refcode"].split(",")]
            refcodes_list.append(refcodes)
        all_refcodes = {rc for rcs in refcodes_list for rc in rcs}
        for rc in all_refcodes:
            hits = [p for p in all_cifs if p.stem == rc]
            if not hits:
                logger.error(f"CIF file not found for {rc} in {target_xtals_dir}")
                continue
            if len(hits) > 1:
                logger.warning(f"Multiple CIFs for {rc}; using {hits[0]}")
            target_cifs[rc] = hits[0].read_text()
            logger.info(f"Loaded {rc} from {hits[0]}")

    return target_cifs, refcodes_list


# Thread-pinning env for worker subprocesses (avoid BLAS oversubscription).
_THREAD_PIN_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


# Inline script run by the CCDC-env Python. Reads {rows, target_cifs} JSON on
# stdin and writes the 15->20->30 packing-similarity ladder results to stdout.
_CSD_CHUNK_SCRIPT = r"""
import sys, json
from ccdc.crystal import Crystal, PackingSimilarity

def make_matcher(sz):
    se = PackingSimilarity()
    se.settings.packing_shell_size = sz
    se.settings.distance_tolerance = (sz + 5) / 100
    se.settings.angle_tolerance = sz + 5
    se.settings.ignore_hydrogen_positions = True
    se.settings.allow_molecular_differences = False
    return se

def best_match(gen, targets, matcher, shell):
    best, best_rmsd = None, float('inf')
    for rc, t in targets.items():
        try:
            r = matcher.compare(gen, t)
            if r is not None and r.nmatched_molecules >= shell and r.rmsd < best_rmsd:
                best, best_rmsd = rc, r.rmsd
        except Exception:
            continue
    if best is None:
        return None, None
    return best, best_rmsd

payload = json.loads(sys.stdin.read())
target_xtals = {}
target_parse_errors = []
for rc, cif in payload["target_cifs"].items():
    try:
        target_xtals[rc] = Crystal.from_string(cif, "cif")
    except Exception as e:
        target_parse_errors.append((rc, repr(e)[:200]))

if target_parse_errors:
    sys.stderr.write(
        "target_cif_parse_error: %d/%d targets failed: %s\n"
        % (len(target_parse_errors),
           len(payload["target_cifs"]),
           "; ".join("%s=%s" % (rc, msg) for rc, msg in target_parse_errors))
    )

m15 = make_matcher(15)
m20 = make_matcher(20)
m30 = make_matcher(30)

out = []
parse_errors = []
for r in payload["rows"]:
    rec = {"__orig_idx": r["__orig_idx"],
           "match15": None, "rmsd15": None,
           "match20": None, "rmsd20": None,
           "match30": None, "rmsd30": None}
    try:
        gen = Crystal.from_string(r["cif_relaxed"], "cif")
    except Exception as e:
        parse_errors.append((r["__orig_idx"], repr(e)[:200]))
        out.append(rec)
        continue
    rec["match15"], rec["rmsd15"] = best_match(gen, target_xtals, m15, 15)
    if rec["match15"] is None:
        out.append(rec)
        continue
    rec["match20"], rec["rmsd20"] = best_match(gen, target_xtals, m20, 20)
    if rec["match20"] is None:
        out.append(rec)
        continue
    rec["match30"], rec["rmsd30"] = best_match(gen, target_xtals, m30, 30)
    out.append(rec)

if parse_errors:
    sys.stderr.write(
        "ccdc_parse_error: %d/%d rows failed; first: idx=%s %s\n"
        % (len(parse_errors), len(payload["rows"]),
           parse_errors[0][0], parse_errors[0][1])
    )

print(json.dumps(out))
"""


def _csd_subprocess_chunk_worker(args):
    """Run one chunk of rows via a CCDC-env Python subprocess.

    Returns ``(mol_name, chunk_id, DataFrame)`` with columns
    ``__orig_idx, match{15,20,30}, rmsd{15,20,30}``.
    """
    mol_name, chunk_id, chunk_df, target_cifs, csd_python_cmd, timeout = args
    cif_col = "cif_relaxed" if "cif_relaxed" in chunk_df.columns else "relaxed_cif"
    rows = [
        {"__orig_idx": int(r["__orig_idx"]), "cif_relaxed": r[cif_col]}
        for _, r in chunk_df.iterrows()
    ]
    env = {**os.environ, **_THREAD_PIN_ENV}
    res = subprocess.run(
        [csd_python_cmd, "-c", _CSD_CHUNK_SCRIPT],
        input=json.dumps({"rows": rows, "target_cifs": target_cifs}),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"[{mol_name} chunk {chunk_id}] rc={res.returncode}: {res.stderr[-2000:]}"
        )
    if res.stderr:
        get_central_logger().warning(
            f"[{mol_name} chunk {chunk_id}] stderr: {res.stderr[-1000:].strip()}"
        )
    return mol_name, chunk_id, pd.DataFrame.from_records(json.loads(res.stdout))


def _pmg_chunk_worker(args):
    """Run one chunk of rows in-process with pymatgen ``StructureMatcher``.

    Returns a DataFrame with columns ``__orig_idx, pymatgen_match, pymatgen_rmsd``.
    """
    chunk_df, target_cifs, match_params = args
    for k, v in _THREAD_PIN_ENV.items():
        os.environ.setdefault(k, v)

    targets: dict[str, Structure] = {}
    target_parse_errors: list[tuple[str, str]] = []
    for rc, cif in target_cifs.items():
        try:
            targets[rc] = Structure.from_str(cif, fmt="cif")
        except Exception as e:
            target_parse_errors.append((rc, repr(e)[:200]))
            continue
    if target_parse_errors:
        get_central_logger().warning(
            "pmg target_cif_parse_error: %d/%d targets failed: %s"
            % (
                len(target_parse_errors),
                len(target_cifs),
                "; ".join(f"{rc}={msg}" for rc, msg in target_parse_errors),
            )
        )

    ltol = match_params.get("ltol", 0.2)
    stol = match_params.get("stol", 0.3)
    angle_tol = match_params.get("angle_tol", 5)
    ignore_H = match_params.get("ignore_H", True)
    matcher = (
        StructureMatcher(
            ltol=ltol, stol=stol, angle_tol=angle_tol, ignored_species=["H"]
        )
        if ignore_H
        else StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    )

    cif_col = "cif_relaxed" if "cif_relaxed" in chunk_df.columns else "relaxed_cif"
    out = []
    parse_errors: list[tuple[int, str]] = []
    for _, r in chunk_df.iterrows():
        rec = {
            "__orig_idx": int(r["__orig_idx"]),
            "pymatgen_match": None,
            "pymatgen_rmsd": None,
        }
        try:
            pred = Structure.from_str(r[cif_col], fmt="cif")
        except Exception as e:
            parse_errors.append((rec["__orig_idx"], repr(e)[:200]))
            out.append(rec)
            continue
        best, best_rmsd = None, float("inf")
        for rc, t in targets.items():
            try:
                if matcher.fit(pred, t):
                    rms = matcher.get_rms_dist(pred, t)[0]
                    if rms < best_rmsd:
                        best, best_rmsd = rc, rms
            except Exception:
                continue
        if best is not None:
            rec["pymatgen_match"] = best
            rec["pymatgen_rmsd"] = best_rmsd
        out.append(rec)
    if parse_errors:
        get_central_logger().warning(
            "pymatgen_parse_error: %d/%d rows failed; first: idx=%s %s"
            % (len(parse_errors), len(chunk_df), parse_errors[0][0], parse_errors[0][1])
        )
    return pd.DataFrame.from_records(out)


def evaluate_pymatgen_file(
    parquet_file: Path,
    refcodes: list[str],
    output_dir: Path,
    target_cifs: dict[str, str],
    n_workers: int,
    match_params: dict[str, Any],
    target_rows_per_chunk: int = 500,
) -> Path | None:
    """Evaluate one parquet with pymatgen in-process. Intended to run inside a SLURM job."""

    logger = get_central_logger()
    outfile = output_dir / parquet_file.name
    if outfile.exists():
        logger.info(f"[skip] {parquet_file.name}: exists")
        return outfile

    filtered = {rc: target_cifs[rc] for rc in refcodes if rc in target_cifs}
    if not filtered:
        logger.warning(f"[skip] {parquet_file.name}: no reference CIFs")
        return None

    structures_df = pd.read_parquet(parquet_file, engine="pyarrow")
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(structures_df) == 0:
        structures_df.to_parquet(outfile, engine="pyarrow", compression="zstd")
        return outfile

    structures_df["__orig_idx"] = np.arange(len(structures_df), dtype=np.int64)
    structures_df_sorted = (
        structures_df.sort_values("energy_relaxed_per_molecule")
        if "energy_relaxed_per_molecule" in structures_df.columns
        else structures_df
    )
    n_chunks = max(
        n_workers,
        (len(structures_df) + target_rows_per_chunk - 1) // target_rows_per_chunk,
    )
    n_chunks = min(n_chunks, len(structures_df))
    chunks = [structures_df_sorted.iloc[i::n_chunks].copy() for i in range(n_chunks)]
    logger.info(
        f"{parquet_file.stem}: {len(structures_df)} rows -> {n_chunks} chunks; {n_workers} workers"
    )

    args_list = [(c, filtered, match_params) for c in chunks]
    chunk_results = p_map(
        _pmg_chunk_worker,
        args_list,
        num_cpus=n_workers,
        desc=f"PMG {parquet_file.stem}",
    )
    results_df = pd.concat(chunk_results, ignore_index=True)
    merged = (
        structures_df.merge(results_df, on="__orig_idx", how="left")
        .sort_values("__orig_idx")
        .drop(columns=["__orig_idx"])
        .reset_index(drop=True)
    )
    merged.to_parquet(outfile, engine="pyarrow", compression="zstd")
    n_matches = int(merged["pymatgen_match"].notna().sum())
    logger.info(f"{parquet_file.stem}: {n_matches} matches; wrote {outfile}")
    return outfile


def evaluate_csd_streaming(
    parquet_files: list[Path],
    refcodes_list: list[list[str]],
    output_dir: Path,
    target_cifs: dict[str, str],
    n_workers: int,
    csd_python_cmd: str,
    chunk_timeout: int | None = None,
    target_rows_per_chunk: int = 200,
) -> list[Path]:
    """CSD streaming evaluator: one global worker pool across all molecules.

    All molecules' chunks are submitted together, shuffled, and processed by a
    single ``multiprocess.Pool``. As soon as every chunk for a given molecule
    has returned, that molecule's merged parquet is flushed to disk. This
    eliminates tail-blocking (idle workers near the end of a large molecule).
    """

    logger = get_central_logger()
    mol_state: dict[str, dict] = {}
    all_tasks: list[tuple] = []

    for parquet_file, refcodes in zip(parquet_files, refcodes_list):
        mol_name = parquet_file.stem
        outfile = output_dir / parquet_file.name
        if outfile.exists():
            logger.info(f"[skip] {mol_name}: {outfile} exists")
            continue

        filtered = {rc: target_cifs[rc] for rc in refcodes if rc in target_cifs}
        if not filtered:
            logger.warning(f"[skip] {mol_name}: no reference CIFs ({refcodes})")
            continue

        structures_df = pd.read_parquet(parquet_file, engine="pyarrow")
        if len(structures_df) == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            structures_df.to_parquet(outfile, engine="pyarrow", compression="zstd")
            continue

        structures_df["__orig_idx"] = np.arange(len(structures_df), dtype=np.int64)
        structures_df_sorted = (
            structures_df.sort_values("energy_relaxed_per_molecule")
            if "energy_relaxed_per_molecule" in structures_df.columns
            else structures_df
        )
        n_chunks = max(
            1, (len(structures_df) + target_rows_per_chunk - 1) // target_rows_per_chunk
        )
        n_chunks = min(n_chunks, len(structures_df))
        chunks = [
            structures_df_sorted.iloc[i::n_chunks].copy() for i in range(n_chunks)
        ]

        mol_state[mol_name] = {
            "full_df": structures_df,
            "outfile": outfile,
            "expected": n_chunks,
            "received": 0,
            "results": [],
        }
        for cid, chunk in enumerate(chunks):
            all_tasks.append(
                (mol_name, cid, chunk, filtered, csd_python_cmd, chunk_timeout)
            )
        logger.info(
            f"[queued] {mol_name}: {len(structures_df)} rows / {n_chunks} chunks"
        )

    if not all_tasks:
        logger.info("Nothing to evaluate.")
        return []

    random.shuffle(all_tasks)
    n_mols = len(mol_state)
    logger.info(
        f"CSD streaming: {n_mols} molecules, {len(all_tasks)} chunks, "
        f"{n_workers} workers"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    completed: list[Path] = []
    pbar = tqdm(total=len(all_tasks), desc="CSD chunks", unit="chunk")
    try:
        with Pool(processes=n_workers) as pool:
            for mol_name, _cid, chunk_df in pool.imap_unordered(
                _csd_subprocess_chunk_worker, all_tasks, chunksize=1
            ):
                st = mol_state[mol_name]
                st["results"].append(chunk_df)
                st["received"] += 1
                pbar.update(1)
                pbar.set_postfix_str(
                    f"{mol_name} {st['received']}/{st['expected']} "
                    f"| mols {len(completed)}/{n_mols}"
                )
                if st["received"] == st["expected"]:
                    merged = (
                        st["full_df"]
                        .merge(
                            pd.concat(st["results"], ignore_index=True),
                            on="__orig_idx",
                            how="left",
                        )
                        .sort_values("__orig_idx")
                        .drop(columns=["__orig_idx"])
                        .reset_index(drop=True)
                    )
                    merged.to_parquet(
                        st["outfile"], engine="pyarrow", compression="zstd"
                    )
                    n_matches = int(merged["match15"].notna().sum())
                    logger.info(
                        f"[done] {mol_name}: {n_matches} matches; wrote {st['outfile']}"
                    )
                    completed.append(st["outfile"])
                    del mol_state[mol_name]
    finally:
        pbar.close()

    for m, st in mol_state.items():
        logger.error(
            f"[incomplete] {m}: {st['received']}/{st['expected']} chunks; not written"
        )
    return completed


def compute_structure_matches(
    input_dir: Path,
    output_dir: Path,
    eval_method: str,
    eval_config: dict[str, Any],
    molecules_file: str | Path,
):
    """Structure matching evaluation for all predicted crystal structures.

    Dispatch differs by method:
      * ``csd``: runs locally on the host node. All molecules' chunks share a
        single ``multiprocess.Pool`` (size = ``evaluate.csd.num_cpus``). Each
        molecule's parquet is flushed as soon as its chunks complete.
      * ``pymatgen``: submits one SLURM job per parquet via
        ``submit_slurm_jobs``. Inside each job, rows are chunked and processed
        in parallel with ``cpus_per_task`` workers (from
        ``evaluate.pymatgen.slurm.cpus_per_task``).

    Output parquets preserve original input row order.

    Args:
        input_dir: Directory containing predicted-structure parquets.
        output_dir: Directory to write evaluation results.
        eval_method: ``"csd"`` or ``"pymatgen"``.
        eval_config: Evaluation configuration dictionary (``config["evaluate"]``).
        molecules_file: CSV mapping molecule name -> refcode(s).
    """
    logger = get_central_logger()

    parquet_files = [p for p in input_dir.iterdir() if p.suffix == ".parquet"]
    random.shuffle(parquet_files)
    logger.info(
        f"Evaluating structure matches: method={eval_method}, "
        f"{len(parquet_files)} parquet files"
    )

    # ---- Load reference structures. -----------------------------------------
    target_cifs, refcodes_list = load_target_structures(
        molecules_file, eval_config.get("target_xtals_dir")
    )
    molecules_df = pd.read_csv(molecules_file)
    name_to_refcodes_list = dict(zip(molecules_df["name"], refcodes_list))
    refcodes_list = [name_to_refcodes_list[Path(p).stem] for p in parquet_files]

    if not target_cifs:
        logger.error("No reference structures loaded - skipping evaluation")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- CSD: local streaming pool. -----------------------------------------
    if eval_method == "csd":
        csd_cfg = eval_config.get("csd", {})
        csd_python_cmd = eval_config.get(
            "csd_python_cmd", csd_cfg.get("python_cmd", "python")
        )
        _ct = csd_cfg.get("chunk_timeout", None)
        chunk_timeout = int(_ct) if _ct else None
        sys_cpus = os.cpu_count() or 1
        num_cpus_config = int(
            eval_config.get("num_cpus", csd_cfg.get("num_cpus", sys_cpus))
        )
        n_workers = max(1, min(num_cpus_config, sys_cpus))
        logger.info(
            f"CSD local execution: {n_workers} workers "
            f"(config={num_cpus_config}, system={sys_cpus} cores)"
        )
        return evaluate_csd_streaming(
            parquet_files=parquet_files,
            refcodes_list=refcodes_list,
            output_dir=output_dir,
            target_cifs=target_cifs,
            n_workers=n_workers,
            csd_python_cmd=csd_python_cmd,
            chunk_timeout=chunk_timeout,
        )

    # ---- Pymatgen: one SLURM job per parquet. -------------------------------
    if eval_method != "pymatgen":
        raise ValueError("Evaluation method must be 'csd' or 'pymatgen'.")

    match_params = dict(
        eval_config.get(
            "pymatgen_match_params",
            eval_config.get("pymatgen", {}).get("match_params", {}),
        )
    )
    slurm_params = get_eval_slurm_config(eval_config.get("pymatgen", {}))
    cpus_per_task = int(slurm_params.get("cpus_per_task", 1))
    logger.info(
        f"Pymatgen SLURM dispatch: {len(parquet_files)} jobs, "
        f"cpus_per_task={cpus_per_task} (= workers per job); "
        f"match_params={match_params}; slurm_params={slurm_params}"
    )
    job_args = [
        (
            evaluate_pymatgen_file,
            (
                parquet_file,
                refcodes,
                output_dir,
                target_cifs,
                cpus_per_task,
                match_params,
            ),
            {},
        )
        for parquet_file, refcodes in zip(parquet_files, refcodes_list)
    ]
    return submit_slurm_jobs(
        job_args,
        output_dir=output_dir.parent / "slurm",
        **slurm_params,
    )


if __name__ == "__main__":
    """Example usage for standalone structure evaluation."""
    import yaml

    root = Path("./").resolve()
    config_path = Path("configs/example_config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    eval_config, eval_method, eval_dir_name = get_eval_config_and_method(cfg)
    compute_structure_matches(
        input_dir=root / "filtered_structures",
        output_dir=root / eval_dir_name,
        eval_method=eval_method,
        eval_config=eval_config,
        molecules_file=Path("configs/example_systems.csv"),
    )
