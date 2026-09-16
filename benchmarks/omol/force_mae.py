"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Evaluate force MAE on the public OMol25 validation set.
This script can be run in three modes:
- prepare: selects unique ωB97M-V structures and creates shards
- run: evaluates one model on a disjoint group of shards
- merge: combines metrics across all parts
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from ase import Atoms

N_STRUCTURES = 2_720_091
N_SHARDS = 800
MODELS = (
    "pet_omol_s",
    "pet_omol_m",
    "pet_omol_l",
    "orbmol_v2",
    "mace_mh_1",
    "uma_s",
    "uma_m",
    "mace_omol_0",
    "mace_polar_1_m",
)
SOURCE_COLUMNS = [
    "structure_hash",
    "method",
    "property_metadata",
    "atomic_forces",
    "cell",
    "positions",
    "pbc",
    "atomic_numbers",
]


def prepare(source_dir: Path, output_dir: Path) -> None:
    """Select ωB97M-V structures and split them into manageable files."""
    source_files = sorted(source_dir.glob("co_*.parquet"))
    if not source_files:
        raise FileNotFoundError(f"no co_*.parquet files in {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_per_shard = math.ceil(N_STRUCTURES / N_SHARDS)
    seen: set[str] = set()
    rows: list[dict] = []
    shards: list[str] = []

    def write_shard() -> None:
        nonlocal rows
        if rows:
            path = output_dir / f"shard{len(shards):03d}.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
            shards.append(path.name)
            rows = []

    for source_file in source_files:
        parquet = pq.ParquetFile(source_file)
        for batch in parquet.iter_batches(batch_size=4096, columns=SOURCE_COLUMNS):
            for row in batch.to_pylist():
                key = row["structure_hash"]
                if row["method"] != "ωB97M-V" or key in seen:
                    continue
                seen.add(key)
                metadata = json.loads(row.pop("property_metadata"))
                row.pop("method")
                row["charge"] = int(metadata["charge"])
                row["spin"] = int(metadata["spin"])
                rows.append(row)
                if len(rows) == rows_per_shard:
                    write_shard()
    write_shard()

    if len(seen) != N_STRUCTURES:
        raise RuntimeError(f"expected {N_STRUCTURES:,} structures, found {len(seen):,}")
    (output_dir / "manifest.json").write_text(
        json.dumps({"structures": len(seen), "shards": shards}, indent=2) + "\n"
    )


def make_atoms(row: dict) -> Atoms:
    charge, spin = int(row["charge"]), int(row["spin"])
    atoms = Atoms(
        numbers=row["atomic_numbers"],
        positions=row["positions"],
        cell=row["cell"],
        pbc=row["pbc"],
    )
    # Different calculators use different names for the same spin multiplicity.
    atoms.info.update(
        charge=charge,
        spin=spin,
        spin_multiplicity=spin,
        multiplicity=spin,
    )
    return atoms


def evaluate(
    model: str,
    data_dir: Path,
    checkpoint_dir: Path,
    part: int,
    parts: int,
) -> dict:
    """Evaluate one disjoint group of prepared parquet shards."""
    import torch
    from speed import make_calculator

    if not 0 <= part < parts:
        raise ValueError("part must be between zero and parts - 1")
    torch.set_float32_matmul_precision("high")
    manifest = json.loads((data_dir / "manifest.json").read_text())
    shard_names = manifest["shards"][part::parts]
    calculator = make_calculator(model, checkpoint_dir, for_speed=False)

    absolute_error = 0.0
    force_components = 0
    evaluated = 0
    skipped = 0
    for name in shard_names:
        for batch in pq.ParquetFile(data_dir / name).iter_batches(batch_size=128):
            for row in batch.to_pylist():
                if model == "mace_mh_1" and (row["charge"] != 0 or row["spin"] != 1):
                    skipped += 1
                    continue  # MACE-MH-1 (OMol head) only supports neutral singlets
                atoms = make_atoms(row)
                atoms.calc = calculator
                reference = np.asarray(row["atomic_forces"])
                prediction = atoms.get_forces()
                error = np.abs(prediction - reference)
                absolute_error += float(error.sum())
                force_components += error.size
                evaluated += 1

    return {
        "model": model,
        "part": part,
        "parts": parts,
        "evaluated_structures": evaluated,
        "skipped_structures": skipped,
        "absolute_error_sum": absolute_error,
        "force_components": force_components,
        "force_mae_meV_per_A": 1000 * absolute_error / force_components,
    }


def merge(files: list[Path]) -> dict:
    """Combine the additive statistics from all parts of one model."""
    parts = [json.loads(path.read_text()) for path in files]
    if len({part["model"] for part in parts}) != 1:
        raise ValueError("all result files must be for the same model")
    expected_parts = parts[0]["parts"]
    if sorted(part["part"] for part in parts) != list(range(expected_parts)):
        raise ValueError("results must contain every part exactly once")
    absolute_error = sum(part["absolute_error_sum"] for part in parts)
    force_components = sum(part["force_components"] for part in parts)
    return {
        "model": parts[0]["model"],
        "evaluated_structures": sum(p["evaluated_structures"] for p in parts),
        "skipped_structures": sum(p["skipped_structures"] for p in parts),
        "force_mae_meV_per_A": 1000 * absolute_error / force_components,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("source_dir", type=Path)
    prepare_parser.add_argument("output_dir", type=Path)

    run_parser = commands.add_parser("run")
    run_parser.add_argument("model", choices=MODELS)
    run_parser.add_argument("data_dir", type=Path)
    run_parser.add_argument("output", type=Path)
    run_parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    run_parser.add_argument("--part", type=int, default=0)
    run_parser.add_argument("--parts", type=int, default=1)

    merge_parser = commands.add_parser("merge")
    merge_parser.add_argument("output", type=Path)
    merge_parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    if args.command == "prepare":
        prepare(args.source_dir, args.output_dir)
        return
    if args.command == "run":
        result = evaluate(
            args.model,
            args.data_dir,
            args.checkpoint_dir,
            args.part,
            args.parts,
        )
    else:
        result = merge(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Force MAE: {result['force_mae_meV_per_A']:.3f} meV/Å")


if __name__ == "__main__":
    main()
