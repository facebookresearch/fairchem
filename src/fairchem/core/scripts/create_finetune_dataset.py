from __future__ import annotations

import argparse
import glob
import logging
import multiprocessing as mp
import os
import random
from pathlib import Path

import numpy as np
from ase.db import connect
from ase.io import read
from tqdm import tqdm

from fairchem.core.datasets import AseDBDataset

logging.basicConfig(level=logging.INFO)


def compute_normalizer_and_linear_reference(train_path, num_workers):
    """
    Given a path to an ASE database file, compute the normalizer value and linear
    reference coefficients. These are used to normalize energies and forces during
    training. For large datasets, compute this for only a subset of the data.
    """
    global dataset
    dataset = AseDBDataset({"src": str(train_path)})

    sample_indices = random.sample(range(len(dataset)), min(100000, len(dataset)))
    with mp.Pool(num_workers) as pool:
        outputs = list(
            tqdm(
                pool.imap(extract_energy_and_forces, sample_indices),
                total=len(sample_indices),
                desc="Computing normalizer values.",
            )
        )
        atomic_numbers = [x[0] for x in outputs]
        energies = [x[1] for x in outputs]
        forces = np.array([force for x in outputs for force in x[2]])
        force_rms = np.sqrt(np.mean(np.square(forces)))
        assert (
            np.isfinite(force_rms).all()
        ), "We found non-finite values in the forces, please check your input data!"
        coeff = compute_lin_ref(atomic_numbers, energies)
    return force_rms, coeff


def extract_energy_and_forces(idx):
    """
    Extract energy and forces from an ASE atoms object at a given index in the dataset.
    """
    atoms = dataset.get_atoms(idx)
    atomic_numbers = atoms.get_atomic_numbers()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    n_atoms = len(atoms)

    fixed_idx = np.zeros(n_atoms)
    if hasattr(atoms, "constraints"):
        from ase.constraints import FixAtoms

        for constraint in atoms.constraints:
            if isinstance(constraint, FixAtoms):
                fixed_idx[constraint.index] = 1

    mask = fixed_idx == 0
    forces = forces[mask]

    return atomic_numbers, energy, forces


def compute_lin_ref(atomic_numbers, energies):
    """
    Compute linear reference coefficients given atomic numbers and energies.
    """
    features = [np.bincount(x, minlength=100).astype(int) for x in atomic_numbers]

    X = np.vstack(features)
    y = energies

    coeff = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.isfinite(
        coeff
    ).all(), "Found some non-finite values while computing element references, please check/sanitize your inputs before proceeding"
    return coeff.tolist()


def write_ase_db(mp_arg):
    """
    Write ASE atoms objects to an ASE database file. This function is designed to be
    run in parallel using multiprocessing.
    """
    db_file, file_list, worker_id = mp_arg

    successful = []
    failed = []
    natoms = []
    with connect(str(db_file)) as db:
        for file in tqdm(file_list, position=worker_id):
            atoms_list = read(file, ":")
            for i, atoms in enumerate(atoms_list):
                try:
                    assert (
                        atoms.calc is not None
                    ), "No calculator attached to atoms object."
                    assert "energy" in atoms.calc.results, "Missing energy result"
                    assert "forces" in atoms.calc.results, "Missing forces result"
                    db.write(atoms, data=atoms.info)
                    natoms.append(len(atoms))
                    successful.append(f"{file},{i}")
                except AssertionError as e:
                    failed.append(f"{file},{i}: {e!s}")

    return db_file, natoms, successful, failed


def launch_processing(data_dir, output_dir, num_workers):
    """
    Driver script to launch processing of ASE atoms files into an ASE database.
    """
    os.makedirs(output_dir, exist_ok=True)
    input_files = [
        f
        for f in glob.glob(os.path.join(data_dir, "**/*"), recursive=True)
        if os.path.isfile(f)
    ]
    chunked_files = np.array_split(input_files, num_workers)
    db_files = [output_dir / f"data.{i:04d}.aselmdb" for i in range(num_workers)]
    mp_args = [(db_files[i], chunked_files[i], i) for i in range(num_workers)]
    with mp.Pool(num_workers) as pool:
        outputs = pool.map(write_ase_db, mp_args)

    # Log results
    natoms = []
    for output in outputs:
        db_file, _natoms, successful, failed = output
        natoms.extend(_natoms)
        log_file = db_file.with_suffix(".log")
        failed_file = db_file.with_suffix(".failed")

        with open(log_file, "w") as log:
            log.write("\n".join(successful))
        with open(failed_file, "w") as failed_log:
            failed_log.write("\n".join(failed))
    np.savez_compressed(output_dir / "metadata.npz", natoms=natoms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-dir",
        type=str,
        required=True,
        help="Directory of ASE atoms objects to convert for training.",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        required=True,
        help="Directory of ASE atoms objects to convert for validation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory to save required finetuning artifacts.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers for processing files.",
    )
    args = parser.parse_args()

    # Launch processing for training data
    train_path = args.output_dir / "train"
    launch_processing(args.train_dir, train_path, args.num_workers)
    force_rms, linref_coeff = compute_normalizer_and_linear_reference(
        train_path, args.num_workers
    )
    val_path = args.output_dir / "val"
    launch_processing(args.val_dir, val_path, args.num_workers)

    logging.info(f"Generated dataset at {args.output_dir}")
