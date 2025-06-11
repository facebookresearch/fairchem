from __future__ import annotations

import argparse
import glob
import logging
import multiprocessing as mp
import os
import random
import shutil
from pathlib import Path

import numpy as np
import yaml
from ase.db import connect
from ase.io import read
from tqdm import tqdm

from fairchem.core.datasets import AseDBDataset
from fairchem.core.units.mlip_unit.api.inference import UMATask

logging.basicConfig(level=logging.INFO)

TEMPLATE_DIR = Path("configs/uma/finetune")
DATA_TASK_YAML = Path("data/uma_conserving_data_task_template.yaml")
UMA_SM_FINETUNE_YAML = Path("uma_sm_finetune_template.yaml")


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
    return coeff.tolist()


def write_ase_db(mp_arg):
    """
    Write ASE atoms objects to an ASE database file. This function is designed to be
    run in parallel using multiprocessing.
    """
    db_file, file_list, worker_id = mp_arg

    successful = []
    failed = []
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
                    successful.append(f"{file},{i}")
                except AssertionError as e:
                    failed.append(f"{file},{i}: {e!s}")

    return db_file, successful, failed


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
    for output in outputs:
        db_file, successful, failed = output
        log_file = db_file.with_suffix(".log")
        failed_file = db_file.with_suffix(".failed")

        with open(log_file, "w") as log:
            log.write("\n".join(successful))
        with open(failed_file, "w") as failed_log:
            failed_log.write("\n".join(failed))


def create_yaml(
    train_path: str,
    val_path: str,
    force_rms: float,
    linref_coeff: list,
    output_dir: str,
    dataset_name: str,
    regress_stress: bool,
):
    data_task_yaml = TEMPLATE_DIR / DATA_TASK_YAML
    with open(data_task_yaml) as file:
        template = yaml.safe_load(file)
        template["dataset_name"] = dataset_name
        template["normalizer_rmsd"] = force_rms
        template["elem_refs"] = linref_coeff
        template["train_dataset"]["splits"]["train"]["src"] = train_path
        template["val_dataset"]["splits"]["val"]["src"] = val_path
        if not regress_stress:
            # remove the stress task
            template["tasks_list"].pop()
        os.makedirs(output_dir / "data", exist_ok=True)
        with open(output_dir / DATA_TASK_YAML, "w") as yaml_file:
            yaml.dump(template, yaml_file, default_flow_style=False, sort_keys=False)

    if not regress_stress:
        uma_finetune_yaml = TEMPLATE_DIR / UMA_SM_FINETUNE_YAML
        with open(uma_finetune_yaml) as file:
            template_ft = yaml.safe_load(file)
            template_ft["regress_stress"] = False
        with open(output_dir / UMA_SM_FINETUNE_YAML, "w") as yaml_file:
            yaml.dump(template_ft, yaml_file, default_flow_style=False, sort_keys=False)
    else:
        shutil.copy2(
            TEMPLATE_DIR / UMA_SM_FINETUNE_YAML, output_dir / UMA_SM_FINETUNE_YAML
        )


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
        "--task-to-finetune",
        type=str,
        required=True,
        choices=[t.value for t in UMATask],
        help="choose a uma task to finetune",
    )
    parser.add_argument(
        "--regress-stress",
        type=bool,
        default=False,
        help="Allow for stress regression tasks, will only work if your dataset has stress labels",
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

    create_yaml(
        train_path=str(train_path),
        val_path=str(val_path),
        force_rms=float(force_rms),
        linref_coeff=linref_coeff,
        output_dir=args.output_dir,
        dataset_name=args.task_to_finetune,
        regress_stress=args.regress_stress,
    )
    logging.info(f"Generated dataset and data config yaml in {args.output_dir}")
    logging.info(
        f"To run finetuning, run fairchem -c {args.output_dir}/{UMA_SM_FINETUNE_YAML}"
    )
