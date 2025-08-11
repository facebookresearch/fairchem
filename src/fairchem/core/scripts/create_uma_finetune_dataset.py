from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import yaml

from fairchem.core.scripts.create_finetune_dataset import (
    compute_normalizer_and_linear_reference,
    launch_processing,
)
from fairchem.core.units.mlip_unit.api.inference import UMATask

logging.basicConfig(level=logging.INFO)

TEMPLATE_DIR = Path("configs/uma/finetune")
DATA_YAML_DIR = Path("data")
REGRESSION_LABEL_TO_TASK_YAML = {
    "e": DATA_YAML_DIR / Path("uma_conserving_data_task_energy.yaml"),
    "ef": DATA_YAML_DIR / Path("uma_conserving_data_task_energy_force.yaml"),
    "efs": DATA_YAML_DIR / Path("uma_conserving_data_task_energy_force_stress.yaml"),
}
UMA_SM_FINETUNE_YAML = Path("uma_sm_finetune_template.yaml")


def create_yaml(
    train_path: str,
    val_path: str,
    force_rms: float,
    linref_coeff: list,
    output_dir: str,
    dataset_name: str,
    regression_tasks: str,
    base_model_name: str,
):
    data_task_yaml = TEMPLATE_DIR / REGRESSION_LABEL_TO_TASK_YAML[regression_tasks]
    with open(data_task_yaml) as file:
        template = yaml.safe_load(file)
        template["dataset_name"] = dataset_name
        template["normalizer_rmsd"] = force_rms
        template["elem_refs"] = linref_coeff
        template["train_dataset"]["splits"]["train"]["src"] = train_path
        template["val_dataset"]["splits"]["val"]["src"] = val_path
        # add extra large vaccum box for molecules
        # if dataset_name == str(UMATask.OMOL):
        #     template["train_dataset"]["a2g_args"]["molecule_cell_size"] = 1000.0
        os.makedirs(output_dir / DATA_YAML_DIR, exist_ok=True)
        with open(
            output_dir / REGRESSION_LABEL_TO_TASK_YAML[regression_tasks], "w"
        ) as yaml_file:
            yaml.dump(template, yaml_file, default_flow_style=False, sort_keys=False)

    uma_finetune_yaml = TEMPLATE_DIR / UMA_SM_FINETUNE_YAML
    with open(uma_finetune_yaml) as file:
        template_ft = yaml.safe_load(file)
        template_ft["base_model_name"] = base_model_name
        template_ft["defaults"][0]["data"] = REGRESSION_LABEL_TO_TASK_YAML[
            regression_tasks
        ].stem
        template_ft["train_dataset"]["dataset_configs"][dataset_name] = template_ft[
            "train_dataset"
        ]["dataset_configs"].pop("DATASET_NAME")
        template_ft["val_dataset"]["dataset_configs"][dataset_name] = template_ft[
            "val_dataset"
        ]["dataset_configs"].pop("DATASET_NAME")
    with open(output_dir / UMA_SM_FINETUNE_YAML, "w") as yaml_file:
        yaml.dump(template_ft, yaml_file, default_flow_style=False, sort_keys=False)


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
        "--uma-task",
        type=str,
        required=True,
        choices=[t.value for t in UMATask],
        help="choose a uma task to finetune",
    )
    parser.add_argument(
        "--regression-tasks",
        type=str,
        choices=["e", "ef", "efs"],
        required=True,
        help="Choose to finetune based on regression task set (you must have the corresponding labels in your dataset), can be energy (e), energy+force (ef) or energy+force+stress(efs)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="uma-s-1",
        help="Name of base uma model",
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
    assert not Path(
        args.output_dir
    ).exists(), f"{args.output_dir} can't already exist, please choose a different dir"

    # Launch processing for training data
    train_path = args.output_dir / "train"
    launch_processing(args.train_dir, train_path, args.num_workers)
    force_rms, linref_coeff = compute_normalizer_and_linear_reference(
        train_path, args.num_workers
    )
    # don't use force rms if doing energy only training
    if args.regression_tasks == "e":
        force_rms = 1.0
    val_path = args.output_dir / "val"
    launch_processing(args.val_dir, val_path, args.num_workers)

    create_yaml(
        train_path=str(train_path),
        val_path=str(val_path),
        force_rms=float(force_rms),
        linref_coeff=linref_coeff,
        output_dir=args.output_dir,
        dataset_name=args.uma_task,
        regression_tasks=args.regression_tasks,
        base_model_name=args.base_model,
    )
    logging.info(f"Generated dataset and data config yaml in {args.output_dir}")
    logging.info(
        f"To run finetuning, run fairchem -c {args.output_dir}/{UMA_SM_FINETUNE_YAML}"
    )
