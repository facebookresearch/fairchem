"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

run using,
pytest -s --inference-checkpoint inference_ckpt.pt tests/core/units/mlip_unit/test_inference_checkpoint.py
python -m pytest tests/core/units/mlip_unit/test_inference_checkpoint.py -s --inference-checkpoint ~/may12_checkpoint.pt
python -m pytest tests/core/units/mlip_unit/test_inference_checkpoint.py::test_conserving_mole_inference_modes_gpu -s --inference-checkpoint ~/may12_checkpoint.pt -vv --inference-dataset /checkpoint/ocp/shared/omol/250430-release/val
"""

from __future__ import annotations

import os
from functools import partial

import numpy as np
import pytest
import torch

from fairchem.core import FAIRChemCalculator
from fairchem.core.datasets.ase_datasets import AseDBDataset
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.simple_collater import data_list_collater
from fairchem.core.units.mlip_unit import InferenceSettings, MLIPPredictUnit


@pytest.mark.inference_check()
def test_inference_checkpoint_direct(
    command_line_inference_checkpoint, fake_uma_dataset, torch_deterministic
):
    predictor = MLIPPredictUnit(command_line_inference_checkpoint, device="cpu")

    db = AseDBDataset(config={"src": os.path.join(fake_uma_dataset, "oc20")})

    a2g = partial(
        AtomicData.from_ase,
        max_neigh=10,
        radius=100,
        r_energy=False,
        r_forces=False,
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )

    energies = []
    forces = []

    sample_idx = 0
    while sample_idx < min(5, len(db)):
        sample = a2g(db.get_atoms(sample_idx), task_name="oc20")
        batch = data_list_collater([sample], otf_graph=False)

        out = predictor.predict(batch)
        energies.append(out["energy"])
        forces.append(out["forces"])
        sample_idx += 1
    forces = torch.vstack(forces)
    energies = torch.stack(energies)

    print(
        f"oc20_energies_abs_mean: {energies.abs().mean().item()}, oc20_forces_abs_mean: {forces.abs().mean().item()}"
    )
    print(f"Keys in output {out.keys()}")


@pytest.mark.inference_check()
@pytest.mark.inference_dataset()
@pytest.mark.gpu()
@pytest.mark.parametrize(
    "tf32, activation_checkpointing, merge_mole, compile, wigner_cuda, external_graph_gen",
    [
        (False, False, False, False, False, True),  # test external graph gen
        (False, False, False, False, False, False),  # test internal graph gen
        (True, False, False, False, True, True),  # test wigner cuda
        (True, False, True, True, False, True),  # test compile and merge
        # with acvitation checkpointing
        (True, True, True, True, True, True),  # test external model graph gen + compile
        (True, True, True, False, True, True),  # test merge but no compile
        (True, True, False, False, True, True),  # test no merge or compile
    ],
)
def test_conserving_mole_inference_modes_gpu(
    tf32,
    activation_checkpointing,
    merge_mole,
    compile,
    wigner_cuda,
    external_graph_gen,
    command_line_inference_checkpoint,
    command_line_inference_dataset,
    fake_uma_dataset,
):
    if command_line_inference_dataset is None:
        task = "oc20"
        db = AseDBDataset(config={"src": os.path.join(fake_uma_dataset, "oc20")})
    else:
        task = "omol"
        db = AseDBDataset(config={"src": command_line_inference_dataset})
    # /checkpoint/ocp/shared/omol/250430-release/val

    calc = FAIRChemCalculator(
        checkpoint_path=command_line_inference_checkpoint,
        device="cuda",
        task_name=task,
        inference_settings=InferenceSettings(
            tf32=tf32,
            activation_checkpointing=activation_checkpointing,
            merge_mole=merge_mole,
            compile=compile,
            wigner_cuda=wigner_cuda,
            external_graph_gen=external_graph_gen,
        ),
    )
    calc.task_name = task

    energies = []
    forces = []

    sample_idx = 0
    while sample_idx < min(5, len(db)):
        atoms = db.get_atoms(sample_idx)
        target_energy = atoms.get_potential_energy()
        target_forces = atoms.get_forces()
        atoms.calc = calc
        energies.append(target_energy - atoms.get_potential_energy())
        forces.append(target_forces - atoms.get_forces())
        sample_idx += 1
    forces = np.vstack(forces)
    energies = np.stack(energies)

    print(
        f"energies_abs_mean: {np.abs(energies).mean().item()}, forces_abs_mean: {np.abs(forces).mean().item()}"
    )
