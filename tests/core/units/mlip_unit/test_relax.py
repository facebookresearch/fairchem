"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from ase.build import bulk, molecule
from ase.optimize import BFGS

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import MLIPPredictUnit


def seed_everywhere(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@pytest.mark.parametrize(
    "system_type,task_name,noise,fmax,max_steps",
    [
        ("crystal", "oc20", 0.05, 0.1, 200),
        ("molecule", "omol", 0.1, 0.05, 200),
    ],
)
@pytest.mark.parametrize("model_cls", [MLIPPredictUnit])
def test_relaxation(
    direct_checkpoint, model_cls, system_type, noise, task_name, fmax, max_steps
):
    seed_everywhere()
    direct_inference_checkpoint_pt, _ = direct_checkpoint

    if system_type == "crystal":
        atoms = bulk("Si", "diamond", a=5.0)
    else:  # molecule
        atoms = molecule("CH4")

    atoms.positions += np.random.normal(0, noise, atoms.positions.shape)

    predictor = MLIPPredictUnit(direct_inference_checkpoint_pt, device="cpu")
    calculator = FAIRChemCalculator(predictor, task_name=task_name)
    atoms.calc = calculator

    initial_energy = atoms.get_potential_energy()
    opt = BFGS(atoms)
    opt.run(fmax=fmax, steps=max_steps)

    forces = atoms.get_forces()
    assert np.max(np.abs(forces)) <= fmax or opt.nsteps >= max_steps
    final_energy = atoms.get_potential_energy()
    assert final_energy <= initial_energy
    assert not np.isnan(final_energy)
    assert not np.isnan(forces).any()
