"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests:  Finite NVE, NPT, and Langevin LAMMPS dynamics with UMA forces.
Models: uma-s-1p1.
CI:     test_lammps_gpu.

Strict energy/force transfer is covered separately by
test_preflight_integration.py. Stress sign, units, and component order are
covered in test_lammps_fc.py; the NPT case here also requires a cell response.
Different ASE and LAMMPS thermostats and barostats are deliberately not
compared at the end of a trajectory.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fairchem.lammps.lammps_fc import run_lammps_with_fairchem

from tests.conftest import get_predict_unit_for_test

pytestmark = [pytest.mark.pretrained("uma-s-1p1")]


def run_lammps(input_file, pretrained_checkpoint):
    predictor = get_predict_unit_for_test(pretrained_checkpoint, device="cuda")
    lmp = run_lammps_with_fairchem(predictor, input_file, "omat")
    thermo = lmp.last_thermo()
    boxlo, boxhi, *_ = lmp.extract_box()
    cell_volume = abs(
        (boxhi[0] - boxlo[0]) * (boxhi[1] - boxlo[1]) * (boxhi[2] - boxlo[2])
    )
    lmp.close()
    return thermo, cell_volume


def assert_finite_thermodynamics(thermo):
    for key in ("KinEng", "PotEng", "TotEng", "Temp", "Press"):
        assert math.isfinite(float(thermo[key])), f"Non-finite {key}: {thermo[key]}"


@pytest.mark.gpu()
@pytest.mark.parametrize(
    "input_file",
    [
        "tests/lammps/lammps_nve.file",
        "tests/lammps/lammps_npt.file",
        "tests/lammps/lammps_langevin.file",
    ],
)
def test_lammps_dynamics_remain_finite(input_file, pretrained_checkpoint):
    thermo, cell_volume = run_lammps(input_file, pretrained_checkpoint)
    assert_finite_thermodynamics(thermo)
    if input_file.endswith("lammps_npt.file"):
        initial_volume = (2 * 3.567) ** 3
        assert not np.isclose(cell_volume, initial_volume, atol=1.0e-6)
