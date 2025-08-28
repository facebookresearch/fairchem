from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import hydra
import numpy as np
import torch
from ase.data import atomic_masses, chemical_symbols
from lammps import lammps

from fairchem.core.datasets.atomic_data import AtomicData

if TYPE_CHECKING:
    from fairchem.core.units.mlip_unit.predict import MLIPPredictUnitProtocol
    from fairchem.core.units.mlip_unit.utils import DictConfig

FIX_EXT_ID = "ext_uma"
FIX_EXTERNAL_CMD = f"fix {FIX_EXT_ID} all external pf/callback 1 1"

FORCE_COMMANDS = ["pair_style", "bond_style", "angle_style", "dihedral_style"]


def check_input_script(input_script: str):
    for cmd in FORCE_COMMANDS:
        if cmd in input_script:
            logging.warning(
                f"Input script contains force field command '{cmd}'. These forces will be incorrectly added to the MLIP forces, please remove them unless you know what you are doing."
            )


def check_atom_id_match_masses(types_arr, masses):
    for atom_id in types_arr:
        assert np.allclose(
            masses[atom_id], atomic_masses[atom_id], atol=1e-1
        ), f"Atom {chemical_symbols[atom_id]} (type {atom_id}) has mass {masses[atom_id]} but is expected to have mass {atomic_masses[atom_id]}."


def lookup_atomic_number_by_mass(mass_arr: np.ndarray | float) -> np.ndarray | int:
    """
    Lookup atomic numbers by closest atomic masses.

    Args:
        mass_arr (float or np.ndarray): Target atomic mass(es)

    Returns:
        int or np.ndarray: Atomic number(s)

    Raises:
        ValueError: If any mass doesn't match within 0.1 tolerance
    """
    # Convert input to numpy array
    mass_arr = np.asarray(mass_arr)
    scalar_input = mass_arr.ndim == 0

    # Ensure we work with at least 1D array
    if scalar_input:
        mass_arr = mass_arr.reshape(1)

    # Convert ASE atomic masses to numpy array (skip index 0 which is empty)
    reference_masses = np.array(atomic_masses[1:])

    # Calculate absolute differences for all input masses vs all reference masses
    # Shape: (len(mass_arr), len(reference_masses))
    diffs = np.abs(mass_arr[:, np.newaxis] - reference_masses[np.newaxis, :])

    # Find minimum difference and corresponding index for each input mass
    min_indices = np.argmin(diffs, axis=1)
    min_diffs = np.min(diffs, axis=1)

    # Check if any mass is outside tolerance
    bad_matches = min_diffs > 0.1
    if np.any(bad_matches):
        bad_masses = mass_arr[bad_matches]
        raise ValueError(f"No atomic mass found within 0.1 of {bad_masses}")

    # Add 1 because we skipped index 0
    atomic_numbers = min_indices + 1

    # Return scalar if input was scalar
    if scalar_input:
        return atomic_numbers[0]

    return atomic_numbers


def separate_run_commands(input_script: str) -> str:
    lines = input_script.splitlines()
    run_cmds = []
    script = []
    for line in lines:
        if line.startswith("run"):
            run_cmds.append(line)
        else:
            script.append(line)
    return script, run_cmds


# TODO: doubles check this
def cell_from_lammps_box(boxlo, boxhi, xy, yz, xz):
    lx = boxhi[0] - boxlo[0]
    ly = boxhi[1] - boxlo[1]
    lz = boxhi[2] - boxlo[2]

    unit_cell_matrix = torch.tensor(
        [
            [lx, xy, xz],  # First column: a vector
            [0, ly, yz],  # Second column: b vector
            [0, 0, lz],  # Third column: c vector
        ],
        dtype=torch.float32,
    )
    return unit_cell_matrix.unsqueeze(0)


def fix_external_call_back(lmp, ntimestep, nlocal, tag, x, f):
    # force copy here, otherwise we can accident modify the original array in lammps
    atom_type_np = lmp.numpy.extract_atom("type")
    masses = lmp.numpy.extract_atom("mass")
    # TODO: do this only once at the beginning to the sim and cache it
    atomic_mass_arr = masses[atom_type_np]
    atomic_numbers = lookup_atomic_number_by_mass(atomic_mass_arr)
    # TODO  is there a way to check atom types are mapped correctly?
    predictor = lmp._predictor
    pos = torch.tensor(x, dtype=torch.float32)
    box_info = lmp.extract_box()
    boxlo, boxhi, xy, yz, xz, periodicity, box_change = box_info
    pbc = torch.tensor(periodicity, dtype=torch.bool).unsqueeze(0)
    cell = cell_from_lammps_box(boxlo, boxhi, xy, yz, xz)
    # if not using lammps neighborlist
    edge_index = torch.empty((2, 0), dtype=torch.long)
    cell_offsets = torch.empty((0, 3), dtype=torch.float32)
    nedges = torch.tensor([0], dtype=torch.long)
    tags = torch.zeros(nlocal, dtype=torch.long)
    fixed = torch.zeros(nlocal, dtype=torch.long)
    batch = torch.zeros(nlocal, dtype=torch.long)
    atomic_data = AtomicData(
        pos=pos,
        atomic_numbers=torch.tensor(atomic_numbers),
        cell=cell,
        pbc=pbc,
        natoms=torch.tensor([nlocal], dtype=torch.long),
        edge_index=edge_index,
        cell_offsets=cell_offsets,
        nedges=nedges,
        charge=torch.LongTensor([0]),
        spin=torch.LongTensor([0]),
        fixed=fixed,
        tags=tags,
        batch=batch,
        dataset=["omat"],
    )
    results = predictor.predict(atomic_data)
    f[:] = results["forces"].cpu().numpy()[:]
    lmp.fix_external_set_energy_global(FIX_EXT_ID, results["energy"].item())


def run_lammps_with_uma(predictor: MLIPPredictUnitProtocol, lammps_input_path: str):
    machine = None
    if "LAMMPS_MACHINE_NAME" in os.environ:
        machine = os.environ["LAMMPS_MACHINE_NAME"]
    lmp = lammps(name=machine, cmdargs=["-nocite", "-log", "none", "-echo", "screen"])
    lmp._predictor = predictor

    run_cmds = []
    with open(lammps_input_path) as f:
        input_script = f.read()
        check_input_script(input_script)
        script, run_cmds = separate_run_commands(input_script)
        logging.info(f"Running input script: {input_script}")
        lmp.commands_list(script)
        lmp.command(FIX_EXTERNAL_CMD)
        lmp.set_fix_external_callback(FIX_EXT_ID, fix_external_call_back, lmp)
        lmp.commands_list(run_cmds)
    del lmp._predictor


@hydra.main(
    version_base=None,
    config_path=".",
    config_name="lammps_uma_config",
)
def main(cfg: DictConfig):
    predict_unit = hydra.utils.instantiate(cfg.predict_unit)
    run_lammps_with_uma(predict_unit, cfg.lmp_in)


if __name__ == "__main__":
    main()
