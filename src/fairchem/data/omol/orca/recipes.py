from __future__ import annotations

import os

import psutil
from omdata.orca.calc import (
    NBO_FLAGS,
    OPT_PARAMETERS,
    ORCA_BASIS,
    ORCA_BLOCKS,
    ORCA_FUNCTIONAL,
    ORCA_SIMPLE_INPUT,
    Vertical,
    get_symm_break_block,
)
from quacc.recipes.orca.core import run_and_summarize, run_and_summarize_opt


def single_point_calculation(
    atoms,
    charge,
    spin_multiplicity,
    xc=ORCA_FUNCTIONAL,
    basis=ORCA_BASIS,
    orcasimpleinput=None,
    orcablocks=None,
    nprocs=12,
    outputdir=os.getcwd(),
    vertical=Vertical.Default,
    nbo=False,
    copy_files=None,
    **calc_kwargs,
):
    """
    Wrapper around QUACC's static job to standardize single-point calculations.
    See github.com/Quantum-Accelerators/quacc/blob/main/src/quacc/recipes/orca/core.py#L22
    for more details.

    Arguments
    ---------

    atoms: Atoms
        Atoms object
    charge: int
        Charge of system
    spin_multiplicity: int
        Multiplicity of the system
    xc: str
        Exchange-correlaction functional
    basis: str
        Basis set
    orcasimpleinput: list
        List of `orcasimpleinput` settings for the calculator
    orcablocks: list
        List of `orcablocks` swaps for the calculator
    nprocs: int
        Number of processes to parallelize across
    nbo: bool
        Run NBO as part of the Orca calculation
    outputdir: str
        Directory to move results to upon completion
    calc_kwargs:
        Additional kwargs for the custom Orca calculator
    """
    from quacc import SETTINGS

    SETTINGS.RESULTS_DIR = outputdir

    if orcasimpleinput is None:
        orcasimpleinput = ORCA_SIMPLE_INPUT.copy()
    if orcablocks is None:
        orcablocks = ORCA_BLOCKS.copy()
    if vertical == Vertical.MetalOrganics and spin_multiplicity == 1:
        orcasimpleinput.append("UKS")
        orcablocks.append(get_symm_break_block(atoms, charge))
    if not nbo:
        orcasimpleinput.extend(["NONBO", "NONPA"])
    else:
        orcablocks.append(NBO_FLAGS)

    nprocs = psutil.cpu_count(logical=False) if nprocs == "max" else nprocs
    default_inputs = [xc, basis, "engrad"]
    default_blocks = [f"%pal nprocs {nprocs} end"]

    doc = run_and_summarize(
        atoms,
        charge=charge,
        spin_multiplicity=spin_multiplicity,
        default_inputs=default_inputs,
        default_blocks=default_blocks,
        input_swaps=orcasimpleinput,
        block_swaps=orcablocks,
        copy_files=copy_files,
        **calc_kwargs,
    )

    return doc


def ase_relaxation(
    atoms,
    charge,
    spin_multiplicity,
    xc=ORCA_FUNCTIONAL,
    basis=ORCA_BASIS,
    orcasimpleinput=None,
    orcablocks=None,
    nprocs=12,
    opt_params=None,
    outputdir=os.getcwd(),
    vertical=Vertical.Default,
    copy_files=None,
    nbo=False,
    step_counter_start=0,
    **calc_kwargs,
):
    """
    Wrapper around QUACC's ase_relax_job to standardize geometry optimizations.
    See github.com/Quantum-Accelerators/quacc/blob/main/src/quacc/recipes/orca/core.py#L22
    for more details.

    Arguments
    ---------

    atoms: Atoms
        Atoms object
    charge: int
        Charge of system
    spin_multiplicity: int
        Multiplicity of the system
    xc: str
        Exchange-correlaction functional
    basis: str
        Basis set
    orcasimpleinput: list
        List of `orcasimpleinput` settings for the calculator
    orcablocks: list
        List of `orcablocks` swaps for the calculator
    nprocs: int
        Number of processes to parallelize across
    opt_params: dict
        Dictionary of optimizer parameters
    nbo: bool
        Run NBO as part of the Orca calculation
    step_counter_start: int
        Index to start step counter from (used for optimization restarts)
    outputdir: str
        Directory to move results to upon completion
    calc_kwargs:
        Additional kwargs for the custom Orca calculator
    """
    from quacc import SETTINGS

    SETTINGS.RESULTS_DIR = outputdir

    if orcasimpleinput is None:
        orcasimpleinput = ORCA_SIMPLE_INPUT.copy()
    if orcablocks is None:
        orcablocks = ORCA_BLOCKS.copy()
    if opt_params is None:
        opt_params = OPT_PARAMETERS.copy()
    if vertical == Vertical.MetalOrganics and spin_multiplicity == 1:
        orcasimpleinput.append("UKS")
        orcablocks.append(get_symm_break_block(atoms, charge))
    if not nbo:
        orcasimpleinput.extend(["NONBO", "NONPA"])
    else:
        orcablocks.append(NBO_FLAGS)

    nprocs = psutil.cpu_count(logical=False) if nprocs == "max" else nprocs
    default_inputs = [xc, basis, "engrad"]
    default_blocks = [f"%pal nprocs {nprocs} end"]

    doc = run_and_summarize_opt(
        atoms,
        charge=charge,
        spin_multiplicity=spin_multiplicity,
        default_inputs=default_inputs,
        default_blocks=default_blocks,
        input_swaps=orcasimpleinput,
        block_swaps=orcablocks,
        opt_params=opt_params,
        copy_files=copy_files,
        step_counter_start=step_counter_start,
        **calc_kwargs,
    )

    return doc
