from __future__ import annotations

import logging
import timeit

import numpy as np
from ase import build

from fairchem.core.calculate.pretrained_mlip import pretrained_checkpoint_path_from_name
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
)
from fairchem.core.units.mlip_unit.predict import ParallelMLIPPredictUnit

logging.basicConfig(level=logging.INFO)


def get_fcc_carbon_xtal(
    num_atoms: int,
    lattice_constant: float = 3.8,
):
    # lattice_constant = 3.8, fcc generates a supercell with ~50 edges/atom
    atoms = build.bulk("C", "fcc", a=lattice_constant)
    n_cells = int(np.ceil(np.cbrt(num_atoms)))
    atoms = atoms.repeat((n_cells, n_cells, n_cells))
    indices = np.random.choice(len(atoms), num_atoms, replace=False)
    sampled_atoms = atoms[indices]
    return sampled_atoms
    # return AtomicData.from_ase(sampled_atoms, task_name=["omol"])


def get_qps(data, predictor, warmups: int = 10, timeiters: int = 100):
    def timefunc():
        predictor.predict(data)

    for _ in range(warmups):
        timefunc()

    result = timeit.timeit(timefunc, number=timeiters)
    qps = timeiters / result
    ns_per_day = qps * 24 * 3600 / 1e6
    return qps, ns_per_day


def main():
    ppunit = ParallelMLIPPredictUnit(
        inference_model_path=pretrained_checkpoint_path_from_name("uma-s-1p1"),
        device="cuda",
        inference_settings=InferenceSettings(
            tf32=True,
            merge_mole=True,
            wigner_cuda=True,
            compile=False,
            activation_checkpointing=False,
            internal_graph_gen_version=2,
            external_graph_gen=False,
        ),
        server_config={"workers": 8},
    )
    atoms = get_fcc_carbon_xtal(5000)

    atomic_data = AtomicData.from_ase(atoms, task_name=["omat"])
    logging.info("Starting profile")
    qps, ns_per_day = get_qps(atomic_data, ppunit, warmups=10, timeiters=10)
    logging.info(f"QPS: {qps}, ns/day: {ns_per_day}")


if __name__ == "__main__":
    main()
