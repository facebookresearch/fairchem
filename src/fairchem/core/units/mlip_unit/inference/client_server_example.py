from __future__ import annotations

import logging
import timeit

from fairchem.core.calculate.pretrained_mlip import pretrained_checkpoint_path_from_name
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.common_structures import get_fcc_carbon_xtal
from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
)
from fairchem.core.units.mlip_unit.predict import ParallelMLIPPredictUnit

logging.basicConfig(level=logging.INFO)


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
            wigner_cuda=False,
            compile=False,
            activation_checkpointing=False,
            internal_graph_gen_version=2,
            external_graph_gen=False,
        ),
        server_config={"workers": 8},
    )
    atoms = get_fcc_carbon_xtal(5000)
    # calc = FAIRChemCalculator(ppunit, task_name="omol")
    # atoms.calc = calc
    # print(atoms.get_potential_energy())
    # ppunit.cleanup()
    atomic_data = AtomicData.from_ase(atoms, task_name=["omat"])
    logging.info("Starting profile")
    qps, ns_per_day = get_qps(atomic_data, ppunit, warmups=10, timeiters=10)
    logging.info(f"QPS: {qps}, ns/day: {ns_per_day}")


if __name__ == "__main__":
    main()
