from __future__ import annotations

from ase import build

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit.api.inference import inference_settings_default
from fairchem.core.units.mlip_unit.predict import ParallelMLIPPredictUnit


def main():
    # Create an AtomicData object
    h2o = build.molecule("H2O")
    atomic_data = AtomicData.from_ase(h2o)
    atomic_data.task_name = ["omol"]

    ppunit = ParallelMLIPPredictUnit(
        inference_model_path="/checkpoint/ocp/shared/uma/release/uma_sm_osc_name_fix.pt",
        device="cuda",
        inference_settings=inference_settings_default(),
        server_config={"workers": 2},
    )
    result = ppunit.predict_step(None, atomic_data)
    print(result)

    # client = MLIPInferenceClient("localhost", 8001)
    # result = client.call(atomic_data)
    # print(result)


if __name__ == "__main__":
    main()
