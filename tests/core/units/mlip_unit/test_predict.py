from __future__ import annotations

import pytest
from ase.build import add_adsorbate, bulk, fcc100, molecule

from fairchem.core import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch


@pytest.fixture(scope="module")
def uma_predict_unit(request):
    uma_models = [name for name in pretrained_mlip.available_models if "uma" in name]
    return pretrained_mlip.get_predict_unit(uma_models[0])


def test_single_dataset_predict(uma_predict_unit):
    atomic_data_list = [
        AtomicData.from_ase(bulk("Pt"), dataset="omat") for _ in range(100)
    ]
    batch = atomicdata_list_to_batch(atomic_data_list)

    preds = uma_predict_unit.predict(batch)

    assert preds["energy"].shape == (100,)
    assert preds["forces"].shape == (100, 3)
    assert preds["stress"].shape == (100, 9)


def test_multiple_dataset_predict(uma_predict_unit):
    h2o = molecule("H2O")
    h2o.info.update({"charge": 0, "spin": 1})
    h2o.pbc = True  # all data points must be pbc if mixing.

    slab = fcc100("Cu", (3, 3, 3), vacuum=8, periodic=True)
    adsorbate = molecule("CO")
    add_adsorbate(slab, adsorbate, 2.0, "bridge")

    pt = bulk("Pt")
    pt.repeat((2, 2, 2))

    atomic_data_list = [
        AtomicData.from_ase(
            h2o, dataset="omol", r_data_keys=["spin", "charge"], molecule_cell_size=10
        ),
        AtomicData.from_ase(slab, dataset="oc20"),
        AtomicData.from_ase(pt, dataset="omat"),
    ]

    batch = atomicdata_list_to_batch(atomic_data_list)
    preds = uma_predict_unit.predict(batch)

    n_systems = len(batch)
    n_atoms = sum(batch.natoms).item()
    assert preds["energy"].shape == (n_systems,)
    assert preds["forces"].shape == (n_atoms, 3)
    assert preds["stress"].shape == (n_systems, 9)
