from __future__ import annotations

from functools import partial

import torch
from ase.build import molecule

from fairchem.core import FAIRChemCalculator, pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.simple_collater import data_list_collater
from fairchem.core.modules.normalization.normalizer import Normalizer
from fairchem.core.units.mlip_unit.mlip_unit import Task
from fairchem.core.units.mlip_unit.predict import AdditionalInferenceTasks


def test_get_descriptors():
    predictor = pretrained_mlip.get_predict_unit("uma-s-1", device="cpu")
    singlet = molecule("CH2_s1A1d")
    singlet.info.update({"spin": 1, "charge": 0})
    calc = FAIRChemCalculator(predictor, task_name="omol")

    embeddings = calc.get_descriptors(singlet)
    assert "embeddings_layer-1_l0" in embeddings

    embeddings = calc.get_descriptors(singlet, layers_and_ls=[(-1, 1)])
    assert "embeddings_layer-1_l1" in embeddings

    embeddings = calc.get_descriptors(singlet, layers_and_ls=[(-1, 1), (-1, 0)])
    assert "embeddings_layer-1_l1" in embeddings
    assert "embeddings_layer-1_l0" in embeddings


def test_embeddings_from_predict():
    layers_and_ls = [(4, 0), (3, 1)]
    predictor = pretrained_mlip.get_predict_unit(
        "uma-s-1",
        device="cpu",
        overrides={
            "heads": {
                "embeddings_head": {
                    "module": "fairchem.core.models.uma.escn_md.Embedding_Head",
                    "layers_and_ls": layers_and_ls,
                }
            }
        },
    )

    dataset = "omol"

    a2g = partial(
        AtomicData.from_ase,
        max_neigh=10,
        radius=100,
        r_edges=False,
        r_data_keys=["spin", "charge"],
        task_name=dataset,
    )

    singlet = molecule("CH2_s1A1d")
    singlet.info.update({"spin": 1, "charge": 0})

    batch = data_list_collater([a2g(singlet)], otf_graph=True)

    additional_inference_tasks = []
    for layer_idx, l_idx in layers_and_ls:
        property_name = f"embeddings_layer{layer_idx}_l{l_idx}"
        additional_inference_tasks.append(
            Task(
                name=property_name,
                level="atom",
                loss_fn=torch.nn.L1Loss(),
                property=property_name,
                out_spec=None,
                normalizer=Normalizer(mean=0.0, rmsd=1.0),
                datasets=[dataset],
            )
        )

    # add and remove an additional head , and tasks
    with AdditionalInferenceTasks(predictor, additional_inference_tasks):
        out = predictor.predict(batch)

    assert "embeddings_layer4_l0" in out
    assert "embeddings_layer3_l1" in out
