from __future__ import annotations

import os
from functools import partial

import torch

from fairchem.core import FAIRChemCalculator, pretrained_mlip
from fairchem.core.datasets.ase_datasets import AseDBDataset
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.simple_collater import data_list_collater
from fairchem.core.modules.normalization.normalizer import Normalizer
from fairchem.core.units.mlip_unit.mlip_unit import Task
from fairchem.core.units.mlip_unit.predict import AdditionalInferenceTasks


def test_get_descriptors(fake_uma_dataset):
    db = AseDBDataset(config={"src": os.path.join(fake_uma_dataset, "oc20")})
    predictor = pretrained_mlip.get_predict_unit("uma-s-1", device="cpu")
    atoms = db.get_atoms(0)

    calc = FAIRChemCalculator(predictor, task_name="omol")

    embeddings = calc.get_descriptors(atoms)
    assert "embeddings_layer-1_l0" in embeddings

    embeddings = calc.get_descriptors(atoms, layers_and_ls=[(-1, 1)])
    assert "embeddings_layer-1_l1" in embeddings

    embeddings = calc.get_descriptors(atoms, layers_and_ls=[(-1, 1), (-1, 0)])
    assert "embeddings_layer-1_l1" in embeddings
    assert "embeddings_layer-1_l0" in embeddings


def test_embeddings_from_predict(fake_uma_dataset):
    db = AseDBDataset(config={"src": os.path.join(fake_uma_dataset, "oc20")})
    layers_and_ls = [(4, 0), (3, 1)]

    predictor = pretrained_mlip.get_predict_unit("uma-s-1", device="cpu")

    predictor.model.module.output_heads["embeddings"] = (
        predictor.model.module.backbone.get_embedding_head(layers_and_ls=layers_and_ls)
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

    atoms = db.get_atoms(0)

    batch = data_list_collater([a2g(atoms)], otf_graph=True)

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


def test_embeddings_from_predict_override(fake_uma_dataset):
    db = AseDBDataset(config={"src": os.path.join(fake_uma_dataset, "oc20")})

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

    atoms = db.get_atoms(0)

    batch = data_list_collater([a2g(atoms)], otf_graph=True)

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
