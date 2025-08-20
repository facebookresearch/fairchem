# example how to use checkpoint fixtures
from __future__ import annotations

import os
from functools import partial

import torch
from e3nn.o3 import rand_matrix

from fairchem.core.datasets.ase_datasets import AseDBDataset
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.simple_collater import data_list_collater
from fairchem.core.modules.normalization.normalizer import Normalizer
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.mlip_unit import Task

# Test equivariance in both fp32 and fp64
# If error in equivariance is due to numerical error in fp
# Then fp64 should have substainally lower variance than fp32
# Otherwise error mi


def test_embeddings(conserving_mole_checkpoint, fake_uma_dataset):
    inference_checkpoint_path, _ = conserving_mole_checkpoint
    db = AseDBDataset(config={"src": os.path.join(fake_uma_dataset, "oc20")})

    predictor = MLIPPredictUnit(inference_checkpoint_path, device="cpu")
    oc20_embeddings_tasks = Task(
        name="oc20_embeddings",
        level="atom",
        loss_fn=torch.nn.L1Loss(),
        property="embeddings",
        out_spec=None,
        normalizer=Normalizer(mean=0.0, rmsd=1.0),
        datasets=["oc20"],
    )
    predictor.tasks["oc20_embeddings"] = oc20_embeddings_tasks
    predictor.dataset_to_tasks["oc20"].append(oc20_embeddings_tasks)
    predictor.model = predictor.model.to(torch.float64)

    a2g = partial(
        AtomicData.from_ase,
        max_neigh=10,
        radius=100,
        r_edges=False,
        r_data_keys=["spin", "charge"],
        target_dtype=torch.float64,
    )

    # check that each system is the correct size
    single_system_embeddings = []
    for sample_idx in range(5):
        torch.manual_seed(42)

        sample = a2g(db.get_atoms(sample_idx), task_name="oc20")
        sample.pos += 500
        sample.cell *= 2000

        batch = data_list_collater([sample], otf_graph=True)

        original_positions = batch.pos.clone()

        out = predictor.predict(batch)

        assert out["forces"].shape[0] == out["embeddings"].shape[0]
        single_system_embeddings.append(out["embeddings"])

        rotation = rand_matrix(dtype=torch.float64)

        # check that L0/L1 embeddings are invariant/equivariant
        batch.pos = torch.einsum("ij,jk->ik", original_positions, rotation)
        out_rotated = predictor.predict(batch)

        assert (
            out["embeddings"][:, 0].isclose(out_rotated["embeddings"][:, 0]).all()
        ), "L0 Embeddings are not invariant under rotation"

    # check that embeddings when batched differently are similar enough
    torch.manual_seed(42)
    samples = [
        a2g(db.get_atoms(sample_idx), task_name="oc20") for sample_idx in range(5)
    ]
    for sample in samples:
        sample.pos += 500
        sample.cell *= 2000

    batch = data_list_collater(samples, otf_graph=True)
    out = predictor.predict(batch)
    assert out["embeddings"].isclose(torch.vstack(single_system_embeddings)).all()
