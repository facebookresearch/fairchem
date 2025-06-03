"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import random
from functools import partial

import numpy as np
import pytest
import torch
from ase import build

from fairchem.core.datasets import data_list_collater
from fairchem.core.datasets.ase_datasets import AseDBDataset
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
    inference_settings_default,
    inference_settings_turbo,
)


@pytest.mark.parametrize(
    "tf32, activation_checkpointing, merge_mole, compile, wigner_cuda, external_graph_gen",
    [
        (False, False, False, False, False, True),  # test external graph gen
        (False, False, False, False, False, False),  # test internal graph gen
        (True, False, False, False, True, True),  # test wigner cuda
        (True, True, True, False, True, True),  # test merge but no compile
        (True, True, False, False, True, True),  # test no merge or compile
    ],
)
def test_direct_mole_inference_modes(
    tf32,
    activation_checkpointing,
    merge_mole,
    compile,
    wigner_cuda,
    external_graph_gen,
    direct_mole_checkpoint,
    fake_uma_dataset,
    torch_deterministic,
    compile_reset_state,
):
    direct_mole_checkpoint_pt, _ = direct_mole_checkpoint
    mole_inference(
        InferenceSettings(
            tf32=tf32,
            activation_checkpointing=activation_checkpointing,
            merge_mole=merge_mole,
            compile=compile,
            wigner_cuda=wigner_cuda,
            external_graph_gen=external_graph_gen,
        ),
        direct_mole_checkpoint_pt,
        fake_uma_dataset,
        device="cpu",
    )


@pytest.mark.parametrize(
    "tf32, activation_checkpointing, merge_mole, compile, wigner_cuda, external_graph_gen",
    [
        (False, False, False, False, False, True),  # test external graph gen
        (False, False, False, False, False, False),  # test internal graph gen
        (True, False, False, False, True, True),  # test wigner cuda
        (True, True, True, False, True, True),  # test merge but no compile
        (True, True, False, False, True, True),  # test no merge or compile
    ],
)
def test_conserving_mole_inference_modes(
    tf32,
    activation_checkpointing,
    merge_mole,
    compile,
    wigner_cuda,
    external_graph_gen,
    conserving_mole_checkpoint,
    fake_uma_dataset,
    torch_deterministic,
    compile_reset_state,
):
    conserving_mole_checkpoint_pt, _ = conserving_mole_checkpoint
    mole_inference(
        InferenceSettings(
            tf32=tf32,
            activation_checkpointing=activation_checkpointing,
            merge_mole=merge_mole,
            compile=compile,
            wigner_cuda=wigner_cuda,
            external_graph_gen=external_graph_gen,
        ),
        conserving_mole_checkpoint_pt,
        fake_uma_dataset,
        device="cpu",
    )


@pytest.mark.gpu()
@pytest.mark.parametrize(
    "tf32, activation_checkpointing, merge_mole, compile, wigner_cuda, external_graph_gen",
    [
        (False, False, False, False, False, True),  # test external graph gen
        (False, False, False, False, False, False),  # test internal graph gen
        (True, False, False, False, True, True),  # test wigner cuda
        (True, False, True, True, False, True),  # test compile and merge
        # with acvitation checkpointing
        (True, True, True, True, True, True),  # test external model graph gen + compile
        (True, True, True, False, True, True),  # test merge but no compile
        (True, True, False, False, True, True),  # test no merge or compile
    ],
)
def test_conserving_mole_inference_modes_gpu(
    tf32,
    activation_checkpointing,
    merge_mole,
    compile,
    wigner_cuda,
    external_graph_gen,
    conserving_mole_checkpoint,
    fake_uma_dataset,
    compile_reset_state,
):
    conserving_mole_checkpoint_pt, _ = conserving_mole_checkpoint
    mole_inference(
        InferenceSettings(
            tf32=tf32,
            activation_checkpointing=activation_checkpointing,
            merge_mole=merge_mole,
            compile=compile,
            wigner_cuda=wigner_cuda,
            external_graph_gen=external_graph_gen,
        ),
        conserving_mole_checkpoint_pt,
        fake_uma_dataset,
        device="cuda",
        forces_rtol=5e-2,
    )


# Test the two main modes inference and MD on CPU for direct and convserving
def test_conserving_mole_inference_mode_default(
    conserving_mole_checkpoint,
    fake_uma_dataset,
    torch_deterministic,
    compile_reset_state,
):
    conserving_mole_checkpoint_pt, _ = conserving_mole_checkpoint
    mole_inference(
        inference_settings_default(),
        conserving_mole_checkpoint_pt,
        fake_uma_dataset,
        device="cpu",
    )


def test_conserving_mole_inference_mode_md(
    conserving_mole_checkpoint,
    fake_uma_dataset,
    torch_deterministic,
    compile_reset_state,
):
    conserving_mole_checkpoint_pt, _ = conserving_mole_checkpoint
    mole_inference(
        inference_settings_turbo(),
        conserving_mole_checkpoint_pt,
        fake_uma_dataset,
        device="cpu",
    )


def test_direct_mole_inference_mode_default(
    direct_mole_checkpoint, fake_uma_dataset, torch_deterministic, compile_reset_state
):
    direct_mole_checkpoint_pt, _ = direct_mole_checkpoint
    mole_inference(
        inference_settings_default(),
        direct_mole_checkpoint_pt,
        fake_uma_dataset,
        device="cpu",
    )


def test_direct_mole_inference_mode_md(
    direct_mole_checkpoint, fake_uma_dataset, torch_deterministic, compile_reset_state
):
    direct_mole_checkpoint_pt, _ = direct_mole_checkpoint
    mole_inference(
        inference_settings_turbo(),
        direct_mole_checkpoint_pt,
        fake_uma_dataset,
        device="cpu",
    )


# Test conserving and two main modes on GPU


@pytest.mark.gpu()
def test_conserving_mole_inference_mode_default_gpu(
    conserving_mole_checkpoint, fake_uma_dataset, compile_reset_state
):
    conserving_mole_checkpoint_pt, _ = conserving_mole_checkpoint
    mole_inference(
        inference_settings_default(),
        conserving_mole_checkpoint_pt,
        fake_uma_dataset,
        device="cuda",
        energy_rtol=1e-4,
        forces_rtol=5e-2,
    )


@pytest.mark.gpu()
def test_conserving_mole_inference_mode_md_gpu(
    conserving_mole_checkpoint, fake_uma_dataset, compile_reset_state
):
    conserving_mole_checkpoint_pt, _ = conserving_mole_checkpoint
    mole_inference(
        inference_settings_turbo(),
        conserving_mole_checkpoint_pt,
        fake_uma_dataset,
        device="cuda",
        energy_rtol=1e-4,
        forces_rtol=5e-2,
    )


def mole_inference(
    inference_mode,
    inference_checkpoint_path,
    dataset_dir,
    device,
    energy_rtol=1e-4,
    forces_rtol=1e-4,
):
    db = AseDBDataset(config={"src": os.path.join(dataset_dir, "oc20")})

    sample = AtomicData.from_ase(
        db.get_atoms(0),
        max_neigh=10,
        radius=100,
        r_energy=False,
        r_forces=False,
        r_edges=inference_mode.external_graph_gen,
        r_data_keys=["spin", "charge"],
    )
    sample["dataset"] = "oc20"
    batch = data_list_collater(
        [sample], otf_graph=not inference_mode.external_graph_gen
    )

    predictor_baseline = MLIPPredictUnit(
        inference_checkpoint_path,
        device=device,
        inference_settings=InferenceSettings(
            tf32=False,
            activation_checkpointing=False,
            merge_mole=False,
            compile=False,
            external_graph_gen=inference_mode.external_graph_gen,
        ),
    )
    output_baseline = predictor_baseline.predict(batch.clone())

    predictor = MLIPPredictUnit(
        inference_checkpoint_path, device=device, inference_settings=inference_mode
    )
    model_outputs = [
        predictor.predict(batch.clone()),
        predictor.predict(batch.clone()),
    ]  # run it twice to make sure merge etc work correct

    for output in model_outputs:
        for k in output_baseline:
            print(
                f"{k}: max rtol detected",
                ((output_baseline[k] - output[k]) / (output[k] + 1e-12))
                .abs()
                .max()
                .item(),
            )
            assert (
                output_baseline[k]
                .isclose(output[k], rtol=energy_rtol if "energy" in k else forces_rtol)
                .all()
            )
            assert output[k].device.type == device
            assert output_baseline[k].device.type == device


# example how to use checkpoint fixtures
def test_checkpoints_work(conserving_mole_checkpoint, direct_mole_checkpoint):
    conserving_inference_checkpoint_pt, conserving_train_state_yaml = (
        conserving_mole_checkpoint
    )
    direct_inference_checkpoint_pt, direct_train_state_yaml = direct_mole_checkpoint


@pytest.mark.gpu()
def test_mole_merge_inference_fail(conserving_mole_checkpoint, fake_uma_dataset):
    conserving_inference_checkpoint_pt, conserving_train_state_yaml = (
        conserving_mole_checkpoint
    )
    inference_mode = InferenceSettings(
        tf32=False,
        activation_checkpointing=False,
        merge_mole=True,
        compile=False,
        external_graph_gen=True,
    )

    db = AseDBDataset(config={"src": os.path.join(fake_uma_dataset, "oc20")})

    a2g = partial(
        AtomicData.from_ase,
        max_neigh=10,
        radius=100,
        r_energy=False,
        r_forces=False,
        r_edges=inference_mode.external_graph_gen,
        r_data_keys=["spin", "charge"],
    )

    sample = a2g(db.get_atoms(0), task_name="oc20")
    batch = data_list_collater(
        [sample], otf_graph=not inference_mode.external_graph_gen
    )
    device = "cuda"
    predictor = MLIPPredictUnit(
        conserving_inference_checkpoint_pt,
        device=device,
        inference_settings=inference_mode,
    )
    _ = predictor.predict(batch.clone())

    sample = a2g(db.get_atoms(1), task_name="oc20")
    batch = data_list_collater(
        [sample], otf_graph=not inference_mode.external_graph_gen
    )
    with pytest.raises(AssertionError):
        _ = predictor.predict(batch.clone())

    sample = a2g(db.get_atoms(0), task_name="not-oc20")
    batch = data_list_collater(
        [sample], otf_graph=not inference_mode.external_graph_gen
    )
    with pytest.raises(AssertionError):
        _ = predictor.predict(batch.clone())

    sample = a2g(db.get_atoms(0), task_name="oc20")
    batch = data_list_collater(
        [sample], otf_graph=not inference_mode.external_graph_gen
    )
    _ = predictor.predict(batch.clone())


def test_mole_merge_on_non_mole_model(direct_checkpoint, fake_uma_dataset):
    direct_non_mole_inference_checkpoint_pt, _ = direct_checkpoint
    inference_mode = InferenceSettings(
        tf32=False,
        activation_checkpointing=False,
        merge_mole=True,
        compile=False,
        external_graph_gen=True,
    )

    db = AseDBDataset(config={"src": os.path.join(fake_uma_dataset, "oc20")})

    a2g = partial(
        AtomicData.from_ase,
        max_neigh=10,
        radius=100,
        r_energy=False,
        r_forces=False,
        r_edges=inference_mode.external_graph_gen,
        r_data_keys=["spin", "charge"],
    )

    sample = a2g(db.get_atoms(0), task_name="oc20")
    batch = data_list_collater(
        [sample], otf_graph=not inference_mode.external_graph_gen
    )
    device = "cpu"
    predictor = MLIPPredictUnit(
        direct_non_mole_inference_checkpoint_pt,
        device=device,
        inference_settings=inference_mode,
    )
    _ = predictor.predict(batch.clone())


def get_batched_system(
    num_atoms_per_system: int,
    systems: int,
    lattice_constant: float = 3.8,
):
    atom_systems = []
    for _ in range(systems):
        atoms = build.bulk("C", "fcc", a=lattice_constant)
        n_cells = int(np.ceil(np.cbrt(num_atoms_per_system)))
        atoms = atoms.repeat((n_cells, n_cells, n_cells))
        indices = np.random.choice(len(atoms), num_atoms_per_system, replace=False)
        sampled_atoms = atoms[indices]
        ad = AtomicData.from_ase(sampled_atoms)
        ad.dataset = "oc20"
        ad.pos.requires_grad = True
        atom_systems.append(ad)
    return atomicdata_list_to_batch(atom_systems)


def reset_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Put this on GPU for now until the atan2 bug is fixed
@pytest.mark.gpu()
@pytest.mark.parametrize(
    # try very small chunk size to force ac to chunk
    "chunk_size, nsystems, natoms, merge_mole",
    [
        # disable these for now until we fix the atan2 issue
        (1024, 3, 100, False),  # batched + no merge
        (1024 * 128, 5, 1000, False),  # batched + no merge
        (1000, 1, 1000, False),  # unbatch + no merge
        (1024 * 128, 1, 1000, False),  # unbatch + no merge
        (1024 * 128, 1, 10000, False),  # unbatch + no merge
        (1024, 1, 100, True),  # unbatched + merge mole
    ],
)
def test_ac_with_chunking_and_batching(
    conserving_mole_checkpoint,
    monkeypatch,
    chunk_size,
    nsystems,
    natoms,
    merge_mole,
):
    monkeypatch.setattr(
        "fairchem.core.models.uma.escn_md.ESCNMD_DEFAULT_EDGE_CHUNK_SIZE", chunk_size
    )
    conserving_mole_checkpoint_pt, _ = conserving_mole_checkpoint
    ifs = InferenceSettings(
        tf32=False,
        activation_checkpointing=False,
        merge_mole=merge_mole,
        compile=False,
        wigner_cuda=False,
        external_graph_gen=False,
        internal_graph_gen_version=2,
    )
    reset_seeds(0)
    batch = get_batched_system(natoms, nsystems)
    device = "cuda"
    predictor_noac = MLIPPredictUnit(
        conserving_mole_checkpoint_pt,
        device=device,
        inference_settings=ifs,
    )
    reset_seeds(0)
    result_no_ac = predictor_noac.predict(batch.clone())
    ifs.activation_checkpointing = True
    predictor_ac = MLIPPredictUnit(
        conserving_mole_checkpoint_pt,
        device=device,
        inference_settings=ifs,
    )
    reset_seeds(0)
    result_ac = predictor_ac.predict(batch.clone())
    assert torch.allclose(result_ac["energy"], result_no_ac["energy"])
    assert torch.allclose(
        result_ac["forces"], result_no_ac["forces"], rtol=1e-5, atol=1e-5
    )
