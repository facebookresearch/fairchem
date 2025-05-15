"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os

import pytest
import torch

from fairchem.core.datasets import data_list_collater
from fairchem.core.datasets.ase_datasets import AseDBDataset
from fairchem.core.preprocessing.atoms_to_graphs import AtomsToGraphs
from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
    inference_settings_default,
    inference_settings_turbo,
)
from fairchem.core.units.mlip_unit.mlip_unit import (
    MLIPPredictUnit,
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
    conserving_mole_checkpoint, fake_uma_dataset, torch_deterministic
):
    conserving_mole_checkpoint_pt, _ = conserving_mole_checkpoint
    mole_inference(
        inference_settings_default(),
        conserving_mole_checkpoint_pt,
        fake_uma_dataset,
        device="cpu",
    )


def test_conserving_mole_inference_mode_md(
    conserving_mole_checkpoint, fake_uma_dataset, torch_deterministic
):
    conserving_mole_checkpoint_pt, _ = conserving_mole_checkpoint
    mole_inference(
        inference_settings_turbo(),
        conserving_mole_checkpoint_pt,
        fake_uma_dataset,
        device="cpu",
    )


def test_direct_mole_inference_mode_default(
    direct_mole_checkpoint, fake_uma_dataset, torch_deterministic
):
    direct_mole_checkpoint_pt, _ = direct_mole_checkpoint
    mole_inference(
        inference_settings_default(),
        direct_mole_checkpoint_pt,
        fake_uma_dataset,
        device="cpu",
    )


def test_direct_mole_inference_mode_md(
    direct_mole_checkpoint, fake_uma_dataset, torch_deterministic
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
    conserving_mole_checkpoint, fake_uma_dataset
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
    conserving_mole_checkpoint, fake_uma_dataset
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
    torch.compiler.reset()
    db = AseDBDataset(config={"src": os.path.join(dataset_dir, "oc20")})

    a2g = AtomsToGraphs(
        max_neigh=10,
        radius=100,
        r_energy=False,
        r_forces=False,
        r_distances=False,
        r_edges=inference_mode.external_graph_gen,
        r_pbc=True,
        r_data_keys=["spin", "charge"],
    )

    sample = a2g.convert(db.get_atoms(0))
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

    a2g = AtomsToGraphs(
        max_neigh=10,
        radius=100,
        r_energy=False,
        r_forces=False,
        r_distances=False,
        r_edges=inference_mode.external_graph_gen,
        r_pbc=True,
        r_data_keys=["spin", "charge"],
    )

    sample = a2g.convert(db.get_atoms(0))
    sample["dataset"] = "oc20"
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

    sample = a2g.convert(db.get_atoms(1))
    sample["dataset"] = "oc20"
    batch = data_list_collater(
        [sample], otf_graph=not inference_mode.external_graph_gen
    )
    with pytest.raises(AssertionError):
        _ = predictor.predict(batch.clone())

    sample = a2g.convert(db.get_atoms(0))
    sample["dataset"] = "not-oc20"
    batch = data_list_collater(
        [sample], otf_graph=not inference_mode.external_graph_gen
    )
    with pytest.raises(AssertionError):
        _ = predictor.predict(batch.clone())

    sample = a2g.convert(db.get_atoms(0))
    sample["dataset"] = "oc20"
    batch = data_list_collater(
        [sample], otf_graph=not inference_mode.external_graph_gen
    )
    _ = predictor.predict(batch.clone())
