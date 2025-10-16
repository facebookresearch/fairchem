"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import tempfile
from itertools import product
from random import choice
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.io import write
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from syrupy.extensions.amber import AmberSnapshotExtension

from fairchem.core.datasets.ase_datasets import AseDBDataset, AseReadDataset
from fairchem.core.units.mlip_unit.mlip_unit import (
    UNIT_INFERENCE_CHECKPOINT,
    UNIT_RESUME_CONFIG,
)
from tests.core.testing_utils import launch_main
from tests.core.units.mlip_unit.create_fake_dataset import (
    create_fake_uma_dataset,
)

if TYPE_CHECKING:
    from syrupy.types import SerializableData

DEFAULT_RTOL = 1.0e-03
DEFAULT_ATOL = 1.0e-03


class Approx:
    """
    Wrapper object for approximately compared numpy arrays.
    """

    def __init__(
        self,
        data: np.ndarray | list,
        *,
        rtol: float | None = None,
        atol: float | None = None,
    ) -> None:
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            raise TypeError(f"Cannot convert {type(data)} to np.array")

        self.rtol: float = rtol if rtol is not None else DEFAULT_RTOL
        self.atol: float = atol if atol is not None else DEFAULT_ATOL
        self.tol_repr = True

    def __repr__(self) -> str:
        data = np.array_repr(self.data)
        data = "\n".join(f"\t{line}" for line in data.splitlines())
        tol_repr = ""
        if self.tol_repr:
            tol_repr = f", \n\trtol={self.rtol}, \n\tatol={self.atol}"
        return f"Approx(\n{data}{tol_repr}\n)"


class _ApproxNumpyFormatter:
    def __init__(self, data) -> None:
        self.data = data

    def __repr__(self) -> str:
        return Approx(
            self.data.expected,
            rtol=self.data.rel,
            atol=self.data.abs,
        ).__repr__()


def _try_parse_approx(data: SerializableData) -> Approx | None:
    """
    Parse the string representation of an Approx object.
    We can just use eval here, since we know the string is safe.
    """
    if not isinstance(data, str):
        return None

    data = data.strip()
    if not data.startswith("Approx("):
        return None

    approx = eval(
        data.replace("dtype=", "dtype=np."),
        {"Approx": Approx, "np": np},
        {"array": np.array},
    )
    if not isinstance(approx, Approx):
        return None

    return approx


class ApproxExtension(AmberSnapshotExtension):
    """
    By default, syrupy uses the __repr__ of the expected (snapshot) and actual values
    to serialize them into strings. Then, it compares the strings to see if they match.

    However, this behavior is not ideal for comparing floats/ndarrays. For example,
    if we have a snapshot with a float value of 0.1, and the actual value is 0.10000000000000001,
    then the strings will not match, even though the values are effectively equal.

    To work around this, we override the serialize method to seralize the expected value
    into a special representation. Then, we override the matches function (which originally does a
    simple string comparison) to parse the expected and actual values into numpy arrays.
    Finally, we compare the arrays using np.allclose.
    """

    def matches(
        self,
        *,
        serialized_data: SerializableData,
        snapshot_data: SerializableData,
    ) -> bool:
        # if both serialized_data and snapshot_data are serialized Approx objects,
        # then we can load them as numpy arrays and compare them using np.allclose
        serialized_approx = _try_parse_approx(serialized_data)
        snapshot_approx = _try_parse_approx(snapshot_data)
        if serialized_approx is not None and snapshot_approx is not None:
            return np.allclose(
                snapshot_approx.data,
                serialized_approx.data,
                rtol=serialized_approx.rtol,
                atol=serialized_approx.atol,
            )

        return super().matches(
            serialized_data=serialized_data, snapshot_data=snapshot_data
        )

    def serialize(self, data, **kwargs):
        # we override the existing serialization behavior
        # of the `pytest.approx()` object to serialize it into a special string.
        if isinstance(data, type(pytest.approx(np.array(0.0)))):
            return super().serialize(_ApproxNumpyFormatter(data), **kwargs)
        elif isinstance(data, type(pytest.approx(0.0))):
            raise NotImplementedError("Scalar approx not implemented yet")
        return super().serialize(data, **kwargs)


@pytest.fixture()
def snapshot(snapshot):
    return snapshot.use_extension(ApproxExtension)


@pytest.fixture()
def torch_deterministic():
    # Setup
    torch.use_deterministic_algorithms(True)
    yield True  # Usability: prints `torch_deterministic=True` if a test fails
    # Tear down
    torch.use_deterministic_algorithms(False)


@pytest.fixture(scope="session")
def dummy_element_refs():
    # create some dummy elemental energies from ionic radii (ignore deuterium and tritium included in pmg)
    return np.concatenate(
        [[0], [e.average_ionic_radius for e in Element if e.name not in ("D", "T")]]
    )


@pytest.fixture(scope="session")
def dummy_binary_dataset_path(tmpdir_factory, dummy_element_refs):
    # a dummy dataset with binaries with energy that depends on composition only plus noise
    # Limit to first 83 elements (up to Bismuth) to avoid CUDA indexing errors with rare/synthetic elements
    common_elements = [Element.from_Z(z) for z in range(1, 84)]  # H to Bi
    all_binaries = list(product(common_elements, repeat=2))
    rng = np.random.default_rng(seed=0)

    tmpdir = tmpdir_factory.mktemp("dataset")
    with connect(str(tmpdir / "dummy.aselmdb")) as db:
        for i in range(10):
            elements = choice(all_binaries)
            structure = Structure.from_prototype("cscl", species=elements, a=2.0)
            energy = (
                sum(e.average_ionic_radius for e in elements)
                + 0.05 * rng.random() * dummy_element_refs.mean()
            )
            atoms = structure.to_ase_atoms()
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=energy,
                forces=rng.random((2, 3)),
                stress=rng.random((3, 3)),
            )
            # write to the lmdb file
            db.write(atoms, data={"sid": f"structure_{i}"})

            # write it as a cif file as well
            write(str(tmpdir / f"structure_{i}.cif"), atoms)

    return tmpdir


@pytest.fixture(scope="session", params=["asedb", "cif"])
def dummy_binary_dataset(dummy_binary_dataset_path, request):
    config = dict(src=str(dummy_binary_dataset_path))

    if request.param == "cif":
        config["pattern"] = "*.cif"
        return AseReadDataset(config=config)
    else:
        return AseDBDataset(config=config)


@pytest.fixture(scope="session")
def dummy_binary_db_dataset(dummy_binary_dataset_path):
    config = dict(src=str(dummy_binary_dataset_path))
    return AseDBDataset(config=config)


@pytest.fixture(autouse=True)
def run_around_tests():
    # If debugging GPU memory issues, uncomment this print statement
    # to get full GPU memory allocations before each test runs
    # print(torch.cuda.memory_summary())
    yield
    torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def direct_mole_checkpoint(fake_uma_dataset):
    # first train to completion
    temp_dir = tempfile.mkdtemp()
    timestamp_id = "12345"
    device = "CPU"

    sys_args = [
        "--config",
        "tests/core/units/mlip_unit/test_mlip_train.yaml",
        "num_experts=8",
        "checkpoint_every=10000",
        "datasets=aselmdb",
        f"+job.run_dir={temp_dir}",
        f"datasets.data_root_dir={fake_uma_dataset}",
        f"job.device_type={device}",
        f"+job.timestamp_id={timestamp_id}",
        "optimizer=savegrad",
        "max_steps=2",
        "max_epochs=null",
        "expected_loss=null",
        "act_type=gate",
        "ff_type=spectral",
    ]
    launch_main(sys_args)

    # Now resume from checkpoint_step and should get the same result
    # TODO, should get the run config and get checkpoint location from there
    checkpoint_dir = os.path.join(temp_dir, timestamp_id, "checkpoints", "step_0")
    checkpoint_state_yaml = os.path.join(checkpoint_dir, UNIT_RESUME_CONFIG)
    inference_checkpoint_pt = os.path.join(checkpoint_dir, UNIT_INFERENCE_CHECKPOINT)
    assert os.path.isdir(checkpoint_dir)
    assert os.path.isfile(checkpoint_state_yaml)
    assert os.path.isfile(inference_checkpoint_pt)

    return inference_checkpoint_pt, checkpoint_state_yaml


@pytest.fixture(scope="session")
def direct_checkpoint(fake_uma_dataset):
    # first train to completion
    temp_dir = tempfile.mkdtemp()
    timestamp_id = "12345"
    device = "CPU"

    sys_args = [
        "--config",
        "tests/core/units/mlip_unit/test_mlip_train.yaml",
        "num_experts=0",
        "checkpoint_every=10000",
        "datasets=aselmdb",
        f"+job.run_dir={temp_dir}",
        f"datasets.data_root_dir={fake_uma_dataset}",
        f"job.device_type={device}",
        f"+job.timestamp_id={timestamp_id}",
        "optimizer=savegrad",
        "max_steps=2",
        "max_epochs=null",
        "expected_loss=null",
        "act_type=gate",
        "ff_type=spectral",
        # "max_neighbors=300"
    ]
    launch_main(sys_args)

    # Now resume from checkpoint_step and should get the same result
    # TODO, should get the run config and get checkpoint location from there
    checkpoint_dir = os.path.join(temp_dir, timestamp_id, "checkpoints", "step_0")
    checkpoint_state_yaml = os.path.join(checkpoint_dir, UNIT_RESUME_CONFIG)
    inference_checkpoint_pt = os.path.join(checkpoint_dir, UNIT_INFERENCE_CHECKPOINT)
    assert os.path.isdir(checkpoint_dir)
    assert os.path.isfile(checkpoint_state_yaml)
    assert os.path.isfile(inference_checkpoint_pt)

    return inference_checkpoint_pt, checkpoint_state_yaml


@pytest.fixture(scope="session")
def conserving_mole_checkpoint(fake_uma_dataset):
    # first train to completion
    temp_dir = tempfile.mkdtemp()
    timestamp_id = "12345"
    device = "CPU"

    sys_args = [
        "--config",
        "tests/core/units/mlip_unit/test_mlip_train_conserving.yaml",
        "num_experts=8",
        "heads.energyandforcehead.module=fairchem.core.models.uma.escn_moe.DatasetSpecificSingleHeadWrapper",
        "checkpoint_every=10000",
        "datasets=aselmdb_conserving",
        f"+job.run_dir={temp_dir}",
        f"datasets.data_root_dir={fake_uma_dataset}",
        f"job.device_type={device}",
        f"+job.timestamp_id={timestamp_id}",
        "optimizer=savegrad",
        "max_steps=2",
        "max_epochs=null",
        "expected_loss=null",
        "act_type=gate",
        "ff_type=spectral",
    ]
    launch_main(sys_args)

    # Now resume from checkpoint_step and should get the same result
    # TODO, should get the run config and get checkpoint location from there
    checkpoint_dir = os.path.join(temp_dir, timestamp_id, "checkpoints", "step_0")
    checkpoint_state_yaml = os.path.join(checkpoint_dir, UNIT_RESUME_CONFIG)
    inference_checkpoint_pt = os.path.join(checkpoint_dir, UNIT_INFERENCE_CHECKPOINT)
    assert os.path.isdir(checkpoint_dir)
    assert os.path.isfile(checkpoint_state_yaml)
    assert os.path.isfile(inference_checkpoint_pt)

    return inference_checkpoint_pt, checkpoint_state_yaml


@pytest.fixture(scope="session")
def fake_uma_dataset():
    with tempfile.TemporaryDirectory() as tempdirname:
        datasets_yaml = create_fake_uma_dataset(tempdirname)
        yield datasets_yaml
