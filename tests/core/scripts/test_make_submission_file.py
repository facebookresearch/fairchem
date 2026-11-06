"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from fairchem.core.scripts.make_submission_file import SPLITS, main


@pytest.mark.parametrize("mode", ["predictions", "relaxations", "hybrid"])
@pytest.mark.parametrize("dataset", ["OC20", "OC22"])
def test_submission_preserves_split_ids_and_final_energies(tmp_path, mode, dataset):
    args = Namespace(
        dataset=dataset,
        out_path=str(tmp_path / "output"),
        is2re_relaxations=mode != "predictions",
        hybrid=mode == "hybrid",
    )
    for i, split in enumerate(SPLITS[dataset]):
        if mode == "relaxations":
            directory = tmp_path / split
            directory.mkdir()
            frames = []
            for energy in (99.0, float(i)):
                atoms = Atoms("H2")
                atoms.calc = SinglePointCalculator(atoms, energy=energy)
                frames.append(atoms)
            write(directory / "123.traj", frames)
            setattr(args, split, str(directory))
        else:
            path = tmp_path / f"{split}.npz"
            np.savez(path, ids=np.array(["123_0"]), energy=np.array([float(i)]))
            setattr(args, split, str(path))
    main(args)
    assert args.out_path.endswith(".npz")
    with np.load(args.out_path) as result:
        assert set(result.files) == {
            f"{split}_{key}" for split in SPLITS[dataset] for key in ("ids", "energy")
        }
        for i, split in enumerate(SPLITS[dataset]):
            assert result[f"{split}_ids"].tolist() == [
                "123_0" if mode == "predictions" else "123"
            ]
            assert result[f"{split}_energy"].tolist() == [float(i)]


def test_submission_rejects_missing_splits(tmp_path):
    args = Namespace(dataset="OC22", out_path=str(tmp_path / "output"), id="missing")
    with pytest.raises(AssertionError, match="Missing ood split"):
        main(args)
