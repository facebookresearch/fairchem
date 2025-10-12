"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fairchem.core.common import distutils
from fairchem.core.components.runner import Runner
from fairchem.core.datasets import data_list_collater

if TYPE_CHECKING:
    from torchtnt.framework import PredictUnit
    from torchtnt.framework.callback import Callback


class IndexedDataset(Dataset):
    """Wrap a dataset to return (data, idx) while delegating API to the base dataset.

    Ensures compatibility with samplers expecting methods like get_metadata.
    """

    def __init__(self, dataset: Dataset):
        self.base = dataset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        return self.base[idx], idx

    # delegate missing attributes/methods to base
    def __getattr__(self, name: str):
        return getattr(self.base, name)

    # explicitly expose get_metadata for samplers
    def get_metadata(self, attr, idx):
        return self.base.get_metadata(attr, idx)


class FixedDatasetCollater:
    def __init__(self, dataset_name: str, otf_graph: bool = True):
        self.dataset_name = dataset_name
        self.otf_graph = otf_graph

    def __call__(self, data_list: list[Any]):
        # support (data, idx) tuples from IndexedDataset
        if len(data_list) > 0 and isinstance(data_list[0], (tuple, list)):
            data_only, indices = zip(*data_list)
        else:
            data_only = data_list
            indices = None

        for d in data_only:
            d.dataset = self.dataset_name
        batch = data_list_collater(list(data_only), otf_graph=self.otf_graph)
        if indices is not None:
            # attach indices for callback to use
            batch.indices = np.asarray(indices, dtype=int)
        return batch


class RecordingPredictUnit:
    """A thin wrapper over a PredictUnit that records the last batch and predictions.

    This enables callbacks to access batch ids/natoms and model outputs.
    """

    def __init__(self, inner: PredictUnit):
        self.inner = inner
        self.last_batch = None
        self.last_preds = None

    # delegate attributes for transparency
    def __getattr__(self, name):
        return getattr(self.inner, name)

    def predict_step(self, state, data):
        self.last_batch = data
        if hasattr(self.inner, "predict_step"):
            preds = self.inner.predict_step(state, data)
        else:
            preds = self.inner.predict(data)
        self.last_preds = preds
        return preds


class NPZWriterCallback:
    def __init__(self, results_dir: str, results_filename: str):
        self.results_dir = results_dir
        self.results_filename = results_filename
        self._batches: list[dict[str, Any]] = []
        self._dataset = None

    def on_predict_epoch_start(self, state, unit):
        self._batches = []
        # try to get the underlying dataset for id resolution
        try:
            dl = state.predict_state.dataloader
            ds = getattr(dl, "dataset", None)
            # if wrapped with IndexedDataset, use .base
            self._dataset = getattr(ds, "base", ds)
        except Exception:
            self._dataset = None

    def on_predict_step_end(self, state, unit: RecordingPredictUnit):
        batch = unit.last_batch
        preds = unit.last_preds
        # resolve ids using atoms.info["source"] when possible
        ids_arr = None
        if hasattr(batch, "indices") and self._dataset is not None:
            ids = []
            for idx in batch.indices.tolist():
                try:
                    atoms = self._dataset.get_atoms(idx)
                    ids.append(atoms.info.get("source", None))
                except Exception:
                    ids.append(None)
            ids_arr = np.asarray(ids, dtype=object)
        else:
            ids_arr = np.asarray(getattr(batch, "sid", []), dtype=object)

        out: dict[str, Any] = {
            "ids": ids_arr,
            "natoms": batch.natoms.detach().cpu().numpy().astype(int).reshape(-1),
        }
        if "energy" in preds and preds["energy"] is not None:
            out["energy"] = preds["energy"].detach().cpu().numpy()
        else:
            out["energy"] = np.zeros((0,), dtype=float)
        if "forces" in preds and preds["forces"] is not None:
            out["forces"] = preds["forces"].detach().cpu().numpy()
        else:
            out["forces"] = np.zeros((0, 3), dtype=float)
        self._batches.append(out)

    def on_predict_end(self, state, unit):
        gathered = distutils.gather_objects(self._batches)
        if not distutils.is_master():
            return
        combined: list[dict[str, Any]] = []
        for part in gathered:
            if part is None:
                continue
            combined.extend(part)
        ids_list = np.concatenate([b["ids"] for b in combined])
        energy_list = np.concatenate([b["energy"] for b in combined])
        forces_list = np.concatenate([b["forces"] for b in combined])
        natoms_list = np.concatenate([b["natoms"] for b in combined])
        os.makedirs(self.results_dir, exist_ok=True)
        out_path = os.path.join(self.results_dir, self.results_filename)
        np.savez_compressed(
            out_path,
            ids=ids_list.astype("<U169"),
            energy=energy_list,
            forces=forces_list,
            natoms=natoms_list,
        )


class PredictRunner(Runner):
    """Run batched predictions using TorchTNT with callbacks.

    Writes a single npz file containing ids, energy, forces, and natoms as
    a list of per-batch arrays (object arrays), letting TNT handle distribution.
    """

    result_glob_pattern: ClassVar[str] = "*.npz"

    def __init__(
        self,
        dataloader: DataLoader,
        predict_unit: PredictUnit,
        callbacks: list[Callback] | None = None,
        results_filename: str | None = None,
    ):
        self.dataloader = dataloader
        self.predict_unit = predict_unit
        self.callbacks = callbacks if callbacks is not None else []
        self.results_filename = results_filename

    def run(self) -> None:
        from torchtnt.framework.predict import predict as tnt_predict

        writer = NPZWriterCallback(
            results_dir=self.job_config.metadata.results_dir,
            results_filename=(
                self.results_filename
                if self.results_filename is not None
                else f"{self.job_config.run_name}_predictions.npz"
            ),
        )
        unit = RecordingPredictUnit(self.predict_unit)
        tnt_predict(unit, self.dataloader, callbacks=[*self.callbacks, writer])

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        return True

    def load_state(self, checkpoint_location: str | None) -> None:
        return None