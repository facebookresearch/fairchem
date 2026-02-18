"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

from ._load_utils import _load_from_config


class ElementReferences(nn.Module):
    def __init__(
        self,
        element_references: torch.Tensor,
    ):
        """
        Args:
            element_references (Tensor): tensor with reference value for each element
        """
        super().__init__()
        self.register_buffer(name="element_references", tensor=element_references)

    @staticmethod
    def compute_references(batch, tensor, elem_refs, operation):
        assert tensor.shape[0] == len(batch.natoms)
        batch_idx = getattr(batch, "batch_full", batch.batch)
        atomic_numbers = getattr(batch, "atomic_numbers_full", batch.atomic_numbers)
        with torch.autocast(elem_refs.device.type, enabled=False):
            refs = torch.zeros(
                tensor.shape, dtype=elem_refs.dtype, device=tensor.device
            ).scatter_reduce(
                0,
                batch_idx,
                atomic_numbers,
                reduce="sum",
            )
            if operation == "subtract":
                return tensor - refs
            elif operation == "add":
                return tensor + refs
            else:
                raise ValueError(f"Unknown operation: {operation}")

    def apply_refs(
        self,
        batch: AtomicData,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        return self.compute_references(
            batch,
            tensor,
            self.element_references,
            operation="subtract",
        )

    def undo_refs(
        self,
        batch: AtomicData,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        return self.compute_references(
            batch,
            tensor,
            self.element_references,
            operation="add",
        )


def create_element_references(
    file: str | Path | None = None,
    state_dict: dict | None = None,
) -> ElementReferences:
    """Create an element reference module.

    Args:
        file (str or Path): path to pt or npz file
        state_dict (dict): a state dict of a element reference module

    Returns:
        ElementReferences
    """
    if file is not None and state_dict is not None:
        logging.warning(
            "Both a file and a state_dict for element references was given."
            "The references will be read from the file and the provided state_dict will be ignored."
        )

    # path takes priority if given
    if file is not None:
        extension = Path(file).suffix
        if extension == ".pt":
            # try to load a pt file
            state_dict = torch.load(file)
        elif extension == ".npz":
            state_dict = {}
            with np.load(file) as values:
                # legacy linref files
                if "coeff" in values:
                    state_dict["element_references"] = torch.tensor(values["coeff"])
                else:
                    state_dict["element_references"] = torch.tensor(
                        values["element_references"]
                    )
        else:
            raise RuntimeError(
                f"Element references file with extension '{extension}' is not supported."
            )

    if "element_references" not in state_dict:
        raise RuntimeError("Unable to load linear element references!")

    return ElementReferences(element_references=state_dict["element_references"])


@torch.no_grad()
def fit_linear_references(
    targets: list[str],
    dataset: Dataset,
    batch_size: int,
    num_batches: int | None = None,
    num_workers: int = 0,
    max_num_elements: int = 118,
    log_metrics: bool = True,
    use_numpy: bool = True,
    driver: str | None = None,
    shuffle: bool = True,
    seed: int = 0,
) -> dict[str, ElementReferences]:
    """Fit a set of element references for a list of targets using a given number of batches.

    Args:
        targets: list of target names
        dataset: data set to fit element references with
        batch_size: size of batch
        num_batches: number of batches to use in fit. If not given will use all batches
        num_workers: number of workers to use in data loader.
            Note setting num_workers > 1 leads to finicky multiprocessing issues when using this function
            in distributed mode. The issue has to do with pickling the functions in load_references_from_config
            see function below...
        max_num_elements: max number of elements in dataset. If not given will use an ambitious value of 118
        log_metrics: if true will compute MAE, RMSE and R2 score of fit and log.
        use_numpy: use numpy.linalg.lstsq instead of torch. This tends to give better solutions.
        driver: backend used to solve linear system. See torch.linalg.lstsq docs. Ignored if use_numpy=True
        shuffle: whether to shuffle when loading the dataset
        seed: random seed used to shuffle the sampler if shuffle=True

    Returns:
        dict of fitted ElementReferences objects
    """
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=atomicdata_list_to_batch,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )

    num_batches = num_batches if num_batches is not None else len(data_loader)
    if num_batches > len(data_loader):
        logging.warning(
            f"The given num_batches {num_batches} is larger than total batches of size {batch_size} in dataset. "
            f"num_batches will be ignored and the whole dataset will be used."
        )
        num_batches = len(data_loader)

    max_num_elements += 1  # + 1 since H starts at index 1
    # solving linear system happens on CPU, which allows handling poorly conditioned and
    # rank deficient matrices, unlike torch lstsq on GPU
    composition_matrix = torch.zeros(
        num_batches * batch_size,
        max_num_elements,
    )

    target_vectors = {
        target: torch.zeros(num_batches * batch_size) for target in targets
    }

    logging.info(
        f"Fitting linear references using {num_batches * batch_size} samples in {num_batches} "
        f"batches of size {batch_size}."
    )
    for i, batch in tqdm(
        enumerate(data_loader), total=num_batches, desc="Fitting linear references"
    ):
        if i == 0:
            assert all(
                len(batch[target].squeeze().shape) == 1 for target in targets
            ), "element references can only be used for scalar targets"
        elif i == num_batches:
            break

        next_batch_size = len(batch) if i == len(data_loader) - 1 else batch_size
        for target in targets:
            target_vectors[target][
                i * batch_size : i * batch_size + next_batch_size
            ] = batch[target].to(torch.float64)
        for j, data in enumerate(batch.batch_to_atomicdata_list()):
            composition_matrix[i * batch_size + j] = torch.bincount(
                data.atomic_numbers.int(),
                minlength=max_num_elements,
            ).to(torch.float64)

    # reduce the composition matrix to only features that are non-zero to improve rank
    mask = composition_matrix.sum(axis=0) != 0.0
    reduced_composition_matrix = composition_matrix[:, mask]
    elementrefs = {}

    for target in targets:
        coeffs = torch.zeros(max_num_elements)

        if use_numpy:
            solution = torch.tensor(
                np.linalg.lstsq(
                    reduced_composition_matrix.numpy(),
                    target_vectors[target].numpy(),
                    rcond=None,
                )[0]
            )
        else:
            lstsq = torch.linalg.lstsq(
                reduced_composition_matrix, target_vectors[target], driver=driver
            )
            solution = lstsq.solution

        coeffs[mask] = solution
        elementrefs[target] = ElementReferences(coeffs)

        if log_metrics is True:
            y = target_vectors[target]
            y_pred = torch.matmul(reduced_composition_matrix, solution)
            y_mean = target_vectors[target].mean()
            N = len(target_vectors[target])
            ss_res = ((y - y_pred) ** 2).sum()
            ss_tot = ((y - y_mean) ** 2).sum()
            mae = (abs(y - y_pred)).sum() / N
            rmse = (((y - y_pred) ** 2).sum() / N).sqrt()
            r2 = 1 - (ss_res / ss_tot)
            logging.info(
                f"Training accuracy metrics for fitted linear element references: mae={mae}, rmse={rmse}, r2 score={r2}"
            )

    return elementrefs


def load_references_from_config(
    config: dict[str, Any],
    dataset: Dataset,
    seed: int = 0,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, ElementReferences]:
    """Create a dictionary with element references from a config."""
    return _load_from_config(
        config,
        "element_references",
        fit_linear_references,
        create_element_references,
        dataset,
        checkpoint_dir,
        seed=seed,
    )
