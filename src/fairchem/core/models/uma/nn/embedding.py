"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn

from .radial import RadialMLP

if TYPE_CHECKING:
    from .execution_backends import ExecutionBackend


class EdgeDegreeEmbedding(torch.nn.Module):
    """

    Args:
        sphere_channels (int):      Number of spherical channels

        lmax (int):                 degrees (l)
        mmax (int):                 orders (m)

        max_num_elements (int):     Maximum number of atomic numbers
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                        The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
        use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance for edge scalar features

        rescale_factor (float):     Rescale the sum aggregation
        cutoff (float):             Cutoff distance for the radial function

        mappingReduced (CoefficientMapping): Class to convert l and m indices once node embedding is rotated
        backend (ExecutionBackend): Execution backend for edge_degree_scatter
    """

    def __init__(
        self,
        sphere_channels: int,
        lmax: int,
        mmax: int,
        edge_channels_list,
        rescale_factor,
        mappingReduced,
        # Enables activation checkpointing in size of
        # activation_checkpoint_chunk_size edge blocks
        activation_checkpoint_chunk_size: int | None,
        backend: ExecutionBackend,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mappingReduced
        self.activation_checkpoint_chunk_size = activation_checkpoint_chunk_size
        self.backend = backend

        self.m_0_num_coefficients: int = self.mappingReduced.m_size[0]
        self.m_all_num_coefficents: int = len(self.mappingReduced.l_harmonic)

        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        edge_channels_list = copy.deepcopy(edge_channels_list)

        # Embedding function of distance
        edge_channels_list.append(self.m_0_num_coefficients * self.sphere_channels)
        self.rad_func = RadialMLP(edge_channels_list)

        self.rescale_factor = rescale_factor

    def forward_chunk(
        self,
        x,
        x_edge,
        edge_index,
        wigner_inv_envelope,
        node_offset=0,
    ):
        radial = self.rad_func(x_edge)

        return self.backend.edge_degree_scatter(
            x,
            radial,
            wigner_inv_envelope,
            edge_index,
            self.m_0_num_coefficients,
            self.sphere_channels,
            self.rescale_factor,
            node_offset,
        )

    def forward(
        self,
        x,
        x_edge,
        edge_index,
        wigner_inv_envelope,
        node_offset=0,
    ):
        if self.activation_checkpoint_chunk_size is None:
            return self.forward_chunk(
                x,
                x_edge,
                edge_index,
                wigner_inv_envelope,
                node_offset,
            )

        edge_index_partitions = edge_index.split(
            self.activation_checkpoint_chunk_size, dim=1
        )
        wigner_inv_partitions = wigner_inv_envelope.split(
            self.activation_checkpoint_chunk_size, dim=0
        )
        x_edge_partitions = x_edge.split(self.activation_checkpoint_chunk_size, dim=0)

        for idx in range(len(edge_index_partitions)):
            x = torch.utils.checkpoint.checkpoint(
                self.forward_chunk,
                x,
                x_edge_partitions[idx],
                edge_index_partitions[idx],
                wigner_inv_partitions[idx],
                node_offset,
                use_reentrant=False,
            )

        return x


class ChgSpinEmbedding(nn.Module):
    def __init__(
        self,
        embedding_type: Literal["pos_emb", "lin_emb", "rand_emb"],
        embedding_target: Literal["charge", "spin"],
        embedding_size: int,
        grad: bool,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        assert embedding_type in ["pos_emb", "lin_emb", "rand_emb"]
        self.embedding_type = embedding_type
        assert embedding_target in ["charge", "spin"]
        self.embedding_target = embedding_target
        assert embedding_size % 2 == 0, f"{embedding_size=} must be even"

        if self.embedding_target == "charge":
            # 100 is a conservative upper bound
            self.target_dict = {str(x): x + 100 for x in range(-100, 101)}
        elif self.embedding_target == "spin":
            # 100 is a conservative upper bound
            self.target_dict = {str(x): x for x in range(101)}

        if self.embedding_type == "pos_emb":
            # dividing by 2 because x_proj multiplies by 2
            if not grad:
                self.W = nn.Parameter(
                    torch.randn(embedding_size // 2) * scale, requires_grad=False
                )
            else:
                self.W = nn.Parameter(
                    torch.randn(embedding_size // 2) * scale, requires_grad=True
                )
        elif self.embedding_type == "lin_emb":
            self.lin_emb = nn.Linear(in_features=1, out_features=embedding_size)
            if not grad:
                for param in self.lin_emb.parameters():
                    param.requires_grad = False
        elif self.embedding_type == "rand_emb":
            self.rand_emb = nn.Embedding(len(self.target_dict), embedding_size)
            if not grad:
                for param in self.rand_emb.parameters():
                    param.requires_grad = False

        else:
            raise ValueError(f"embedding type {self.embedding_type} not implemented")

    def forward(self, x):
        # null token for spin is 0
        # charge is default 0
        if self.embedding_type == "pos_emb":
            x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
            if self.embedding_target == "charge":
                return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            elif self.embedding_target == "spin":
                zero_idxs = torch.where(x == 0)[0]
                emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
                # this sets the null spin embedding to zero
                emb[zero_idxs] = 0
                return emb
        elif self.embedding_type == "lin_emb":
            if self.embedding_target == "spin":
                x[x == 0] = -100
            return self.lin_emb(x.unsqueeze(-1).float())
        elif self.embedding_type == "rand_emb":
            return self.rand_emb(
                torch.tensor(
                    [self.target_dict[str(i)] for i in x.tolist()],
                    device=x.device,
                    dtype=torch.long,
                )
            )
        raise ValueError(f"embedding type {self.embedding_type} not implemented")


class DatasetEmbedding(nn.Module):
    def __init__(self, embedding_size, enable_grad, dataset_mapping) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.enable_grad = enable_grad
        self.dataset_mapping = dataset_mapping  # mapping from dataset name to dataset embedding name e.g. {"omol": "omol", "oc20": "oc20", "oc20_subset": "oc20"}, this allows multiple subsets to use the same dataset embedding.
        self.dataset_emb_dict = nn.ModuleDict({})
        for dataset in dataset_mapping:
            if dataset not in self.dataset_emb_dict:
                self.dataset_emb_dict[dataset] = nn.Embedding(1, embedding_size)
            if not self.enable_grad:
                for param in self.dataset_emb_dict[dataset].parameters():
                    param.requires_grad = False

    def forward(self, dataset_list):
        device = list(self.parameters())[0].device
        emb_idx = torch.tensor(0, device=device, dtype=torch.long)
        # apply dataset mapping
        dataset_list = [self.dataset_mapping[dataset] for dataset in dataset_list]

        if self.enable_grad and self.training:
            # If gradients are enabled we need to ensure that all embeddings are included
            # in the graph even if they are missing from the batch
            safety_loss_emb = torch.stack(
                [
                    self.dataset_emb_dict[dataset](emb_idx) * 0.0
                    for dataset in self.dataset_emb_dict
                ]
            ).sum(dim=0)

            emb_for_datasets = [
                (
                    self.dataset_emb_dict[dataset](emb_idx) + safety_loss_emb
                    if i == 0
                    else self.dataset_emb_dict[dataset](emb_idx)
                )
                for i, dataset in enumerate(dataset_list)
            ]

        else:
            emb_for_datasets = [
                (self.dataset_emb_dict[dataset](emb_idx)) for dataset in dataset_list
            ]

        return torch.stack(emb_for_datasets, dim=0)
