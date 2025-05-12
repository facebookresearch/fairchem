"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .radial import PolynomialEnvelope, RadialMLP


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
    """

    def __init__(
        self,
        sphere_channels: int,
        lmax: int,
        mmax: int,
        max_num_elements: int,
        edge_channels_list,
        rescale_factor,
        cutoff,
        mappingReduced,
        # Enables activation checkpointing in size of
        # activation_checkpoint_chunk_size edge blocks
        activation_checkpoint_chunk_size: int | None,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mappingReduced
        self.activation_checkpoint_chunk_size = activation_checkpoint_chunk_size

        self.m_0_num_coefficients: int = self.mappingReduced.m_size[0]
        self.m_all_num_coefficents: int = len(self.mappingReduced.l_harmonic)

        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        self.max_num_elements = max_num_elements
        self.edge_channels_list = copy.deepcopy(edge_channels_list)

        # Embedding function of distance
        self.edge_channels_list.append(self.m_0_num_coefficients * self.sphere_channels)
        self.rad_func = RadialMLP(self.edge_channels_list)

        self.rescale_factor = rescale_factor

        self.cutoff = cutoff
        self.envelope = PolynomialEnvelope(exponent=5)

    def forward_chunk(
        self,
        x,
        x_edge,
        edge_distance,
        edge_index,
        wigner_and_M_mapping_inv,
        node_offset=0,
    ):
        x_edge_m_0 = self.rad_func(x_edge)
        x_edge_m_0 = x_edge_m_0.reshape(
            -1, self.m_0_num_coefficients, self.sphere_channels
        )
        x_edge_m_pad = torch.zeros(
            (
                x_edge_m_0.shape[0],
                (self.m_all_num_coefficents - self.m_0_num_coefficients),
                self.sphere_channels,
            ),
            device=x_edge_m_0.device,
            dtype=x_edge_m_0.dtype,
        )
        x_edge_embedding = torch.cat((x_edge_m_0, x_edge_m_pad), dim=1)
        x_edge_embedding = torch.bmm(wigner_and_M_mapping_inv, x_edge_embedding)

        # envelope
        dist_scaled = edge_distance / self.cutoff
        env = self.envelope(dist_scaled)
        x_edge_embedding = x_edge_embedding * env.view(-1, 1, 1)

        # TODO is this needed?
        x_edge_embedding = x_edge_embedding.to(x.dtype)

        return x.index_add(
            0, edge_index[1] - node_offset, x_edge_embedding / self.rescale_factor
        )

    def forward(
        self,
        x,
        x_edge,
        edge_distance,
        edge_index,
        wigner_and_M_mapping_inv,
        node_offset=0,
    ):
        if self.activation_checkpoint_chunk_size is None:
            return self.forward_chunk(
                x,
                x_edge,
                edge_distance,
                edge_index,
                wigner_and_M_mapping_inv,
                node_offset,
            )

        edge_index_partitions = edge_index.split(
            self.activation_checkpoint_chunk_size, dim=1
        )
        wigner_inv_partitions = wigner_and_M_mapping_inv.split(
            self.activation_checkpoint_chunk_size, dim=0
        )
        edge_distance_parititons = edge_distance.split(
            self.activation_checkpoint_chunk_size, dim=0
        )
        x_edge_partitions = x_edge.split(self.activation_checkpoint_chunk_size, dim=0)

        for idx in range(len(edge_index_partitions)):
            x = torch.utils.checkpoint.checkpoint(
                self.forward_chunk,
                x,
                x_edge_partitions[idx],
                edge_distance_parititons[idx],
                edge_index_partitions[idx],
                wigner_inv_partitions[idx],
                node_offset,
                use_reentrant=False,
            )

        return x


class ChgSpinEmbedding(nn.Module):
    def __init__(
        self,
        embedding_type,
        embedding_target,
        embedding_size,
        grad,
        scale=1.0,
    ):
        super().__init__()
        assert embedding_type in ["pos_emb", "lin_emb", "rand_emb"]
        self.embedding_type = embedding_type
        assert embedding_target in ["charge", "spin"]
        self.embedding_target = embedding_target

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
    def __init__(self, embedding_size, grad, dataset_list):
        super().__init__()
        self.embedding_size = embedding_size
        self.dataset_emb_dict = nn.ModuleDict({})
        for dataset in dataset_list:
            if dataset not in self.dataset_emb_dict:
                self.dataset_emb_dict[dataset] = nn.Embedding(1, embedding_size)
            if not grad:
                for param in self.dataset_emb_dict[dataset].parameters():
                    param.requires_grad = False

    def forward(self, dataset_list):
        device = list(self.parameters())[0].device
        emb_idx = torch.tensor(0, device=device, dtype=torch.long)

        # TODO: this is a hack to accomodate the MPA finetuning
        # emb_for_datasets = [
        #     self.dataset_emb_dict[dataset](emb_idx) for dataset in dataset_list
        # ]
        emb_for_datasets = [
            self.dataset_emb_dict["omat"](emb_idx)
            if dataset in ["mptrj", "salex"]
            else self.dataset_emb_dict[dataset](emb_idx)
            for dataset in dataset_list
        ]

        return torch.stack(emb_for_datasets, dim=0)
