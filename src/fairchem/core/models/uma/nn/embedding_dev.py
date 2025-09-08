"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
from typing import Literal

import torch
import torch.nn as nn

from .radial import PolynomialEnvelope, RadialMLP


def detach_variable(inputs):
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ",
            type(inputs).__name__,
        )


class InferenceChunkedEdgeEmbedding(torch.autograd.Function):
    @staticmethod
    def run_fn(ctx, input_tensors=None, grad_S=None):
        (
            x,
            edge_distance_embedding,
            source_atom_embedding,
            target_atom_embedding,
            edge_index,
            wigner_and_M_mapping_inv,
            edge_distance,
        ) = input_tensors if input_tensors is not None else ctx.saved_tensors

        if grad_S is not None:
            g_edge_distance_embedding = []
            g_wigner_and_M_mapping_inv = []
            g_edge_distance = []

        grad_ctx = torch.no_grad() if grad_S is None else torch.enable_grad()
        with grad_ctx:  # TODO could swap these two loops/ctx
            # split up into chunks
            edge_index_partitions = edge_index.split(
                ctx.activation_checkpoint_chunk_size, dim=1
            )
            wigner_inv_partitions = wigner_and_M_mapping_inv.split(
                ctx.activation_checkpoint_chunk_size, dim=0
            )
            edge_distance_parititons = edge_distance.split(
                ctx.activation_checkpoint_chunk_size, dim=0
            )
            edge_distance_embedding_partitions = edge_distance_embedding.split(
                ctx.activation_checkpoint_chunk_size, dim=0
            )

            for idx in range(len(edge_index_partitions)):
                (
                    _x,
                    _edge_distance_embedding,
                    _edge_distance,
                    _edge_index,
                    _wigner_inv,
                ) = detach_variable(
                    (
                        x,
                        edge_distance_embedding_partitions[idx],
                        edge_distance_parititons[idx],
                        edge_index_partitions[idx],
                        wigner_inv_partitions[idx],
                    )
                )

                x_edge = torch.cat(
                    (
                        _edge_distance_embedding,
                        source_atom_embedding[_edge_index[0]],
                        target_atom_embedding[_edge_index[1]],
                    ),
                    dim=1,
                )

                out_x = ctx.forward_chunk(
                    _x,
                    x_edge,
                    _edge_distance,
                    _edge_index,
                    _wigner_inv,
                    node_offset=ctx.node_offset,
                )

                if grad_S is None:
                    x = out_x  # accumulate
                else:
                    term = (out_x * grad_S).sum()
                    (
                        this_g_edge_distance_embedding,
                        this_g_wigner_and_M_mapping_inv,
                        this_g_edge_distance,
                    ) = torch.autograd.grad(
                        term,
                        [
                            _edge_distance_embedding,
                            _wigner_inv,
                            _edge_distance,
                        ],
                        # create_graph=True, #can be false?
                        # retain_graph=True,
                        # allow_unused=True
                    )
                    g_edge_distance_embedding.append(this_g_edge_distance_embedding)
                    g_wigner_and_M_mapping_inv.append(this_g_wigner_and_M_mapping_inv)
                    g_edge_distance.append(this_g_edge_distance)

        if grad_S is None:
            return x
        else:
            # One grad per TENSOR input of forward; None for non-tensors/flags
            return (
                None,  # forward_chunk (callable)
                None,  # g_x,
                torch.cat(g_edge_distance_embedding, dim=0),
                None,  # g_source_atom_embedding,
                None,  # g_target_atom_embedding,
                torch.cat(g_edge_distance, dim=0),
                None,  # edge_index,
                torch.cat(g_wigner_and_M_mapping_inv, dim=0),
                None,  # node_offset
                None,  # create_graph_for_input (bool)
            )

    @staticmethod
    def forward(
        ctx,
        forward_chunk,
        x,
        edge_distance_embedding,
        source_atom_embedding,
        target_atom_embedding,
        edge_distance,
        edge_index,
        wigner_and_M_mapping_inv,
        activation_checkpoint_chunk_size,
        node_offset,
        create_graph_for_input: bool = False,
    ):
        ctx.forward_chunk = forward_chunk
        ctx.create_graph_for_input = bool(create_graph_for_input)
        ctx.node_offset = node_offset
        ctx.activation_checkpoint_chunk_size = activation_checkpoint_chunk_size

        input_tensors = (
            x,
            edge_distance_embedding,
            source_atom_embedding,
            target_atom_embedding,
            edge_index,
            wigner_and_M_mapping_inv,
            edge_distance,
        )
        ctx.save_for_backward(*input_tensors)

        return InferenceChunkedEdgeEmbedding.run_fn(
            ctx,
            input_tensors=input_tensors,
            grad_S=None,
        )

    @staticmethod
    def backward(ctx, grad_S):
        return InferenceChunkedEdgeEmbedding.run_fn(ctx, grad_S=grad_S)


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
        out=None,
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

        # acc = x if out is None else out

        # TODO can do index_add_ in fwd?
        return x.index_add(
            0, edge_index[1] - node_offset, x_edge_embedding / self.rescale_factor
        )

    def forward(
        self,
        x,
        edge_distance_embedding,
        source_atom_embedding,
        target_atom_embedding,
        edge_distance,
        edge_index,
        wigner_and_M_mapping_inv,
        node_offset=0,
    ):
        if self.activation_checkpoint_chunk_size is None:
            x_edge = torch.cat(
                (
                    edge_distance_embedding,
                    source_atom_embedding[edge_index[0]],
                    target_atom_embedding[edge_index[1]],
                ),
                dim=1,
            )
            return self.forward_chunk(
                x,
                x_edge,
                edge_distance,
                edge_index,
                wigner_and_M_mapping_inv,
                node_offset,
            )

        ret = InferenceChunkedEdgeEmbedding.apply(
            self.forward_chunk,
            x,
            edge_distance_embedding,
            source_atom_embedding,
            target_atom_embedding,
            edge_distance,
            edge_index,
            wigner_and_M_mapping_inv,
            self.activation_checkpoint_chunk_size,
            node_offset,
        )
        print("DONE CALLING APPLY!")
        return ret
        edge_index_partitions = edge_index.split(
            self.activation_checkpoint_chunk_size, dim=1
        )
        wigner_inv_partitions = wigner_and_M_mapping_inv.split(
            self.activation_checkpoint_chunk_size, dim=0
        )
        edge_distance_parititons = edge_distance.split(
            self.activation_checkpoint_chunk_size, dim=0
        )
        x_edge = torch.cat(
            (
                edge_distance_embedding,
                source_atom_embedding[edge_index[:, 0]],
                target_atom_embedding[edge_index[:, 1]],
            ),
            dim=1,
        )
        x_edge_partitions = x_edge.split(self.activation_checkpoint_chunk_size, dim=0)

        for idx in range(len(edge_index_partitions)):
            x_edge = torch.cat(
                (edge_distance_embedding, source_embedding, target_embedding), dim=1
            )
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
