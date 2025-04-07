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
        out_mask,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mappingReduced

        self.m_0_num_coefficients: int = self.mappingReduced.m_size[0]
        self.m_all_num_coefficents: int = len(self.mappingReduced.l_harmonic)

        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        self.max_num_elements = max_num_elements
        self.edge_channels_list = copy.deepcopy(edge_channels_list)

        self.source_embedding = nn.Embedding(
            self.max_num_elements, self.edge_channels_list[-1]
        )
        self.target_embedding = nn.Embedding(
            self.max_num_elements, self.edge_channels_list[-1]
        )
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
        self.edge_channels_list[0] = (
            self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        )

        # Embedding function of distance
        self.edge_channels_list.append(self.m_0_num_coefficients * self.sphere_channels)
        self.rad_func = RadialMLP(self.edge_channels_list)

        self.rescale_factor = rescale_factor

        self.cutoff = cutoff
        self.envelope = PolynomialEnvelope(exponent=5)
        self.out_mask = out_mask

    def forward(
        self,
        r,
        atomic_numbers,
        edge_distance_embedding,
        edge_index,
        wigner,
        node_offset=0,
    ):
        source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
        target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
        source_embedding = self.source_embedding(source_element)
        target_embedding = self.target_embedding(target_element)
        x_edge = torch.cat(
            (edge_distance_embedding, source_embedding, target_embedding), dim=1
        )

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
        )
        x_edge_embedding = torch.cat((x_edge_m_0, x_edge_m_pad), dim=1)

        # Reshape the spherical harmonics based on l (degree)
        x_edge_embedding = torch.einsum(
            "nac,ab->nbc", x_edge_embedding, self.mappingReduced.to_m
        )

        # Rotate back the irreps
        wigner_inv = torch.transpose(wigner, 1, 2).contiguous()
        x_edge_embedding = torch.bmm(wigner_inv[:, :, self.out_mask], x_edge_embedding)

        # envelope
        dist_scaled = r / self.cutoff
        env = self.envelope(dist_scaled)
        x_edge_embedding = x_edge_embedding * env.view(-1, 1, 1)

        # Compute the sum of the incoming neighboring messages for each target node
        degree_embedding = torch.zeros(
            (atomic_numbers.size(0),) + x_edge_embedding.shape[1:],
            dtype=x_edge_embedding.dtype,
            device=x_edge_embedding.device,
        )
        degree_embedding.index_add_(0, edge_index[1] - node_offset, x_edge_embedding)
        # NOTE: this is a sum aggregation so normalization over number of edges is good. sqrt of avg number of neighbors.
        degree_embedding = degree_embedding / self.rescale_factor
        return degree_embedding
