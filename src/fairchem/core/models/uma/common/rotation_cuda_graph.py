"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging

import torch

from fairchem.core.models.uma.common.rotation import eulers_to_wigner


class RotMatWignerCudaGraph:
    def __init__(self):
        assert torch.cuda.is_initialized(), "Cuda Graphs can only be used with GPUs"
        # lazy graph capture
        self.graph_mod = None
        # number of times graph capture has run, can be used to add logic to fail after certain number of times
        self.graph_capture_count = 0
        self.max_edge_size = None
        logging.info("Using Cuda graphs for wigner matrix creation")

    def _capture_graph(self, edge_dist_vec: torch.Tensor, jds: list[torch.Tensor]):
        self.max_edge_size = edge_dist_vec.shape[0]
        self.graph_mod = capture_rotmat_and_wigner_with_make_graph_callable(
            edge_dist_vec, jds
        )
        self.graph_capture_count += 1
        if self.graph_capture_count % 10 == 5:
            logging.warning(
                f"CUDA Graph capture for Wigner Matrix has been called {self.graph_capture_count} times, it might slow down inference if called too frequently, consider turning this feature off."
            )

    def get_rotmat_and_wigner(
        self, edge_dist_vec: torch.Tensor, jds: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(edge_dist_vec.shape) == 2
        assert edge_dist_vec.shape[1] == 3

        # if size of edge_dist_vec is less than max_edges, we pad up and select a subset,
        # otherwise we recompute the graph
        input_padded = edge_dist_vec
        if self.graph_mod is None or edge_dist_vec.shape[0] > self.max_edge_size:
            self._capture_graph(edge_dist_vec, jds)
        elif edge_dist_vec.shape[0] < self.max_edge_size:
            pad_n = self.max_edge_size - edge_dist_vec.shape[0]
            input_padded = torch.nn.functional.pad(edge_dist_vec, (0, 0, 0, pad_n))

        jds_clone = [jd.clone() for jd in jds]
        out = edge_rot_and_wigner_graph_capture_region(input_padded, jds_clone)

        wigner = torch.narrow(out[0], 0, 0, edge_dist_vec.shape[0])
        wigner_inv = torch.narrow(out[1], 0, 0, edge_dist_vec.shape[0])
        return wigner, wigner_inv


def capture_rotmat_and_wigner_with_make_graph_callable(
    edge_dist_vec: torch.Tensor, jds: list[torch.Tensor]
):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        edge_dist_vec_clone = edge_dist_vec.clone()
        jds_clone = [jd.clone() for jd in jds]
        graph_mod = torch.cuda.make_graphed_callables(
            edge_rot_and_wigner_graph_capture_region,
            (edge_dist_vec_clone, jds_clone),
        )
        torch.cuda.current_stream().wait_stream(s)
        return graph_mod


def edge_rot_and_wigner_graph_capture_region(
    edge_distance_vecs: torch.Tensor,
    Jd_buffers: list[torch.Tensor],
):
    lmax = len(Jd_buffers) - 1
    mask, alpha, beta, gamma = init_edge_rot_euler_angles_wigner_cuda_graph(
        edge_distance_vecs
    )
    wigner = eulers_to_wigner((alpha, beta, gamma), 0, lmax, Jd_buffers)

    # detaching the gradients here prevents exploding gradients during training, not certain if its needed for inference
    alpha_copy = alpha.clone().detach()
    gamma_copy = gamma.clone().detach()
    beta_copy = beta.clone().detach()

    wigner_filtered = eulers_to_wigner(
        (alpha_copy, beta_copy, gamma_copy), 0, lmax, Jd_buffers
    )

    wigner = torch.where(mask.view(mask.size(0), 1, 1), wigner_filtered, wigner)
    wigner_inv = torch.transpose(wigner, 1, 2).contiguous()
    return wigner, wigner_inv


def init_edge_rot_euler_angles_wigner_cuda_graph(edge_distance_vec):
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

    # make unit vectors
    xyz = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

    # are we standing at the north pole
    mask = xyz[:, 1].abs().isclose(xyz.new_ones(1))

    # compute alpha and beta

    # latitude (beta)
    beta = torch.acos(xyz[:, 1])

    # longitude (alpha)
    alpha = torch.atan2(xyz[:, 0], xyz[:, 2])

    # random gamma (roll)
    gamma = torch.rand_like(alpha) * 2 * torch.pi
    # gamma = torch.zeros_like(alpha)

    # intrinsic to extrinsic swap
    return mask, -gamma, -beta, -alpha
