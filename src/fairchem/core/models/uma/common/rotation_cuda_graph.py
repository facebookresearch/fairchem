"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import math

import torch
from e3nn import o3

from fairchem.core.models.uma.common.rotation import wigner_D

YTOL = 0.999999


class RotMatWignerCudaGraph:
    def __init__(self):
        assert torch.cuda.is_initialized(), "Cuda Graphs can only be used with GPUs"
        # lazy graph capture
        self.graph_mod = None
        # number of times graph capture has run, can be used to add logic to fail after certain number of times
        self.graph_capture_count = 0
        self.max_edge_size = None
        logging.warning("Using Cuda graphs for wigner matrix creation")

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

        x_hat = torch.tensor(
            [0.0, 1.0, 0.0],
            device=edge_dist_vec.device,
            dtype=edge_dist_vec.dtype,
        )
        mask, neg_mask = create_masks(input_padded, x_hat)
        out = self.graph_mod(input_padded, jds, x_hat, mask, neg_mask)

        edge_rot_mat = torch.narrow(out[0], 0, 0, edge_dist_vec.shape[0])
        wigner = torch.narrow(out[1], 0, 0, edge_dist_vec.shape[0])
        wigner_inv = torch.narrow(out[2], 0, 0, edge_dist_vec.shape[0])
        return edge_rot_mat, wigner, wigner_inv


def capture_rotmat_and_wigner_with_make_graph_callable(
    edge_dist_vec: torch.Tensor, jds: list[torch.Tensor]
):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        edge_dist_vec_clone = edge_dist_vec.clone()
        jds_clone = [jd.clone() for jd in jds]
        x_hat = torch.tensor(
            [0.0, 1.0, 0.0], device=edge_dist_vec.device, dtype=edge_dist_vec.dtype
        )
        mask, neg_mask = create_masks(edge_dist_vec_clone, x_hat)
        graph_mod = torch.cuda.make_graphed_callables(
            edge_rot_and_wigner_graph_capture_region,
            (edge_dist_vec_clone, jds_clone, x_hat, mask, neg_mask),
        )
        torch.cuda.current_stream().wait_stream(s)
        return graph_mod


# this region is being captured by the cuda graph
# cannot contain dynamic inputs and conditional operators based on inputs
def edge_rot_and_wigner_graph_capture_region(
    edge_distance_vecs: torch.Tensor,
    Jd_buffers: list[torch.Tensor],
    x_hat: torch.Tensor,
    mask: torch.Tensor,
    neg_mask: torch.Tensor,
):
    lmax = len(Jd_buffers) - 1
    edge_rot_mat = init_edge_rot_mat_cuda_graph(
        edge_distance_vecs, mask, neg_mask, x_hat
    )
    alpha, beta, gamma = euler_from_edge_rot_mat(edge_rot_mat, x_hat)
    wigner = eulers_to_wigner(alpha, beta, gamma, 0, lmax, Jd_buffers)
    # create a wigner copy that has beta fixed to 0 and pi
    alpha_copy = alpha.clone().detach()
    gamma_copy = gamma.clone().detach()
    beta_copy = beta.clone().detach()
    # TODO: setting these values should be redudant operations because the clipping also happens in the edge_rot_mat creation
    # detaching the gradients here prevents exploding gradients during training, not certain if its needed for inference
    beta_copy[mask] = 0.0
    beta_copy[neg_mask] = math.pi
    wigner_filtered = eulers_to_wigner(
        alpha_copy, beta_copy, gamma_copy, 0, lmax, Jd_buffers
    )
    cond = (~mask & ~neg_mask).view(mask.size(0), 1, 1)
    wigner = torch.where(cond, wigner, wigner_filtered)

    wigner_inv = torch.transpose(wigner, 1, 2).contiguous()
    return edge_rot_mat, wigner, wigner_inv


def create_masks(
    edge_distance_vec: torch.Tensor, x_hat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))
    norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))
    yprod = norm_x @ x_hat
    mask = yprod > YTOL
    neg_mask = yprod < -YTOL
    return mask, neg_mask


def init_edge_rot_mat_cuda_graph(
    edge_distance_vec: torch.Tensor,
    mask: torch.Tensor,
    neg_mask: torch.Tensor,
    x_hat: torch.Tensor,
) -> torch.Tensor:
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

    # Make sure the atoms are far enough apart
    # TODO: move this out of the graph region
    # assert torch.min(edge_vec_0_distance) < 0.0001
    # if len(edge_vec_0_distance) > 0 and torch.min(edge_vec_0_distance) < 0.0001:
    #     logging.error(f"Error edge_vec_0_distance: {torch.min(edge_vec_0_distance)}")

    norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

    norm_x = torch.where(
        mask.unsqueeze(1).expand(-1, 3), x_hat.expand_as(norm_x), norm_x
    )
    norm_x = torch.where(
        neg_mask.unsqueeze(1).expand(-1, 3), -x_hat.expand_as(norm_x), norm_x
    )

    edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
    edge_vec_2 = edge_vec_2 / (torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1))
    # Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
    # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
    edge_vec_2b = edge_vec_2.clone()
    edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
    edge_vec_2b[:, 1] = edge_vec_2[:, 0]
    edge_vec_2c = edge_vec_2.clone()
    edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
    edge_vec_2c[:, 2] = edge_vec_2[:, 1]
    vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(-1, 1)
    vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(-1, 1)

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2)
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2)

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
    # TODO: move this out of the graph region or maybe dont need this
    # Check the vectors aren't aligned
    # if len(vec_dot) > 0:
    #     assert torch.max(vec_dot) < 0.99

    norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
    norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True)))
    norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1))
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / (torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True)))

    # Construct the 3D rotation matrix
    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)

    edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

    return edge_rot_mat


def euler_from_edge_rot_mat(
    edge_rot_mat: torch.Tensor, x_hat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = edge_rot_mat @ x_hat
    alpha, beta = o3.xyz_to_angles(x)
    R = (
        o3.angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2)
        @ edge_rot_mat
    )
    gamma = torch.atan2(R[..., 0, 2], R[..., 0, 0])
    return alpha, beta, gamma


def eulers_to_wigner(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    start_lmax: int,
    end_lmax: int,
    Jd: list[torch.Tensor],
) -> torch.Tensor:
    size = int((end_lmax + 1) ** 2) - int((start_lmax) ** 2)
    wigner = torch.zeros(len(alpha), size, size, device=alpha.device, dtype=alpha.dtype)
    start = 0
    for lmax in range(start_lmax, end_lmax + 1):
        block = wigner_D(lmax, alpha, beta, gamma, Jd)
        end = start + block.size()[1]
        wigner[:, start:end, start:end] = block
        start = end
    return wigner
