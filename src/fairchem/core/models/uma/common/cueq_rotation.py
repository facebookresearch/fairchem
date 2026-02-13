"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance.group_theory.descriptors.rotations import x_rotation, y_rotation
from fairchem.core.models.uma.common.so3 import CoefficientMapping
from fairchem.core.models.uma.common.rotation import Safeacos, Safeatan2
from torch.profiler import record_function
from contextlib import contextmanager


# Optional NVTX support for Nsight Systems ranges
try:
    from torch.cuda import nvtx as _nvtx  # type: ignore[attr-defined]
    _nvtx_available = True
except Exception:
    _nvtx_available = False

@contextmanager
def nvtx_range(name: str):
    if _nvtx_available and torch.cuda.is_available():
        _nvtx.range_push(name)
        try:
            yield
        finally:
            _nvtx.range_pop()
    else:
        yield




def yx_rotation(irreps, lmax=None):
    """
    This is a fixed version of cue.descriptors.yx_rotation. It would not be necessary for cuEq >= 0.9.0 where this bug is fixed.
    """

    # Fix: use .polynomial.operations[0][1] instead of .d
    cio = x_rotation(irreps, lmax).polynomial.operations[0][1]  # phi, input, A
    bio = y_rotation(irreps, lmax).polynomial.operations[0][1]  # theta, A, output
    cibo = cue.segmented_polynomials.dot(cio, bio, (2, 1))  # phi, input, theta, output
    cbio = cibo.move_operand(1, 2)  # phi, theta, input, output
    return cue.EquivariantPolynomial(
        [
            cue.IrrepsAndLayout(irreps.new_scalars(cbio.operands[0].size), cue.ir_mul),
            cue.IrrepsAndLayout(irreps.new_scalars(cbio.operands[1].size), cue.ir_mul),
            cue.IrrepsAndLayout(irreps, cue.ir_mul),
        ],
        [cue.IrrepsAndLayout(irreps, cue.ir_mul)],
        cue.SegmentedPolynomial.eval_last_operand(cbio),
    )

class cuEquivariantRotation:
    def __init__(self, lmax: int, mmax: int, sphere_channels: int, mappingReduced: CoefficientMapping):
        self.lmax = lmax
        self.mmax = mmax
        self.sphere_channels = sphere_channels
        self.mappingReduced = mappingReduced


        # Build irreps string for l in [0, lmax]
        def _parity(l: int) -> str:
            return "e" if l % 2 == 0 else "o"

        # build the rotator (xy_rotation: 3 inputs -> 4 values per path)
        irreps_str = " + ".join(
            [f"{self.sphere_channels}x{l}{_parity(l)}" for l in range(self.lmax + 1)]
        )
        irreps = cue.Irreps("O3", irreps_str)
        perm = [i.item() for i in torch.where(mappingReduced.to_m.T==1.0)[1]]
        poly = cue.descriptors.xy_rotation(irreps).polynomial
        self._cue_rot = cuet.SegmentedPolynomial(poly, method="uniform_1d")
        for i in range(self._cue_rot.m.num_paths[0]):
            self._cue_rot.m.path_indices[i*4+3] = perm[self._cue_rot.m.path_indices[i*4+3]]


        # build the inverse rotator
        irreps_str_inv = " + ".join(
            [f"{1 * self.sphere_channels}x{l}{_parity(l)}" for l in range(self.lmax + 1)]
        )
        irreps_inv = cue.Irreps("O3", irreps_str_inv)
        poly_inv = yx_rotation(irreps_inv).polynomial
        self._cue_rot_inv = cuet.SegmentedPolynomial(poly_inv, method="uniform_1d")
        for i in range(self._cue_rot_inv.m.num_paths[0]):
            # Permute theta (Y angle, index 2) - NOT the x input (index 3)
            self._cue_rot_inv.m.path_indices[i*4+2] = perm[self._cue_rot_inv.m.path_indices[i*4+2]]

        self._cueq_available = True

    def init_edge_rot_euler_angles(self, edge_distance_vec):
        # we need to clamp the output here because if using compile
        # normalize can return >1.0 , pytorch #163082
        xyz = torch.nn.functional.normalize(edge_distance_vec).clamp(-1.0, 1.0)
        x, y, z = torch.split(xyz, 1, dim=1)

        # x-rotation (titlts vec up to the Y axis)
        beta = Safeacos.apply(y.squeeze(-1))

        # y-rotation (brings vec onto the YZ plane)
        gamma = Safeatan2.apply(x.squeeze(-1), z.squeeze(-1))

        return -gamma, -beta, None



    def rotate_euler(self, n_edges: int, sph_features: int, x_full: torch.Tensor, euler_angles: tuple[torch.Tensor, torch.Tensor, torch.Tensor], edge_index: torch.Tensor) -> torch.Tensor:
        return self._rotate_euler_2_rotations(n_edges, sph_features, x_full, euler_angles, edge_index)

    def rotate_euler_inv(self, x_message: torch.Tensor, n_nodes: int, euler_angles: tuple[torch.Tensor, torch.Tensor, torch.Tensor], edge_index: torch.Tensor) -> torch.Tensor:
        return self._rotate_euler_inv_2_rotations(x_message, n_nodes, euler_angles, edge_index)

    def rotate_euler_inv_scatter(self, x_message: torch.Tensor, n_nodes: int, euler_angles: tuple[torch.Tensor, torch.Tensor, torch.Tensor], edge_index: torch.Tensor, edge_envelope: torch.Tensor, rescale_factor: float = 1.0, node_offset: int = 0) -> torch.Tensor:
        return self._rotate_euler_inv_scatter_2_rotations(x_message, n_nodes, euler_angles, edge_index, edge_envelope, rescale_factor, node_offset)


    def _rotate_euler_2_rotations(
        self,
        # x_message: torch.Tensor,
        n_edges: int,
        sph_features: int,
        x_full: torch.Tensor,
        euler_angles: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward rotation using xy_rotation (2 angles).

        euler_angles from init_edge_rot_euler_angles_for_cueq:
            euler_angles[0] = -gamma (Y rotation) = -atan2(x, z)
            euler_angles[1] = -beta (X rotation) = -acos(y)
            euler_angles[2] = None (not used)
        """
        with record_function("cueq.encode_angles"), nvtx_range("cueq.encode_angles"):
            gamma = cuet.encode_rotation_angle(euler_angles[0], self.lmax)
            beta = cuet.encode_rotation_angle(euler_angles[1], self.lmax)
            beta_double = torch.cat([beta, beta], dim=0)
            gamma_double = torch.cat([gamma, gamma], dim=0)

        with record_function("L -> flat"), nvtx_range("L -> flat"):
            x_full_flat = x_full.reshape(x_full.shape[0], -1)

        with record_function("cuEq.SegmentedPolynomial"), nvtx_range("cuEq.SegmentedPolynomial"):
            # xy_rotation: [gamma, beta, x] -> x is at index 2
            cue_out = self._cue_rot(
                [gamma_double, beta_double, x_full_flat],
                input_indices={2: edge_index.flatten()}
                )[0].view(2, n_edges, sph_features, self.sphere_channels)
        x_message = torch.cat([cue_out[0], cue_out[1]], dim=2)
        return x_message




    def _rotate_euler_inv_2_rotations(
        self,
        x_message: torch.Tensor,
        n_nodes: int,
        euler_angles: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
        do_scatter: bool = True,
    ) -> torch.Tensor:
        """
        Inverse rotation using euler angles.

        Args:
            x_message: Edge embeddings of shape (n_edges, sph_features, channels)
            n_nodes: Number of nodes for scatter output
            euler_angles: Tuple of (alpha, beta, gamma) from init_edge_rot_euler_angles
            edge_index: Edge indices [2, n_edges]
            do_scatter: If True, scatter-add to nodes. If False, just rotate (keep edge shape).

        Returns:
            If do_scatter=True: Node embeddings of shape (n_nodes, sph_features, channels)
            If do_scatter=False: Rotated edge embeddings of shape (n_edges, sph_features, channels)
        """
        with record_function("cueq.encode_angles_inv"), nvtx_range("cueq.encode_angles_inv"):
            phi_inv = cuet.encode_rotation_angle(-euler_angles[1], self.lmax)
            theta_inv = cuet.encode_rotation_angle(-euler_angles[0], self.lmax)

        with record_function("L -> flat"), nvtx_range("L -> flat"):
            x_message_flat = x_message.reshape(x_message.shape[0], -1)

        with record_function("cuEq.SegmentedPolynomial.inv"), nvtx_range("cuEq.SegmentedPolynomial.inv"):
            if do_scatter:
                dummy = torch.empty(n_nodes, 1)
                cue_out_inv = self._cue_rot_inv(
                    [phi_inv, theta_inv, x_message_flat],
                    output_indices={0: edge_index[1]},
                    output_shapes={0: dummy}
                )[0]
                new_embedding = cue_out_inv.view(n_nodes, x_message.shape[1], self.sphere_channels)
            else:
                # Just rotation, no scatter
                cue_out_inv = self._cue_rot_inv([phi_inv, theta_inv, x_message_flat])[0]
                new_embedding = cue_out_inv.view(x_message.shape[0], x_message.shape[1], self.sphere_channels)
        return new_embedding



    def _rotate_euler_inv_scatter_2_rotations(
        self,
        x_message: torch.Tensor,
        n_nodes: int,
        euler_angles: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
        edge_envelope: torch.Tensor,
        rescale_factor: float = 1.0,
        node_offset: int = 0,
    ) -> torch.Tensor:
        """
        Combined inverse rotation + envelope scaling + scatter-add to nodes.
        This is more efficient than doing these operations separately.

        Args:
            x_message: Edge embeddings of shape (n_edges, sph_features, channels)
            n_nodes: Number of nodes for scatter output
            euler_angles: Tuple of (alpha, beta, gamma) from init_edge_rot_euler_angles
            edge_index: Edge indices [2, n_edges]
            edge_envelope: Envelope scaling per edge, shape (n_edges, 1, 1)
            rescale_factor: Factor to divide the result by
            node_offset: Offset to subtract from edge_index[1] for graph parallelism

        Returns:
            Node embeddings of shape (n_nodes, sph_features, channels)
        """
        with record_function("cueq.encode_angles_inv"), nvtx_range("cueq.encode_angles_inv"):
            # UMA's euler_angles = (-gamma_orig, -beta_orig, -alpha_orig)
            phi_inv = cuet.encode_rotation_angle(-euler_angles[1], self.lmax)
            theta_inv = cuet.encode_rotation_angle(-euler_angles[0], self.lmax)

        with record_function("apply_envelope"), nvtx_range("apply_envelope"):
            # Apply envelope and rescale BEFORE scatter (while still per-edge)
            x_scaled = (x_message * edge_envelope) / rescale_factor

        with record_function("L -> flat"), nvtx_range("L -> flat"):
            x_scaled_flat = x_scaled.reshape(x_scaled.shape[0], -1)

        with record_function("cuEq.SegmentedPolynomial.inv_scatter"), nvtx_range("cuEq.SegmentedPolynomial.inv_scatter"):
            dummy = torch.empty(n_nodes, 1)
            # Apply node_offset for graph parallelism support
            scatter_indices = edge_index[1] - node_offset if node_offset != 0 else edge_index[1]
            cue_out_inv = self._cue_rot_inv(
                [phi_inv, theta_inv, x_scaled_flat],
                output_indices={0: scatter_indices},
                output_shapes={0: dummy}
            )[0]
            new_embedding = cue_out_inv.view(n_nodes, x_message.shape[1], self.sphere_channels)
        return new_embedding