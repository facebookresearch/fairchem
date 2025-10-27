"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

EPS = 1e-7


class Safeacos(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x_clamped = x.clamp(-1 + EPS, 1 - EPS)
        ctx.save_for_backward(x_clamped)
        return torch.acos(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x_clamped,) = ctx.saved_tensors
        denom = torch.sqrt(1 - x_clamped.pow(2)).clamp(min=EPS)
        return -grad_output / denom


class Safeatan2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, x):
        ctx.save_for_backward(y, x)
        return torch.atan2(y, x)

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, grad_output):
        y, x = ctx.saved_tensors
        denom = (x.pow(2) + y.pow(2)).clamp(min=EPS)
        return (x / denom) * grad_output, (-y / denom) * grad_output


def init_edge_rot_euler_angles(edge_distance_vec):
    # we need to clamp the output here because if using compile
    # normalize can return >1.0 , pytorch #163082
    xyz = torch.nn.functional.normalize(edge_distance_vec).clamp(-1.0, 1.0)
    x, y, z = torch.split(xyz, 1, dim=1)

    # latitude (beta)
    beta = Safeacos.apply(y.squeeze(-1))

    # longitude (alpha)
    alpha = Safeatan2.apply(x.squeeze(-1), z.squeeze(-1))

    # random gamma (roll)
    gamma = torch.rand_like(alpha) * 2 * torch.pi
    # intrinsic to extrinsic swap
    return -gamma, -beta, -alpha


# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L37
# In 0.5.0, e3nn shifted to torch.matrix_exp which is significantly slower:
# https://github.com/e3nn/e3nn/blob/0.5.0/e3nn/o3/_wigner.py#L92
def wigner_D(
    lv: int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    _Jd: list[torch.Tensor],
) -> torch.Tensor:
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = _Jd[lv]
    Xa = _z_rot_mat(alpha, lv)
    Xb = _z_rot_mat(beta, lv)
    Xc = _z_rot_mat(gamma, lv)
    return Xa @ J @ Xb @ J @ Xc


def _z_rot_mat(angle: torch.Tensor, lv: int) -> torch.Tensor:
    M = angle.new_zeros((*angle.shape, 2 * lv + 1, 2 * lv + 1))

    # The following code needs to replaced for a for loop because
    # torch.export barfs on outer product like operations
    # ie: torch.outer(frequences, angle) (same as frequencies * angle[..., None])
    # will place a non-sense Guard on the dimensions of angle when attempting to export setting
    # angle (edge dimensions) as dynamic. This may be fixed in torch2.4.

    # inds = torch.arange(0, 2 * lv + 1, 1, device=device)
    # reversed_inds = torch.arange(2 * lv, -1, -1, device=device)
    # frequencies = torch.arange(lv, -lv - 1, -1, dtype=dtype, device=device)
    # M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    # M[..., inds, inds] = torch.cos(frequencies * angle[..., None])

    inds = list(range(0, 2 * lv + 1, 1))
    reversed_inds = list(range(2 * lv, -1, -1))
    frequencies = list(range(lv, -lv - 1, -1))
    for i in range(len(frequencies)):
        M[..., inds[i], reversed_inds[i]] = torch.sin(frequencies[i] * angle)
        M[..., inds[i], inds[i]] = torch.cos(frequencies[i] * angle)
    return M


def eulers_to_wigner(
    eulers: torch.Tensor,
    start_lmax: int,
    end_lmax: int,
    Jd: list[torch.Tensor],
) -> torch.Tensor:
    """
    set <rot_clip=True> to handle gradient instability when using gradient-based force/stress prediction.
    """
    alpha, beta, gamma = eulers

    size = int((end_lmax + 1) ** 2) - int((start_lmax) ** 2)
    wigner = torch.zeros(len(alpha), size, size, device=alpha.device, dtype=alpha.dtype)
    start = 0
    for lmax in range(start_lmax, end_lmax + 1):
        block = wigner_D(lmax, alpha, beta, gamma, Jd)
        end = start + block.size()[1]
        wigner[:, start:end, start:end] = block
        start = end

    return wigner
