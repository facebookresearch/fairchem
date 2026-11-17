"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np
import torch
from ase.geometry.minkowski_reduction import minkowski_reduce


def wrap_positions(
    pos: np.ndarray | torch.Tensor,
    cell: np.ndarray | torch.Tensor,
    pbc: np.ndarray | torch.Tensor | bool,
) -> np.ndarray | torch.Tensor:
    """
    Wrap positions into the unit cell along periodic dimensions.

    Equivalent to ase.geometry.wrap_positions with eps=0.  Dispatches to a
    pure-torch path for tensor inputs (stays on GPU, no autograd break) and
    a pure-numpy path for array inputs.

    Args:
        pos: Cartesian positions, shape (n_atoms, 3).
        cell: Unit cell with rows as lattice vectors, shape (3, 3).
        pbc: Periodic boundary flags — scalar bool or shape (3,).

    Returns:
        Wrapped positions, same type and shape as pos.
    """
    if isinstance(pos, torch.Tensor):
        return _wrap_torch(pos, cell, pbc)
    pos_np = np.asarray(pos, dtype=np.float64)
    cell_np = np.asarray(cell, dtype=np.float64)
    pbc_np = np.broadcast_to(np.asarray(pbc, dtype=bool), (3,)).copy()
    frac = np.linalg.solve(cell_np.T, pos_np.T).T
    for i in range(3):
        if pbc_np[i]:
            frac[:, i] %= 1.0
    return (frac @ cell_np).astype(np.asarray(pos).dtype)


def wrap_positions_torch(
    pos: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """
    Wrap positions into the unit cell (torch, batched per-atom cells).

    Each atom carries its own cell matrix — the natural layout after
    ``data.cell[data.batch]``.  Non-periodic dimensions are left unchanged.

    Args:
        pos: Cartesian positions, shape (n_atoms, 3).
        cell: Per-atom unit cells with rows as lattice vectors, (n_atoms, 3, 3).
        pbc: Per-atom periodic flags, shape (n_atoms, 3).

    Returns:
        Wrapped positions, shape (n_atoms, 3).
    """
    frac = torch.linalg.solve(cell.transpose(1, 2), pos.unsqueeze(-1)).squeeze(-1)
    frac = torch.where(pbc, frac % 1.0, frac)
    return (frac.unsqueeze(1) @ cell).squeeze(1)


def compute_minkowski_op(
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Minkowski reduction operator for a single cell.

    Args:
        cell: Unit cell, shape (3, 3) or (1, 3, 3).
        pbc: Periodic flags, shape (3,) or (1, 3).

    Returns:
        Integer reduction operator, shape matching input (3, 3) or (1, 3, 3).
    """
    squeeze = cell.dim() == 2
    c = cell.squeeze(0) if not squeeze else cell
    p = pbc.squeeze(0) if pbc.dim() == 2 else pbc

    _, op = minkowski_reduce(c.detach().cpu().numpy(), pbc=p.detach().cpu().numpy())
    op_t = torch.tensor(op, dtype=cell.dtype, device=cell.device)

    if not squeeze:
        return op_t.unsqueeze(0)
    return op_t


def minimum_image_displacement(
    dr: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch: torch.Tensor | None = None,
    minkowski_op: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply minimum image convention to displacement vectors.

    Uses Minkowski reduction to transform the cell into a compact basis where
    a 27-neighbour search is guaranteed to find the true shortest image.  This
    is correct for arbitrary cell shapes, including highly skewed cells.
    Non-periodic dimensions are unchanged.

    Args:
        dr: Displacement vectors, shape (n_atoms, 3).
        cell: Unit cells — either per-system (n_systems, 3, 3) when ``batch``
            is provided, or per-atom (n_atoms, 3, 3).
        pbc: Periodic flags — either per-system (n_systems, 3) when ``batch``
            is provided, or per-atom (n_atoms, 3).
        batch: System index for each atom, shape (n_atoms,).  When provided,
            ``cell`` and ``pbc`` are per-system and will be expanded
            internally.
        minkowski_op: Precomputed Minkowski reduction operators, per-system
            (n_systems, 3, 3).  If ``None``, computed on the fly.

    Returns:
        Minimum-image displacements, shape (n_atoms, 3).
    """
    if not pbc.any():
        return dr

    if batch is not None:
        pbc_per_atom = torch.atleast_2d(pbc)[batch]
        cell_per_atom = cell[batch]
    else:
        pbc_per_atom = pbc
        cell_per_atom = cell

    eye = torch.eye(3, dtype=cell_per_atom.dtype, device=cell_per_atom.device)
    off_diag_max = (cell_per_atom * (1.0 - eye)).abs().max()
    if off_diag_max < 1e-10:
        diag = torch.diagonal(cell_per_atom, dim1=1, dim2=2)
        frac = dr / diag
        frac = torch.where(pbc_per_atom, frac - torch.round(frac), frac)
        return frac * diag

    system_cell = cell if batch is not None else cell_per_atom
    system_pbc = torch.atleast_2d(pbc) if batch is not None else pbc_per_atom
    if minkowski_op is not None:
        ops = minkowski_op if batch is None else minkowski_op
        rcell_system = torch.bmm(ops.to(dtype=system_cell.dtype), system_cell)
    else:
        rcell_system = _minkowski_reduce_batched(system_cell, system_pbc)
    if batch is not None:
        rcell = rcell_system[batch]
    else:
        rcell = rcell_system

    frac = torch.linalg.solve(rcell.transpose(1, 2), dr.unsqueeze(-1)).squeeze(-1)
    frac = torch.where(pbc_per_atom, frac - torch.floor(frac + 0.5), frac)

    r = torch.tensor([-1, 0, 1], dtype=frac.dtype, device=frac.device)
    shifts = torch.cartesian_prod(r, r, r)

    shifts_masked = shifts.unsqueeze(0) * pbc_per_atom.unsqueeze(1).to(frac.dtype)

    candidates_frac = frac.unsqueeze(1) + shifts_masked

    candidates_cart = torch.bmm(candidates_frac, rcell)

    lengths = torch.linalg.norm(candidates_cart, dim=2)
    indices = torch.argmin(lengths, dim=1)
    return candidates_cart[torch.arange(dr.shape[0], device=dr.device), indices]


def _minkowski_reduce_batched(
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """
    Minkowski-reduce each cell in a batch.

    Computes integer unimodular operators ``op`` via ASE (cheap integer
    arithmetic on 3x3 matrices), then applies ``rcell = op @ cell`` in
    torch so that ``cell`` stays in the autograd graph.

    Args:
        cell: Unit cells, shape (n_systems, 3, 3).
        pbc: Periodic flags, shape (n_systems, 3).

    Returns:
        Reduced cells, shape (n_systems, 3, 3), same dtype/device as cell.
    """
    cell_cpu = cell.detach().cpu()
    pbc_cpu = pbc.detach().cpu()
    n = cell.shape[0]

    ops = torch.empty(n, 3, 3, dtype=cell.dtype, device=cell.device)
    for i in range(n):
        _, op = minkowski_reduce(cell_cpu[i].numpy(), pbc=pbc_cpu[i].numpy())
        ops[i] = torch.tensor(op, dtype=cell.dtype, device=cell.device)

    return torch.bmm(ops, cell)


def _wrap_torch(
    pos: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor | bool,
) -> torch.Tensor:
    """Single-system torch wrap, broadcasting a (3, 3) cell to per-atom."""
    n = pos.shape[0]
    cell_batched = torch.as_tensor(cell, dtype=pos.dtype, device=pos.device)
    cell_batched = cell_batched.unsqueeze(0).expand(n, -1, -1)
    if isinstance(pbc, bool):
        pbc_tensor = pos.new_full((n, 3), pbc, dtype=torch.bool)
    else:
        pbc_tensor = torch.as_tensor(pbc, dtype=torch.bool, device=pos.device)
        if pbc_tensor.dim() == 1:
            pbc_tensor = pbc_tensor.unsqueeze(0).expand(n, -1)
        elif pbc_tensor.dim() == 2 and pbc_tensor.shape[0] == 1:
            pbc_tensor = pbc_tensor.expand(n, -1)
    return wrap_positions_torch(pos, cell_batched, pbc_tensor)
