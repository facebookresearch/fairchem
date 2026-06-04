"""Geometry utilities for atomic systems."""

from __future__ import annotations

import numpy as np
import torch


def wrap_positions(
    pos: np.ndarray | torch.Tensor,
    cell: np.ndarray | torch.Tensor,
    pbc: np.ndarray | torch.Tensor | bool,
) -> np.ndarray | torch.Tensor:
    """Wrap positions into the unit cell along periodic dimensions.

    Equivalent to ase.geometry.wrap_positions with eps=0.  Dispatches to a
    pure-torch path for tensor inputs (stays on GPU, no autograd break) and
    a pure-numpy path for array inputs.

    Args:
        pos: Cartesian positions, shape (n_atoms, 3).
        cell: Unit cell with rows as lattice vectors, shape (3, 3).
        pbc: Periodic boundary flags — scalar bool, shape (3,), or (n_atoms, 3).

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
    """Wrap positions into the unit cell (torch, batched per-atom cells).

    Each atom carries its own cell matrix — the natural layout after
    ``data.cell[data.batch]``.  Non-periodic dimensions are left unchanged.

    Args:
        pos: Cartesian positions, shape (n_atoms, 3).
        cell: Per-atom unit cells with rows as lattice vectors, (n_atoms, 3, 3).
        pbc: Per-atom periodic flags, shape (n_atoms, 3).

    Returns:
        Wrapped positions, shape (n_atoms, 3).
    """
    # frac = pos @ inv(cell)  (row-vector form)
    frac = torch.linalg.solve(cell.transpose(1, 2), pos.unsqueeze(-1)).squeeze(-1)
    frac = torch.where(pbc, frac % 1.0, frac)
    # pos = frac @ cell  (row-vector form)
    return (frac.unsqueeze(1) @ cell).squeeze(1)


def minimum_image_displacement(
    dr: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """Apply minimum image convention to displacement vectors (torch, batched per-atom cells).

    Subtracts the nearest lattice translation along periodic dimensions so the
    result lies in (-cell/2, +cell/2).  Non-periodic dimensions are unchanged.

    Args:
        dr: Displacement vectors, shape (n_atoms, 3).
        cell: Per-atom unit cells with rows as lattice vectors, (n_atoms, 3, 3).
        pbc: Per-atom periodic flags, shape (n_atoms, 3).

    Returns:
        Minimum-image displacements, shape (n_atoms, 3).
    """
    frac = torch.linalg.solve(cell.transpose(1, 2), dr.unsqueeze(-1)).squeeze(-1)
    frac = torch.where(pbc, frac - torch.round(frac), frac)
    return (frac.unsqueeze(1) @ cell).squeeze(1)


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
