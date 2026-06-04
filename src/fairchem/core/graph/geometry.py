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

    Mirrors ase.geometry.wrap_positions (with eps=0, center=(0.5,0.5,0.5))
    but dispatches to a pure-torch path for tensor inputs so the operation
    stays on GPU without leaving the autograd graph.

    Args:
        pos: Cartesian positions, shape (n_atoms, 3).
        cell: Unit cell matrix with rows as lattice vectors, shape (3, 3).
            For torch inputs with a batched per-atom cell (n_atoms, 3, 3),
            use wrap_positions_torch directly.
        pbc: Periodic boundary flags.  Scalar bool, shape (3,), or (n_atoms, 3).

    Returns:
        Wrapped positions, same type and shape as pos.
    """
    if isinstance(pos, torch.Tensor):
        return _wrap_torch(pos, cell, pbc)
    from ase.geometry import wrap_positions as _ase_wrap
    pbc_np = np.asarray(pbc, dtype=bool)
    return _ase_wrap(np.asarray(pos), np.asarray(cell), pbc=pbc_np, eps=0)


def wrap_positions_torch(
    pos: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """Wrap positions into the unit cell (torch, batched per-atom cells).

    Each atom carries its own cell matrix, which is the natural layout after
    ``data.cell[data.batch]``.  Along non-periodic dimensions positions are
    left unchanged.

    Args:
        pos: Cartesian positions, shape (n_atoms, 3).
        cell: Per-atom unit cells with rows as lattice vectors, (n_atoms, 3, 3).
        pbc: Per-atom periodic flags, shape (n_atoms, 3).

    Returns:
        Wrapped positions, shape (n_atoms, 3).
    """
    # Fractional coordinates via cell.T @ frac = pos (column-vector form),
    # equivalent to frac = pos @ inv(cell) in row-vector form.
    frac = torch.linalg.solve(cell.transpose(1, 2), pos.unsqueeze(-1)).squeeze(-1)
    frac = torch.where(pbc, frac % 1.0, frac)
    # Reconstruct: pos = frac @ cell (row-vector form).
    return (frac.unsqueeze(1) @ cell).squeeze(1)


def minimum_image_displacement(
    dr: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """Apply minimum image convention to displacement vectors (torch, batched per-atom cells).

    For each displacement, subtracts the nearest lattice translation along
    periodic dimensions so the result lies in (-cell/2, +cell/2).
    Non-periodic dimensions are left unchanged.

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

    pos: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor | bool,
) -> torch.Tensor:
    """Single-system torch wrap, broadcasting cell to per-atom."""
    n = pos.shape[0]
    cell_batched = cell.unsqueeze(0).expand(n, -1, -1)
    if isinstance(pbc, bool):
        pbc_tensor = pos.new_ones(n, 3, dtype=torch.bool) * pbc
    else:
        pbc_tensor = torch.as_tensor(pbc, dtype=torch.bool, device=pos.device)
        if pbc_tensor.dim() == 1:
            pbc_tensor = pbc_tensor.unsqueeze(0).expand(n, -1)
    return wrap_positions_torch(pos, cell_batched, pbc_tensor)
