"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from ase.neighborlist import neighbor_list
from ase.stress import full_3x3_to_voigt_6_stress
from ase.units import Bohr, Hartree

if TYPE_CHECKING:
    from ase import Atoms

logger = logging.getLogger(__name__)

_sqrt_z_r4r2_cache: dict[torch.device, torch.Tensor] = {}


def _get_sqrt_z_r4r2(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Lazily load sqrt_z_r4_over_r2 from dftd4 and cache as a torch tensor.
    """
    if device not in _sqrt_z_r4r2_cache:
        from dftd4.data import sqrt_z_r4_over_r2

        _sqrt_z_r4r2_cache[device] = torch.tensor(
            sqrt_z_r4_over_r2, device=device, dtype=dtype
        )
    return _sqrt_z_r4r2_cache[device].to(dtype=dtype)


def _build_neighbor_data(
    atoms: Atoms,
    cutoff: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build padded neighbor list tensors from an ASE Atoms object.

    Returns:
        neighbor_idx: (N, max_k) padded neighbor atom indices
        neighbor_offsets: (N, max_k, 3) integer cell image offsets
        neighbor_mask: (N, max_k) boolean mask for valid neighbors
    """
    n_atoms = len(atoms)
    idx_i, idx_j, offsets = neighbor_list("ijS", atoms, cutoff)

    counts = np.bincount(idx_i, minlength=n_atoms)
    max_k = int(counts.max()) if len(counts) > 0 else 0

    if max_k == 0:
        return (
            torch.zeros(n_atoms, 0, device=device, dtype=torch.long),
            torch.zeros(n_atoms, 0, 3, device=device, dtype=dtype),
            torch.zeros(n_atoms, 0, device=device, dtype=torch.bool),
        )

    pad_idx = np.zeros((n_atoms, max_k), dtype=np.int64)
    pad_off = np.zeros((n_atoms, max_k, 3), dtype=np.float64)
    pad_mask = np.zeros((n_atoms, max_k), dtype=bool)

    fill_pos = np.zeros(n_atoms, dtype=np.int64)
    for edge in range(len(idx_i)):
        i = idx_i[edge]
        k = fill_pos[i]
        pad_idx[i, k] = idx_j[edge]
        pad_off[i, k] = offsets[edge]
        pad_mask[i, k] = True
        fill_pos[i] += 1

    return (
        torch.tensor(pad_idx, device=device, dtype=torch.long),
        torch.tensor(pad_off, device=device, dtype=dtype),
        torch.tensor(pad_mask, device=device, dtype=torch.bool),
    )


def _compute_chunk_energy(
    chunk_start: int,
    chunk_end: int,
    positions: torch.Tensor,
    cell: torch.Tensor,
    c6: torch.Tensor,
    r4r2_per_atom: torch.Tensor,
    neighbor_idx: torch.Tensor,
    neighbor_offsets: torch.Tensor,
    neighbor_mask: torch.Tensor,
    a1: float,
    a2: float,
    s9: float,
    alp: int,
    cutoff: float,
) -> torch.Tensor:
    """
    Compute ATM energy contribution from center atoms [chunk_start, chunk_end).
    """
    max_k = neighbor_idx.shape[1]
    device = positions.device

    # Slice neighbor data for this chunk
    nb_idx = neighbor_idx[chunk_start:chunk_end]  # (C, max_k)
    nb_off = neighbor_offsets[chunk_start:chunk_end]  # (C, max_k, 3)
    nb_mask = neighbor_mask[chunk_start:chunk_end]  # (C, max_k)

    # Displacement vectors for this chunk
    pos_j = positions[nb_idx]  # (C, max_k, 3)
    pos_i = positions[chunk_start:chunk_end].unsqueeze(1)  # (C, 1, 3)
    offset_contrib = torch.einsum("nkd,de->nke", nb_off, cell)
    vec_ij = pos_j + offset_contrib - pos_i  # (C, max_k, 3)

    r_ij = torch.linalg.norm(vec_ij, dim=-1)  # (C, max_k)
    safe_mask = nb_mask & (r_ij > 1e-10)
    r_ij = torch.where(safe_mask, r_ij, torch.ones_like(r_ij))

    # Pairwise distances between neighbors
    vec_jk = vec_ij.unsqueeze(1) - vec_ij.unsqueeze(2)  # (C, max_k, max_k, 3)
    r_jk = torch.linalg.norm(vec_jk, dim=-1)  # (C, max_k, max_k)
    r_jk = torch.where(r_jk > 1e-10, r_jk, torch.ones_like(r_jk))

    # Triple mask
    j_range = torch.arange(max_k, device=device)
    upper_tri = j_range.unsqueeze(0) < j_range.unsqueeze(1)
    triple_mask = (
        safe_mask.unsqueeze(2)
        & safe_mask.unsqueeze(1)
        & upper_tri.unsqueeze(0)
        & (r_jk < cutoff)
    )

    # Unique triple mask: center i < j_atom and i < k_atom
    i_idx_3d = torch.arange(chunk_start, chunk_end, device=device).view(-1, 1, 1)
    j_idx_3d = nb_idx.unsqueeze(2)
    k_idx_3d = nb_idx.unsqueeze(1)
    unique_mask = (i_idx_3d < j_idx_3d) & (i_idx_3d < k_idx_3d)
    full_mask = triple_mask & unique_mask

    if not full_mask.any():
        return torch.tensor(0.0, device=device, dtype=positions.dtype)

    # C6 / C9
    i_atoms_chunk = torch.arange(chunk_start, chunk_end, device=device)
    c6_ij = c6[i_atoms_chunk.unsqueeze(1), nb_idx]  # (C, max_k)
    c6_ij_3d = c6_ij.unsqueeze(2)
    c6_ik_3d = c6_ij.unsqueeze(1)
    c6_jk_full = c6[nb_idx.unsqueeze(2), nb_idx.unsqueeze(1)]

    c6_prod = c6_ij_3d * c6_ik_3d * c6_jk_full
    c6_prod = torch.clamp(c6_prod, min=0.0)
    c9 = torch.sqrt(c6_prod)

    # BJ radii
    r4r2_j = r4r2_per_atom[nb_idx]
    r4r2_i = r4r2_per_atom[chunk_start:chunk_end].unsqueeze(1)
    q_ij = 3.0 * r4r2_i * r4r2_j
    r0_ij = a1 * torch.sqrt(q_ij) + a2

    r0_ij_3d = r0_ij.unsqueeze(2)
    r0_ik_3d = r0_ij.unsqueeze(1)
    q_jk = 3.0 * r4r2_j.unsqueeze(2) * r4r2_j.unsqueeze(1)
    r0_jk = a1 * torch.sqrt(q_jk) + a2

    # Distances in Bohr
    r_ij_bohr = r_ij / Bohr
    r_jk_bohr = r_jk / Bohr

    r_prod = r_ij_bohr.unsqueeze(2) * r_ij_bohr.unsqueeze(1) * r_jk_bohr
    r_avg = torch.pow(torch.clamp(r_prod, min=1e-30), 1.0 / 3.0)

    r0_prod = r0_ij_3d * r0_ik_3d * r0_jk
    r0_avg = torch.pow(torch.clamp(r0_prod, min=1e-30), 1.0 / 3.0)

    # Damping
    ratio = r0_avg / r_avg
    fdamp = 1.0 / (1.0 + 6.0 * torch.pow(ratio, float(alp)))

    # Angular term
    dot_jk = torch.sum(vec_ij.unsqueeze(2) * vec_ij.unsqueeze(1), dim=-1)
    r_ij_2 = r_ij.unsqueeze(2)
    r_ik_2 = r_ij.unsqueeze(1)

    cos_a = dot_jk / (r_ij_2 * r_ik_2)
    cos_b = (r_ij_2**2 - dot_jk) / (r_ij_2 * r_jk)
    cos_c = (r_ik_2**2 - dot_jk) / (r_ik_2 * r_jk)
    ang_term = 3.0 * cos_a * cos_b * cos_c + 1.0

    # Energy
    r_prod_cubed = torch.clamp(r_prod**3, min=1e-60)
    e_triple = c9 * fdamp * ang_term / r_prod_cubed

    return s9 * torch.sum(e_triple * full_mask)


def compute_atm(
    atoms: Atoms,
    c6_matrix: np.ndarray,
    a1: float,
    a2: float,
    s9: float = 1.0,
    alp: int = 16,
    cutoff: float = 15.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    chunk_size: int = 50,
) -> dict[str, np.ndarray]:
    """
    Compute ATM three-body dispersion energy, forces, and stress on GPU.

    Uses autograd for forces and stress. The C6 matrix must be
    precomputed (e.g., from dftd4's ``get_properties()``). Center
    atoms are processed in chunks to control GPU memory usage.

    Args:
        atoms: ASE Atoms object.
        c6_matrix: (N, N) pairwise C6 dispersion coefficients in
            atomic units (Hartree Bohr^6).
        a1: BJ damping parameter (scaling of critical radii).
        a2: BJ damping parameter (offset of critical radii, in Bohr).
        s9: Scaling factor for three-body dispersion energy.
        alp: ATM damping exponent (default 16).
        cutoff: Real-space cutoff in Angstrom.
        device: Torch device (defaults to CUDA if available).
        dtype: Torch floating point type.
        chunk_size: Number of center atoms per chunk.

    Returns:
        Dict with keys ``"energy"`` (float, eV), ``"forces"``
        (N x 3 ndarray, eV/A), and optionally ``"stress"``
        (6-element Voigt ndarray, eV/A^3) for periodic systems.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_atoms = len(atoms)
    is_periodic = atoms.pbc.any()

    neighbor_idx, neighbor_offsets, neighbor_mask = _build_neighbor_data(
        atoms, cutoff, device, dtype
    )
    max_k = neighbor_idx.shape[1]

    if max_k == 0:
        results: dict[str, np.ndarray] = {
            "energy": 0.0,
            "forces": np.zeros((n_atoms, 3)),
        }
        if is_periodic:
            results["stress"] = np.zeros(6)
        return results

    positions = torch.tensor(
        atoms.positions, device=device, dtype=dtype, requires_grad=True
    )
    if is_periodic:
        cell = torch.tensor(
            atoms.cell.array, device=device, dtype=dtype, requires_grad=True
        )
    else:
        cell = torch.zeros(3, 3, device=device, dtype=dtype)

    c6 = torch.tensor(c6_matrix, device=device, dtype=dtype)
    r4r2 = _get_sqrt_z_r4r2(device, dtype)
    numbers_t = torch.tensor(atoms.numbers, device=device, dtype=torch.long)
    r4r2_per_atom = r4r2[numbers_t]

    # Accumulate energy over chunks
    energy_hartree = torch.tensor(0.0, device=device, dtype=dtype)
    for start in range(0, n_atoms, chunk_size):
        end = min(start + chunk_size, n_atoms)
        energy_hartree = energy_hartree + _compute_chunk_energy(
            start,
            end,
            positions,
            cell,
            c6,
            r4r2_per_atom,
            neighbor_idx,
            neighbor_offsets,
            neighbor_mask,
            a1,
            a2,
            s9,
            alp,
            cutoff,
        )

    energy_ev = energy_hartree * Hartree

    # Forces and stress via autograd
    grad_targets = [positions]
    if is_periodic:
        grad_targets.append(cell)

    grads = torch.autograd.grad(energy_ev, grad_targets, create_graph=False)
    forces_ev_ang = -grads[0].detach().cpu().numpy()

    results: dict[str, np.ndarray | float] = {
        "energy": float(energy_ev.detach().cpu()),
        "forces": forces_ev_ang,
    }

    if is_periodic:
        de_dcell = grads[1].detach().cpu().numpy()
        volume = atoms.get_volume()
        virial_3x3 = de_dcell / volume
        stress_voigt = full_3x3_to_voigt_6_stress(virial_3x3)
        results["stress"] = stress_voigt

    return results


def compute_c6_dftd4(
    atoms: Atoms,
) -> np.ndarray:
    """
    Compute D4 C6 coefficients using dftd4.

    Args:
        atoms: ASE Atoms object.

    Returns:
        (N, N) C6 coefficient matrix in atomic units (Hartree Bohr^6).
    """
    from dftd4.interface import DispersionModel

    disp_kwargs: dict = {
        "numbers": atoms.numbers,
        "positions": atoms.positions / Bohr,
    }
    if atoms.pbc.any():
        disp_kwargs["lattice"] = atoms.cell.array / Bohr
        disp_kwargs["periodic"] = atoms.pbc
    if hasattr(atoms, "info") and "charge" in atoms.info:
        disp_kwargs["charge"] = float(atoms.info["charge"])

    disp = DispersionModel(**disp_kwargs)
    props = disp.get_properties()
    return np.asarray(props["c6 coefficients"])
