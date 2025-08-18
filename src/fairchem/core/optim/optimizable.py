
from __future__ import annotations

from functools import cached_property
from types import SimpleNamespace
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
from ase.calculators.calculator import PropertyNotImplementedError
from ase.optimize.optimize import Optimizable
from ase.stress import voigt_6_to_full_3x3_stress

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit import MLIPPredictUnit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms
    from numpy.typing import NDArray


ALL_CHANGES: set[str] = {
    "pos",
    "atomic_numbers",
    "cell",
    "pbc",
}


def compare_batches(
    batch1: AtomicData | None,
    batch2: AtomicData,
    tol: float = 1e-6,
    excluded_properties: set[str] | None = None,
) -> list[str]:
    """Compare properties between two batches

    Args:
        batch1: atoms batch
        batch2: atoms batch
        tol: tolerance used to compare equility of floating point properties
        excluded_properties: list of properties to exclude from comparison

    Returns:
        list of system changes, property names that are differente between batch1 and batch2
    """
    system_changes = []

    if batch1 is None:
        system_changes = ALL_CHANGES
    else:
        properties_to_check = set(ALL_CHANGES)
        if excluded_properties:
            properties_to_check -= set(excluded_properties)

        # Check properties that aren't
        for prop in ALL_CHANGES:
            if prop in properties_to_check:
                properties_to_check.remove(prop)
                if not torch.allclose(
                    getattr(batch1, prop), getattr(batch2, prop), atol=tol
                ):
                    system_changes.append(prop)

    return system_changes


class OptimizableBatch(Optimizable):
    """A Batch version of ase Optimizable Atoms

    This class can be used with ML relaxations in fairchem.core.relaxations.ml_relaxation
    or in ase relaxations classes, i.e. ase.optimize.lbfgs
    """

    ignored_changes: ClassVar[set[str]] = set()

    def __init__(
        self,
        batch: AtomicData,
        predictor: MLIPPredictUnit,
        transform: torch.nn.Module | None = None,
        mask_converged: bool = True,
        numpy: bool = False,
        masked_eps: float = 1e-8,
    ):
        """Initialize Optimizable Batch

        Args:
            batch: A batch of atoms graph data
            model: An instance of a BaseTrainer derived class
            transform: graph transform
            mask_converged: if true will mask systems in batch that are already converged
            numpy: whether to cast results to numpy arrays
            masked_eps: masking systems that are converged when using ASE optimizers results in divisions by zero
                from zero differences in masked positions at future steps, we add a small number to prevent this.
        """
        self.batch = batch.to(predictor.device)
        self.predictor = predictor
        self.transform = transform
        self.numpy = numpy
        self.mask_converged = mask_converged
        self._cached_batch = None
        self._update_mask = None
        self.torch_results = {}
        self.results = {}
        self._eps = masked_eps

    @property
    def device(self):
        return self.predictor.device

    @property
    def batch_indices(self):
        """Get the batch indices specifying which position/force corresponds to which batch."""
        return self.batch.batch

    @property
    def converged_mask(self):
        if self._update_mask is not None:
            return torch.logical_not(self._update_mask)
        return None

    @property
    def update_mask(self):
        if self._update_mask is None:
            return torch.ones(len(self.batch), dtype=bool)
        return self._update_mask

    def check_state(self, batch: AtomicData, tol: float = 1e-12) -> bool:
        """Check for any system changes since last calculation."""
        return compare_batches(
            self._cached_batch,
            batch,
            tol=tol,
            excluded_properties=set(self.ignored_changes),
        )

    def _predict(self) -> None:
        """Run prediction if batch has any changes."""
        system_changes = self.check_state(self.batch)
        if len(system_changes) > 0:
            self.torch_results = self.predictor.predict(self.batch)
            # save only subset of props in simple namespace instead of cloning the whole batch to save memory
            changes = ALL_CHANGES - set(self.ignored_changes)
            self._cached_batch = SimpleNamespace(
                **{prop: self.batch[prop].clone() for prop in changes}
            )

    def get_property(self, name, no_numpy: bool = False) -> torch.Tensor | NDArray:
        """Get a predicted property by name."""
        self._predict()
        if self.numpy:
            self.results = {
                key: pred.item() if pred.numel() == 1 else pred.cpu().numpy()
                for key, pred in self.torch_results.items()
            }
        else:
            self.results = self.torch_results

        if name not in self.results:
            raise PropertyNotImplementedError(f"{name} not present in this calculation")

        return self.results[name] if no_numpy is False else self.torch_results[name]

    def get_positions(self) -> torch.Tensor | NDArray:
        """Get the batch positions"""
        pos = self.batch.pos.clone()
        if self.numpy:
            if self.mask_converged:
                pos[~self.update_mask[self.batch.batch]] = self._eps
            pos = pos.cpu().numpy()

        return pos

    def set_positions(self, positions: torch.Tensor | NDArray) -> None:
        """Set the atom positions in the batch."""
        if isinstance(positions, np.ndarray):
            positions = torch.tensor(positions)

        positions = positions.to(dtype=torch.float32, device=self.device)
        if self.mask_converged and self._update_mask is not None:
            mask = self.update_mask[self.batch.batch]
            self.batch.pos[mask] = positions[mask]
        else:
            self.batch.pos = positions

    def get_forces(
        self, apply_constraint: bool = False, no_numpy: bool = False
    ) -> torch.Tensor | NDArray:
        """Get predicted batch forces."""
        forces = self.get_property("forces", no_numpy=no_numpy)
        if apply_constraint:
            fixed_idx = torch.where(self.batch.fixed == 1)[0]
            if isinstance(forces, np.ndarray):
                fixed_idx = fixed_idx.tolist()
            forces[fixed_idx] = 0.0
        return forces

    def get_potential_energy(self, **kwargs) -> torch.Tensor | NDArray:
        """Get predicted energy as the sum of all batch energies."""
        # ASE 3.22.1 expects a check for force_consistent calculations
        if kwargs.get("force_consistent", False) is True:
            raise PropertyNotImplementedError(
                "force_consistent calculations are not implemented"
            )
        if (
            len(self.batch) == 1
        ):  # unfortunately batch size 1 returns a float, not a tensor
            return self.get_property("energy")
        return self.get_property("energy").sum()

    def get_potential_energies(self) -> torch.Tensor | NDArray:
        """Get the predicted energy for each system in batch."""
        return self.get_property("energy")

    def get_cells(self) -> torch.Tensor:
        """Get batch crystallographic cells."""
        return self.batch.cell

    def set_cells(self, cells: torch.Tensor | NDArray) -> None:
        """Set batch cells."""
        assert self.batch.cell.shape == cells.shape, "Cell shape mismatch"
        if isinstance(cells, np.ndarray):
            cells = torch.tensor(cells, dtype=torch.float32, device=self.device)
        cells = cells.to(dtype=torch.float32, device=self.device)
        self.batch.cell[self.update_mask] = cells[self.update_mask]

    def get_volumes(self) -> torch.Tensor:
        """Get a tensor of volumes for each cell in batch"""
        cells = self.get_cells()
        return torch.linalg.det(cells)

    def iterimages(self) -> AtomicData:
        # XXX document purpose of iterimages - this is just needed to work with ASE optimizers
        yield self.batch

    def get_max_forces(
        self, forces: torch.Tensor | None = None, apply_constraint: bool = False
    ) -> torch.Tensor:
        """Get the maximum forces per structure in batch"""
        # from torch_scatter import scatter

        if forces is None:
            forces = self.get_forces(apply_constraint=apply_constraint, no_numpy=True)
        # return scatter((forces**2).sum(axis=1).sqrt(), self.batch_indices, reduce="max")
        force_norms = torch.norm(forces, dim=1)
        num_systems = self.batch.natoms.shape[0]
        result = torch.zeros(num_systems, dtype=force_norms.dtype, device=force_norms.device)
        result.scatter_reduce_(0, self.batch.batch, force_norms, reduce="amax")
        return result

    def converged(
        self,
        forces: torch.Tensor | NDArray | None,
        fmax: float,
        max_forces: torch.Tensor | None = None,
    ) -> bool:
        """Check if norm of all predicted forces are below fmax"""
        if forces is not None:
            if isinstance(forces, np.ndarray):
                forces = torch.tensor(forces, device=self.device, dtype=torch.float32)
            max_forces = self.get_max_forces(forces)
        elif max_forces is None:
            max_forces = self.get_max_forces()

        update_mask = max_forces.ge(fmax)
        # update cached mask
        if self.mask_converged:
            if self._update_mask is None:
                self._update_mask = update_mask
            else:
                # some models can have random noise in their predictions, so the mask is updated by
                # keeping all previously converged structures masked even if new force predictions
                # push it slightly above threshold
                self._update_mask = torch.logical_and(self._update_mask, update_mask)
            update_mask = self._update_mask

        return not torch.any(update_mask).item()

    def get_atoms_list(self) -> list[Atoms]:
        """Get ase Atoms objects corresponding to the batch"""
        self._predict()  # in case no predictions have been run
        return self.batch.cpu().to_ase()

    def __len__(self) -> int:
        # TODO: this might be changed in ASE to be 3 * len(self.atoms)
        return len(self.batch.pos)


class OptimizableUnitCellBatch(OptimizableBatch):
    """Modify the supercell and the atom positions in relaxations.

    Based on ase UnitCellFilter to work on data batches
    """

    def __init__(
        self,
        batch: AtomicData,
        predictor: MLIPPredictUnit,
        transform: torch.nn.Module | None = None,
        numpy: bool = False,
        mask_converged: bool = True,
        mask: Sequence[bool] | None = None,
        cell_factor: float | torch.Tensor | None = None,
        hydrostatic_strain: bool = False,
        constant_volume: bool = False,
        scalar_pressure: float = 0.0,
        masked_eps: float = 1e-8,
    ):
        """Create a filter that returns the forces and unit cell stresses together, for simultaneous optimization.

        For full details see:
            E. B. Tadmor, G. S. Smith, N. Bernstein, and E. Kaxiras,
            Phys. Rev. B 59, 235 (1999)

        Args:
            batch: A batch of atoms graph data
            model: An instance of a BaseTrainer derived class
            transform: graph transform
            numpy: whether to cast results to numpy arrays
            mask_converged: if true will mask systems in batch that are already converged
            mask: a boolean mask specifying which strain components are allowed to relax
            cell_factor:
                Factor by which deformation gradient is multiplied to put
                it on the same scale as the positions when assembling
                the combined position/cell vector. The stress contribution to
                the forces is scaled down by the same factor. This can be thought
                of as a very simple preconditioner. Default is number of atoms
                which gives approximately the correct scaling.
            hydrostatic_strain:
                Constrain the cell by only allowing hydrostatic deformation.
                The virial tensor is replaced by np.diag([np.trace(virial)]*3).
            constant_volume:
                Project out the diagonal elements of the virial tensor to allow
                relaxations at constant volume, e.g. for mapping out an
                energy-volume curve. Note: this only approximately conserves
                the volume and breaks energy/force consistency so can only be
                used with optimizers that do require a line minimisation
                (e.g. FIRE).
            scalar_pressure:
                Applied pressure to use for enthalpy pV term. As above, this
                breaks energy/force consistency.
            masked_eps: masking systems that are converged when using ASE optimizers results in divisions by zero
                from zero differences in masked positions at future steps, we add a small number to prevent this.
        """
        super().__init__(
            batch=batch,
            predictor=predictor,
            transform=transform,
            numpy=numpy,
            mask_converged=mask_converged,
            masked_eps=masked_eps,
        )

        self.orig_cells = self.get_cells().clone()
        self.stress = None

        if mask is None:
            mask = torch.eye(3, device=self.device)

        # TODO make sure mask is on GPU
        if mask.shape == (6,):
            self.mask = torch.tensor(
                voigt_6_to_full_3x3_stress(mask.detach().cpu()),
                device=self.device,
            )
        elif mask.shape == (3, 3):
            self.mask = mask
        else:
            raise ValueError("shape of mask should be (3,3) or (6,)")

        if isinstance(cell_factor, float):
            cell_factor = cell_factor * torch.ones(
                (3 * len(batch), 1), requires_grad=False
            )
        if cell_factor is None:
            cell_factor = self.batch.natoms.repeat_interleave(3).unsqueeze(dim=1)

        self.hydrostatic_strain = hydrostatic_strain
        self.constant_volume = constant_volume
        self.pressure = scalar_pressure * torch.eye(3, device=self.device)
        self.cell_factor = cell_factor
        self.stress = None
        self._batch_trace = torch.vmap(torch.trace)
        self._batch_diag = torch.vmap(lambda x: x * torch.eye(3, device=x.device))

    @cached_property
    def batch_indices(self):
        """Get the batch indices specifying which position/force corresponds to which batch.

        We augment this to specify the batch indices for augmented positions and forces.
        """
        augmented_batch = torch.repeat_interleave(
            torch.arange(
                len(self.batch), dtype=self.batch.batch.dtype, device=self.device
            ),
            3,
        )
        return torch.cat([self.batch.batch, augmented_batch])

    def deform_grad(self):
        """Get the cell deformation matrix"""
        return torch.transpose(
            torch.linalg.solve(self.orig_cells, self.get_cells()), 1, 2
        )

    def get_positions(self):
        """Get positions and cell deformation gradient."""
        cur_deform_grad = self.deform_grad()
        natoms = self.batch.num_nodes
        pos = torch.zeros(
            (natoms + 3 * len(self.get_cells()), 3),
            dtype=self.batch.pos.dtype,
            device=self.device,
        )

        # Augmented positions are the self.atoms.positions but without the applied deformation gradient
        pos[:natoms] = torch.linalg.solve(
            cur_deform_grad[self.batch.batch, :, :],
            self.batch.pos.view(-1, 3, 1),
        ).view(-1, 3)
        # cell DOFs are the deformation gradient times a scaling factor
        pos[natoms:] = self.cell_factor * cur_deform_grad.view(-1, 3)
        return pos.cpu().numpy() if self.numpy else pos

    def set_positions(self, positions: torch.Tensor | NDArray):
        """Set positions and cell.

        positions has shape (natoms + ncells * 3, 3).
        the first natoms rows are the positions of the atoms, the last nsystems * three rows are the deformation tensor
        for each cell.
        """
        if isinstance(positions, np.ndarray):
            positions = torch.tensor(positions)

        positions = positions.to(dtype=torch.float32, device=self.device)
        natoms = self.batch.num_nodes
        new_atom_positions = positions[:natoms]
        new_deform_grad = (positions[natoms:] / self.cell_factor.to(self.device)).view(-1, 3, 3)

        # TODO check that in fact symmetry is preserved setting cells and positions
        # Set the new cell from the original cell and the new deformation gradient.  Both current and final structures
        # should preserve symmetry.
        new_cells = torch.bmm(self.orig_cells, torch.transpose(new_deform_grad, 1, 2))
        self.set_cells(new_cells)

        # Set the positions from the ones passed in (which are without the deformation gradient applied) and the new
        # deformation gradient. This should also preserve symmetry
        new_atom_positions = torch.bmm(
            new_atom_positions.view(-1, 1, 3),
            torch.transpose(
                new_deform_grad[self.batch.batch, :, :].view(-1, 3, 3), 1, 2
            ),
        )
        super().set_positions(new_atom_positions.view(-1, 3))

    def get_potential_energy(self, **kwargs):
        """
        returns potential energy including enthalpy PV term.
        """
        atoms_energy = super().get_potential_energy(**kwargs)
        return atoms_energy + self.pressure[0, 0] * self.get_volumes().sum()

    def get_forces(
        self, apply_constraint: bool = False, no_numpy: bool = False
    ) -> torch.Tensor | NDArray:
        """Get forces and unit cell stress."""
        stress = self.get_property("stress", no_numpy=True).view(-1, 3, 3)
        atom_forces = self.get_property("forces", no_numpy=True)

        if apply_constraint:
            fixed_idx = torch.where(self.batch.fixed == 1)[0]
            atom_forces[fixed_idx] = 0.0

        volumes = self.get_volumes().view(-1, 1, 1)
        virial = -volumes * stress + self.pressure.view(-1, 3, 3)
        cur_deform_grad = self.deform_grad()
        atom_forces = torch.bmm(
            atom_forces.view(-1, 1, 3),
            cur_deform_grad[self.batch.batch, :, :].view(-1, 3, 3),
        )
        virial = torch.linalg.solve(
            cur_deform_grad, torch.transpose(virial, dim0=1, dim1=2)
        )
        virial = torch.transpose(virial, dim0=1, dim1=2)

        # TODO this does not work yet! maybe _batch_trace gives an issue
        if self.hydrostatic_strain:
            virial = self._batch_diag(self._batch_trace(virial) / 3.0)

        # Zero out components corresponding to fixed lattice elements
        if (self.mask != 1.0).any():
            virial *= self.mask.view(-1, 3, 3)

        if self.constant_volume:
            virial[:, range(3), range(3)] -= self._batch_trace(virial).view(3, -1) / 3.0

        natoms = self.batch.num_nodes
        augmented_forces = torch.zeros(
            (natoms + 3 * len(self.get_cells()), 3),
            device=self.device,
            dtype=atom_forces.dtype,
        )
        augmented_forces[:natoms] = atom_forces.view(-1, 3)
        augmented_forces[natoms:] = virial.view(-1, 3) / self.cell_factor

        self.stress = -virial.view(-1, 9) / volumes.view(-1, 1)

        if self.numpy and not no_numpy:
            augmented_forces = augmented_forces.cpu().numpy()

        return augmented_forces

    def __len__(self):
        return len(self.batch.pos) + 3 * len(self.batch)


class OptimizableFrechetCellBatch(OptimizableUnitCellBatch):
    """Modify the supercell and the atom positions in relaxations using Fréchet derivatives.

    Based on ase FrechetCellFilter to work on data batches. Uses logarithm of deformation tensor
    for better convergence properties compared to OptimizableUnitCellBatch.
    """

    def __init__(
        self,
        batch: AtomicData,
        predictor: MLIPPredictUnit,
        transform: torch.nn.Module | None = None,
        numpy: bool = False,
        mask_converged: bool = True,
        mask: Sequence[bool] | None = None,
        exp_cell_factor: float | torch.Tensor | None = None,
        hydrostatic_strain: bool = False,
        constant_volume: bool = False,
        scalar_pressure: float = 0.0,
        masked_eps: float = 1e-8,
    ):
        """Create a filter that returns the forces and unit cell stresses together, for simultaneous optimization.

        Uses logarithm of deformation tensor for cell degrees of freedom, providing better convergence
        properties compared to OptimizableUnitCellBatch.

        Args:
            batch: A batch of atoms graph data
            predictor: An instance of a MLIPPredictUnit
            transform: graph transform
            numpy: whether to cast results to numpy arrays
            mask_converged: if true will mask systems in batch that are already converged
            mask: a boolean mask specifying which strain components are allowed to relax
            exp_cell_factor:
                Scaling factor for cell variables. The cell gradients are divided by exp_cell_factor.
                By default, set to the number of atoms. We recommend to set
                an extensive value for this parameter.
            hydrostatic_strain:
                Constrain the cell by only allowing hydrostatic deformation.
                The virial tensor is replaced by diag([trace(virial)]*3).
            constant_volume:
                Project out the diagonal elements of the virial tensor to allow
                relaxations at constant volume, e.g. for mapping out an
                energy-volume curve.
            scalar_pressure:
                Applied pressure to use for enthalpy pV term. As above, this
                breaks energy/force consistency.
            masked_eps: masking systems that are converged when using ASE optimizers results in divisions by zero
                from zero differences in masked positions at future steps, we add a small number to prevent this.
        """
        # Initialize parent with cell_factor parameter (will be overridden)
        super().__init__(
            batch=batch,
            predictor=predictor,
            transform=transform,
            numpy=numpy,
            mask_converged=mask_converged,
            mask=mask,
            cell_factor=1.0,  # Will be overridden by exp_cell_factor
            hydrostatic_strain=hydrostatic_strain,
            constant_volume=constant_volume,
            scalar_pressure=scalar_pressure,
            masked_eps=masked_eps,
        )

        # Scaling factor for cell gradients
        if isinstance(exp_cell_factor, float):
            exp_cell_factor = exp_cell_factor * torch.ones(
                (3 * len(batch), 1), requires_grad=False, device=self.device
            )
        if exp_cell_factor is None:
            exp_cell_factor = self.batch.natoms.repeat_interleave(3).unsqueeze(dim=1)

        self.exp_cell_factor = exp_cell_factor

        # Import scipy functions for matrix operations (similar to reference FrechetCellFilter)
        from scipy.linalg import expm, logm
        self.expm = expm
        self.logm = logm
        
        # AtomicData wraps coordinates, but ASE FrechetCellFilter expects unwrapped coordinates
        # Use original positions if available, otherwise try to unwrap
        if hasattr(self.batch, 'original_positions'):
            self.unwrapped_positions = self.batch.original_positions.to(self.device)
        else:
            # Fallback to unwrapping (may not always be accurate)
            self.unwrapped_positions = self._unwrap_positions(self.batch.pos, self.get_cells())
    
    def _unwrap_positions(self, pos, cells):
        """Unwrap positions from [0, cell] back to original coordinate system.
        
        This reverses the PBC wrapping done by AtomicData.from_ase().
        """
        pos_unwrapped = pos.clone()
        
        # For each coordinate, if it's > cell/2, subtract cell to unwrap
        for i, cell in enumerate(cells):
            # Get lattice vectors (diagonal elements for orthorhombic cells)
            cell_lengths = torch.diag(cell)
            batch_mask = (self.batch.batch == i)
            
            for dim in range(3):
                cell_len = cell_lengths[dim]
                
                # Unwrap coordinates that are > cell_length/2
                atom_coords = pos_unwrapped[batch_mask, dim]
                wrapped_mask = atom_coords > cell_len / 2
                atom_coords[wrapped_mask] -= cell_len
                pos_unwrapped[batch_mask, dim] = atom_coords
                
        return pos_unwrapped

    def get_positions(self):
        """Get positions and logarithm of cell deformation gradient."""
        # Get the current atomic positions (not fractional coordinates from parent)
        natoms = self.batch.num_nodes
        pos = torch.zeros(
            (natoms + 3 * len(self.get_cells()), 3),
            dtype=self.batch.pos.dtype,
            device=self.device,
        )
        
        # Use unwrapped atomic positions to match ASE FrechetCellFilter behavior
        pos[:natoms] = self.unwrapped_positions
        
        # Get the cell DOFs as log of deformation gradient
        cur_deform_grad = self.deform_grad()
        
        # Apply matrix logarithm to each 3x3 matrix using scipy
        cell_dofs_log_list = []
        for i in range(cur_deform_grad.shape[0]):
            deform_matrix_np = cur_deform_grad[i].detach().cpu().numpy()
            deform_log_np = self.logm(deform_matrix_np)
            cell_dofs_log_list.append(torch.tensor(deform_log_np, device=self.device, dtype=torch.float32))

        cell_dofs_log = torch.stack(cell_dofs_log_list)
        # Apply scaling factor per system (exp_cell_factor has shape [3*nsystems, 1], reshape to [nsystems, 3, 1])
        exp_cell_factor_reshaped = self.exp_cell_factor.view(-1, 3, 1)
        cell_dofs_log = cell_dofs_log * exp_cell_factor_reshaped
        pos[natoms:] = cell_dofs_log.view(-1, 3)

        return pos.cpu().numpy() if self.numpy else pos

    def set_positions(self, positions: torch.Tensor | NDArray):
        """Set positions and cell using exponential of log deformation tensor.

        positions has shape (natoms + ncells * 3, 3).
        the first natoms rows are the positions of the atoms, the last nsystems * three rows are the logarithm
        of deformation tensor for each cell.
        """
        if isinstance(positions, np.ndarray):
            positions = torch.tensor(positions, device=self.device, dtype=torch.float32)

        positions = positions.to(dtype=torch.float32, device=self.device)
        natoms = self.batch.num_nodes
        new_atom_positions = positions[:natoms]
        
        # Update stored unwrapped positions
        self.unwrapped_positions = new_atom_positions
        new_deform_grad_log_flat = positions[natoms:] / self.exp_cell_factor.view(-1, 3, 1).view(-1, 1)
        new_deform_grad_log = new_deform_grad_log_flat.view(-1, 3, 3)

        # Convert from log space to deformation gradient using scipy
        new_deform_grad_list = []
        for i in range(new_deform_grad_log.shape[0]):
            log_matrix_np = new_deform_grad_log[i].detach().cpu().numpy()
            exp_matrix_np = self.expm(log_matrix_np)
            new_deform_grad_list.append(torch.tensor(exp_matrix_np, device=self.device, dtype=torch.float32))

        new_deform_grad = torch.stack(new_deform_grad_list).to(self.device)

        # Create augmented positions with the exponential deformation gradient
        augmented_positions = torch.zeros_like(positions)
        augmented_positions[:natoms] = new_atom_positions
        # Reshape cell_factor to match new_deform_grad dimensions
        cell_factor_reshaped = self.cell_factor.view(-1, 3, 1).to(self.device)
        augmented_positions[natoms:] = (cell_factor_reshaped * new_deform_grad).view(-1, 3)

        # Use parent's set_positions with the exponential transformation
        super().set_positions(augmented_positions)

    def get_forces(
        self, apply_constraint: bool = False, no_numpy: bool = False
    ) -> torch.Tensor | NDArray:
        """Get forces and unit cell stress using Fréchet derivatives."""
        stress = self.get_property("stress", no_numpy=True).view(-1, 3, 3)
        atom_forces = self.get_property("forces", no_numpy=True)

        if apply_constraint:
            fixed_idx = torch.where(self.batch.fixed == 1)[0]
            atom_forces[fixed_idx] = 0.0

        volumes = self.get_volumes().view(-1, 1, 1)
        virial = -volumes * stress + self.pressure.view(-1, 3, 3)
        cur_deform_grad = self.deform_grad()

        # Compute matrix logarithm for each system using scipy
        cur_deform_grad_log_list = []
        for i in range(cur_deform_grad.shape[0]):
            deform_matrix_np = cur_deform_grad[i].detach().cpu().numpy()
            log_matrix_np = self.logm(deform_matrix_np)
            cur_deform_grad_log_list.append(torch.tensor(log_matrix_np, device=self.device, dtype=torch.float32))
        cur_deform_grad_log = torch.stack(cur_deform_grad_log_list)

        # Transform atomic forces
        atom_forces = torch.bmm(
            atom_forces.view(-1, 1, 3),
            cur_deform_grad[self.batch.batch, :, :].view(-1, 3, 3),
        )

        if self.hydrostatic_strain:
            virial = self._batch_diag(self._batch_trace(virial) / 3.0)

        # Zero out components corresponding to fixed lattice elements
        if (self.mask != 1.0).any():
            virial *= self.mask.view(-1, 3, 3)

        # Cell gradient for UnitCellFilter (before log transformation)
        ucf_cell_grad = torch.bmm(virial, torch.linalg.inv(torch.transpose(cur_deform_grad, 1, 2)))

        # Cell gradient for FrechetCellFilter using batch matrix exponential Fréchet derivatives
        num_systems = len(self.batch)
        deform_grad_log_force = torch.zeros((num_systems, 3, 3), device=self.device)

        # Use ASE FrechetCellFilter approach (matches reference implementation exactly)
        for i in range(num_systems):
            virial_i = virial[i].detach().cpu().numpy()
            cur_log_i = cur_deform_grad_log[i].detach().cpu().numpy()
            
            # Naive force approach (from ASE line 852)
            deform_grad_log_force_naive = virial_i.copy()
            
            # Complex Fréchet derivative approach (from ASE lines 853-857)
            Y = np.zeros((6, 6))
            Y[0:3, 0:3] = cur_log_i
            Y[3:6, 3:6] = cur_log_i
            Y[0:3, 3:6] = -virial_i @ self.expm(-cur_log_i)
            deform_grad_log_force_complex = -self.expm(Y)[0:3, 3:6]
            
            # Make symmetric (from ASE lines 858-862)
            for i1, i2 in [(0, 1), (0, 2), (1, 2)]:
                ff = 0.5 * (deform_grad_log_force_complex[i1, i2] + deform_grad_log_force_complex[i2, i1])
                deform_grad_log_force_complex[i1, i2] = ff
                deform_grad_log_force_complex[i2, i1] = ff
            
            # Check alignment and choose approach (from ASE lines 864-872)
            all_are_equal = np.allclose(deform_grad_log_force_complex, deform_grad_log_force_naive)
            if all_are_equal:
                final_force = deform_grad_log_force_naive
            else:
                dot_product = np.sum(deform_grad_log_force_complex * deform_grad_log_force_naive)
                complex_norm_sq = np.sum(deform_grad_log_force_complex**2)
                naive_norm_sq = np.sum(deform_grad_log_force_naive**2)
                
                if complex_norm_sq > 0 and naive_norm_sq > 0:
                    alignment = dot_product / np.sqrt(complex_norm_sq * naive_norm_sq)
                    if alignment > 0.8:
                        final_force = deform_grad_log_force_naive
                    else:
                        final_force = deform_grad_log_force_complex
                else:
                    final_force = deform_grad_log_force_naive
            
            deform_grad_log_force[i] = torch.tensor(final_force, device=self.device, dtype=torch.float32)

        # Cauchy stress used for convergence testing
        convergence_crit_stress = -(virial / volumes)

        if self.constant_volume:
            # Apply constraint to force
            dglf_trace = self._batch_trace(deform_grad_log_force).view(-1, 1, 1)
            deform_grad_log_force[:, range(3), range(3)] -= dglf_trace.squeeze() / 3.0
            # Apply constraint to Cauchy stress used for convergence testing
            ccs_trace = self._batch_trace(convergence_crit_stress).view(-1, 1, 1)
            convergence_crit_stress[:, range(3), range(3)] -= ccs_trace.squeeze() / 3.0

        natoms = self.batch.num_nodes
        augmented_forces = torch.zeros(
            (natoms + 3 * len(self.get_cells()), 3),
            device=self.device,
            dtype=atom_forces.dtype,
        )
        augmented_forces[:natoms] = atom_forces.view(-1, 3)
        augmented_forces[natoms:] = deform_grad_log_force.view(-1, 3) / self.exp_cell_factor

        self.stress = convergence_crit_stress.view(-1, 9)

        if self.numpy and not no_numpy:
            augmented_forces = augmented_forces.cpu().numpy()

        return augmented_forces
