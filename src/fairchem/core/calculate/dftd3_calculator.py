"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

ASE calculator wrapper for nvalchemi DFT-D3(BJ) corrections.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.mixing import SumCalculator

if TYPE_CHECKING:
    from os import PathLike

    from ase import Atoms


@dataclass(frozen=True, slots=True)
class _DFTD3Parameters:
    """Becke-Johnson damping parameters used by DFT-D3."""

    a1: float
    a2: float
    s6: float
    s8: float


# ``a2`` is in Bohr, as expected by nvalchemi's DFTD3ModelWrapper.
#
# PBE-D3(BJ): S. Grimme, S. Ehrlich, and L. Goerigk, J. Comput. Chem. 32,
# 1456-1465 (2011), https://doi.org/10.1002/jcc.21759.
#
# r2SCAN-D3(BJ): S. Ehlert et al., J. Chem. Phys. 154, 061101 (2021),
# https://doi.org/10.1063/5.0041008. Although the paper focuses on r2SCAN-D4,
# it also reports and benchmarks the r2SCAN-D3(BJ) parameterization used here.
_DFTD3_BJ_PARAMETERS: dict[str, _DFTD3Parameters] = {
    "r2scan": _DFTD3Parameters(
        a1=0.49484001,
        a2=5.73083694,
        s6=1.0,
        s8=0.78981345,
    ),
    "pbe": _DFTD3Parameters(
        a1=0.4289,
        a2=4.4407,
        s6=1.0,
        s8=0.7875,
    ),
}


def _import_nvalchemi():
    try:
        from nvalchemi.data import AtomicData, Batch
        from nvalchemi.models.dftd3 import DFTD3ModelWrapper
        from nvalchemi.neighbors import compute_neighbors
    except ImportError as exc:
        raise ImportError(
            "DFTD3Calculator requires nvalchemi-toolkit. Reinstall or update "
            "fairchem-core to restore its required dependencies."
        ) from exc
    return DFTD3ModelWrapper, AtomicData, Batch, compute_neighbors


class _NVAlchemiDFTD3Calculator(Calculator):
    """ASE adapter around nvalchemi's analytic DFT-D3(BJ) implementation."""

    implemented_properties: ClassVar[list[str]] = [
        "energy",
        "free_energy",
        "forces",
        "stress",
    ]

    def __init__(
        self,
        functional: Literal["r2scan", "pbe"],
        *,
        device: str | torch.device | None,
        cutoff: float,
        smoothing_fraction: float,
        param_file: str | PathLike[str] | None,
        auto_download: bool,
    ) -> None:
        super().__init__()

        functional = functional.lower()
        if functional not in _DFTD3_BJ_PARAMETERS:
            choices = ", ".join(sorted(_DFTD3_BJ_PARAMETERS))
            raise ValueError(
                f"Unknown DFT-D3 functional {functional!r}; choose one of: {choices}"
            )
        if cutoff <= 0.0:
            raise ValueError(f"cutoff must be positive, got {cutoff!r}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.functional = functional
        self.damping_parameters = _DFTD3_BJ_PARAMETERS[functional]
        self.cutoff = float(cutoff)
        self.smoothing_fraction = float(smoothing_fraction)

        model_cls, self._atomic_data_cls, self._batch_cls, self._compute_neighbors = (
            _import_nvalchemi()
        )
        self.model = model_cls(
            **asdict(self.damping_parameters),
            k1=16.0,
            k3=-4.0,
            cutoff=self.cutoff,
            smoothing_fraction=self.smoothing_fraction,
            param_file=param_file,
            auto_download=auto_download,
        ).to(self.device)
        self.model.set_config("active_outputs", {"energy", "forces"})
        self.model.eval()

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        if self.atoms is None or len(self.atoms) == 0:
            raise ValueError("Atoms object has no atoms inside.")

        # Rebuild from the current positions and cell on every calculation.
        # This gives skin=0 behavior and avoids stale periodic-image shifts in
        # variable-cell simulations such as NPT molecular dynamics.
        data = self._atomic_data_cls.from_atoms(
            self.atoms,
            device=self.device,
            dtype=torch.float32,
        )
        batch = self._batch_cls.from_data_list(
            [data], device=self.device, skip_validation=True
        )
        self._compute_neighbors(batch, config=self.model.model_config.neighbor_config)

        actual_cutoff = float(batch._neighbor_list_cutoff)
        if not np.isclose(actual_cutoff, self.cutoff, rtol=0.0, atol=1.0e-12):
            raise RuntimeError(
                "nvalchemi built a neighbor list with cutoff "
                f"{actual_cutoff} A; expected {self.cutoff} A"
            )

        active_outputs = {"energy", "forces"}
        if np.any(self.atoms.pbc):
            active_outputs.add("stress")
        self.model.set_config("active_outputs", active_outputs)

        with torch.inference_mode():
            output = self.model(batch)

        energy = float(output["energy"].reshape(-1)[0].detach().cpu())
        forces = output["forces"].detach().cpu().numpy().astype(np.float64, copy=False)
        if not (np.isfinite(energy) and np.isfinite(forces).all()):
            raise FloatingPointError("Non-finite DFT-D3 energy or force")

        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
        }
        if "stress" in output:
            stress = (
                output["stress"]
                .reshape(3, 3)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
            # Numerical kernels can introduce tiny antisymmetric components.
            # ASE expects symmetric Cauchy stress in xx, yy, zz, yz, xz, xy
            # order.
            stress = 0.5 * (stress + stress.T)
            stress_voigt = stress.flat[[0, 4, 8, 5, 2, 1]]
            if not np.isfinite(stress_voigt).all():
                raise FloatingPointError("Non-finite DFT-D3 stress")
            self.results["stress"] = stress_voigt


class DFTD3Calculator(SumCalculator):
    """Add an nvalchemi DFT-D3(BJ) correction to an ASE calculator.

    The named functional selects the damping parameters associated with the
    level of theory used to train the wrapped calculator. The default 15 Å
    neighbor list is rebuilt for each changed atomic configuration, including
    cell changes, and nvalchemi supplies analytic energy, forces, and stress.

    Args:
        calculator: Base ASE calculator whose predictions receive the D3 term.
        functional: DFT-D3(BJ) parameterization, either ``"r2scan"`` or
            ``"pbe"``.
        device: Torch device for D3. Defaults to CUDA when available, otherwise
            CPU.
        cutoff: D3 neighbor cutoff in Angstrom. Defaults to 15 Å.
        smoothing_fraction: Fraction of the cutoff over which C5 smoothing is
            applied. Defaults to 0.2 (the outer 20% of the cutoff).
        param_file: Optional local nvalchemi D3 parameter-table file.
        auto_download: Allow nvalchemi to download and cache its parameter
            table when ``param_file`` is not supplied.
    """

    def __init__(
        self,
        calculator: Calculator,
        functional: Literal["r2scan", "pbe"],
        *,
        device: str | torch.device | None = None,
        cutoff: float = 15.0,
        smoothing_fraction: float = 0.2,
        param_file: str | PathLike[str] | None = None,
        auto_download: bool = True,
    ) -> None:
        self.base_calculator = calculator
        self.dispersion_calculator = _NVAlchemiDFTD3Calculator(
            functional,
            device=device,
            cutoff=cutoff,
            smoothing_fraction=smoothing_fraction,
            param_file=param_file,
            auto_download=auto_download,
        )
        super().__init__([self.base_calculator, self.dispersion_calculator])

    @property
    def functional(self) -> str:
        """DFT functional associated with the selected damping parameters."""

        return self.dispersion_calculator.functional

    @property
    def damping_parameters(self) -> _DFTD3Parameters:
        """Selected DFT-D3(BJ) damping parameters."""

        return self.dispersion_calculator.damping_parameters


__all__ = ["DFTD3Calculator"]
