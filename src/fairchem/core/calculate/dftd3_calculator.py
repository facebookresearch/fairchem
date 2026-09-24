"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

ASE calculator wrapper for nvalchemiops DFT-D3(BJ) corrections.
"""

from __future__ import annotations

import io
import re
import tarfile
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
import requests
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

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


# ``a2`` is in Bohr, as expected by the nvalchemiops DFT-D3 kernel.
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


_BOHR_TO_ANGSTROM = 0.529177210544
_ANGSTROM_TO_BOHR = 1.0 / _BOHR_TO_ANGSTROM
_HARTREE_TO_EV = 27.211386245981
_DFTD3_TGZ_URL = "https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/dftd3.tgz"
_DFTD3_TGZ_MD5 = "a76c752e587422c239c99109547516d2"


def _download_dftd3_sources() -> dict[str, str]:
    """Download and verify the reference DFT-D3 Fortran sources."""

    try:
        response = requests.get(_DFTD3_TGZ_URL, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Failed to download the DFT-D3 parameter sources. Provide param_file "
            "to use a local parameter table instead."
        ) from exc
    archive = response.content
    digest = md5(archive, usedforsecurity=False).hexdigest()
    if digest != _DFTD3_TGZ_MD5:
        raise ValueError(
            "DFT-D3 reference archive checksum mismatch: "
            f"expected {_DFTD3_TGZ_MD5}, got {digest}"
        )

    sources = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if member.isfile() and name in {"dftd3.f", "pars.f"}:
                extracted = tar.extractfile(member)
                if extracted is not None:
                    sources[name] = extracted.read().decode("utf-8", errors="ignore")
    missing = {"dftd3.f", "pars.f"} - sources.keys()
    if missing:
        raise RuntimeError(
            "Missing DFT-D3 reference source file(s): " + ", ".join(sorted(missing))
        )
    return sources


def _find_fortran_array(content: str, name: str) -> np.ndarray:
    """Parse a simple ``data NAME / ... /`` array from Fortran source."""

    match = re.search(
        rf"^\s*data\s+{name}\s*/\s*(.*?)\s*/",
        content,
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise ValueError(f"Variable {name!r} not found in DFT-D3 source")
    values = re.findall(r"[-+]?\d+\.\d+(?:_wp)?", match.group(1))
    return np.asarray([float(value.replace("_wp", "")) for value in values])


def _parse_fortran_c6_table(content: str) -> np.ndarray:
    """Parse ``pars`` records containing C6 and coordination references."""

    values: list[float] = []
    in_record = False
    for line in content.splitlines():
        if "pars(" in line.lower() and "=(" in line:
            in_record = True
        if not in_record:
            continue
        data_line = line[: line.index("!")] if "!" in line else line
        values.extend(
            float(value.replace("D", "e").replace("d", "e"))
            for value in re.findall(r"[-+]?\d+\.\d+[eEdD][-+]?\d+", data_line)
        )
        if "/)" in line:
            in_record = False
    if len(values) % 5:
        raise ValueError("Malformed DFT-D3 C6 parameter table")
    return np.asarray(values).reshape(-1, 5)


def _decode_dftd3_element(encoded: int) -> tuple[int, int]:
    cn_index = 1
    while encoded > 100:
        encoded -= 100
        cn_index += 1
    return encoded, cn_index


def _extract_dftd3_parameters() -> dict[str, torch.Tensor]:
    """Build the tensor table expected by ``nvalchemiops.dftd3``."""

    sources = _download_dftd3_sources()
    r4r2_values = _find_fortran_array(sources["dftd3.f"], "r2r4")
    rcov_values = _find_fortran_array(sources["dftd3.f"], "rcov")
    records = _parse_fortran_c6_table(sources["pars.f"])

    r4r2 = np.zeros(95, dtype=np.float32)
    rcov = np.zeros(95, dtype=np.float32)
    r4r2[1:] = r4r2_values.astype(np.float32)
    rcov[1:] = rcov_values.astype(np.float32)
    c6ab = np.zeros((95, 95, 5, 5), dtype=np.float32)
    cn_ref = np.full((95, 95, 5, 5), -1.0, dtype=np.float32)
    cn_values: dict[int, dict[int, float]] = {element: {} for element in range(95)}

    for c6, encoded_i, encoded_j, cn_i, cn_j in records:
        element_i, index_i = _decode_dftd3_element(int(encoded_i))
        element_j, index_j = _decode_dftd3_element(int(encoded_j))
        if not (1 <= element_i <= 94 and 1 <= element_j <= 94):
            continue
        if not (1 <= index_i <= 5 and 1 <= index_j <= 5):
            continue
        index_i -= 1
        index_j -= 1
        c6ab[element_i, element_j, index_i, index_j] = c6
        c6ab[element_j, element_i, index_j, index_i] = c6
        cn_values[element_i].setdefault(index_i, cn_i)
        cn_values[element_j].setdefault(index_j, cn_j)

    for element in range(1, 95):
        for index, value in cn_values[element].items():
            cn_ref[element, :, index, :] = value

    return {
        "rcov": torch.from_numpy(rcov),
        "r4r2": torch.from_numpy(r4r2),
        "c6ab": torch.from_numpy(c6ab),
        "cn_ref": torch.from_numpy(cn_ref),
    }


def _load_dftd3_parameters(
    param_file: str | PathLike[str] | None, auto_download: bool
) -> dict[str, torch.Tensor]:
    """Load the D3 table, generating the standard cache on first use."""

    path = (
        Path(param_file)
        if param_file is not None
        else Path.home() / ".cache" / "nvalchemiops" / "dftd3_parameters.pt"
    )
    if not path.exists():
        if not auto_download:
            raise FileNotFoundError(f"DFT-D3 parameter file not found: {path}")
        parameters = _extract_dftd3_parameters()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(parameters, path)

    parameters = torch.load(path, map_location="cpu", weights_only=True)
    required = {"rcov", "r4r2", "c6ab", "cn_ref"}
    if not isinstance(parameters, dict) or not required <= parameters.keys():
        raise ValueError(
            f"Invalid DFT-D3 parameter file {path}; expected keys {sorted(required)}"
        )
    return parameters


def _import_nvalchemiops():
    try:
        from nvalchemiops.torch.interactions.dispersion import D3Parameters, dftd3
        from nvalchemiops.torch.neighbors import neighbor_list
    except ImportError as exc:
        raise ImportError(
            "DFTD3Calculator requires nvalchemi-toolkit-ops. Reinstall or update "
            "fairchem-core to restore its required dependencies."
        ) from exc
    return D3Parameters, dftd3, neighbor_list


class _NValChemiDFTD3Calculator(Calculator):
    """ASE adapter around nvalchemiops' analytic DFT-D3(BJ) kernel."""

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
        if not 0.0 <= smoothing_fraction < 1.0:
            raise ValueError(
                "smoothing_fraction must be in [0, 1), got " f"{smoothing_fraction!r}"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.functional = functional
        self.damping_parameters = _DFTD3_BJ_PARAMETERS[functional]
        self.cutoff = float(cutoff)
        self.smoothing_fraction = float(smoothing_fraction)

        d3_parameters_cls, self._dftd3, self._neighbor_list = _import_nvalchemiops()
        parameters = _load_dftd3_parameters(param_file, auto_download)
        self._d3_parameters = d3_parameters_cls(**parameters).to(
            device=self.device, dtype=torch.float32
        )
        self._positions = None
        self._numbers = None
        self._cell = None
        self._pbc = None
        self._neighbor_matrix = None
        self._num_neighbors = None
        self._neighbor_matrix_shifts = None
        self._structure_key = None

    def _prepare_inputs(self, atoms: Atoms) -> bool:
        structure_changed = self._structure_key is None or not (
            np.array_equal(atoms.numbers, self._structure_key[0])
            and np.array_equal(atoms.pbc, self._structure_key[1])
        )
        if self._positions is None or structure_changed:
            self._positions = torch.as_tensor(
                atoms.positions, dtype=torch.float32, device=self.device
            )
            self._numbers = torch.as_tensor(
                atoms.numbers, dtype=torch.int32, device=self.device
            )
            self._pbc = torch.as_tensor(atoms.pbc, dtype=torch.bool, device=self.device)
            self._cell = torch.as_tensor(
                atoms.cell.array, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            self._neighbor_matrix = None
            self._num_neighbors = None
            self._neighbor_matrix_shifts = None
            self._structure_key = (atoms.numbers.copy(), atoms.pbc.copy())
        else:
            self._positions.copy_(
                torch.as_tensor(
                    atoms.positions,
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            self._cell.copy_(
                torch.as_tensor(
                    atoms.cell.array,
                    dtype=torch.float32,
                    device=self.device,
                )
            )

        periodic = bool(np.any(atoms.pbc))
        neighbor_kwargs = {}
        if self._neighbor_matrix is not None:
            neighbor_kwargs = {
                "neighbor_matrix": self._neighbor_matrix,
                "num_neighbors": self._num_neighbors,
            }
            if periodic:
                neighbor_kwargs["neighbor_matrix_shifts"] = self._neighbor_matrix_shifts
        neighbor_result = self._neighbor_list(
            positions=self._positions,
            cutoff=self.cutoff,
            cell=self._cell if periodic else None,
            pbc=self._pbc.unsqueeze(0) if periodic else None,
            half_fill=False,
            fill_value=len(atoms),
            **neighbor_kwargs,
        )
        self._neighbor_matrix = neighbor_result[0]
        self._num_neighbors = neighbor_result[1]
        self._neighbor_matrix_shifts = neighbor_result[2] if periodic else None
        return periodic

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        if self.atoms is None or len(self.atoms) == 0:
            raise ValueError("Atoms object has no atoms inside.")

        # Reuse input and neighbor-list buffers, but rebuild the actual skin=0
        # neighbor list from current positions and cell on every call.
        # This is safe for variable-cell simulations such as NPT dynamics.
        periodic = self._prepare_inputs(self.atoms)

        with torch.inference_mode():
            output = self._dftd3(
                positions=self._positions * _ANGSTROM_TO_BOHR,
                numbers=self._numbers,
                a1=self.damping_parameters.a1,
                a2=self.damping_parameters.a2,
                s6=self.damping_parameters.s6,
                s8=self.damping_parameters.s8,
                k1=16.0,
                k3=-4.0,
                s5_smoothing_on=(
                    self.cutoff * (1.0 - self.smoothing_fraction) * _ANGSTROM_TO_BOHR
                ),
                s5_smoothing_off=self.cutoff * _ANGSTROM_TO_BOHR,
                fill_value=len(self.atoms),
                d3_params=self._d3_parameters,
                cell=self._cell * _ANGSTROM_TO_BOHR if periodic else None,
                neighbor_matrix=self._neighbor_matrix,
                neighbor_matrix_shifts=(
                    self._neighbor_matrix_shifts if periodic else None
                ),
                compute_virial=periodic,
                num_systems=1,
            )

        energy = float(output[0].reshape(-1)[0].detach().cpu()) * _HARTREE_TO_EV
        forces = output[1].detach().cpu().numpy().astype(np.float64, copy=False) * (
            _HARTREE_TO_EV / _BOHR_TO_ANGSTROM
        )
        if not (np.isfinite(energy) and np.isfinite(forces).all()):
            raise FloatingPointError("Non-finite DFT-D3 energy or force")

        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
        }
        if periodic:
            stress = (
                (-output[3] * (_HARTREE_TO_EV / self.atoms.get_volume()))
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


class DFTD3Calculator(Calculator):
    """Add an nvalchemiops DFT-D3(BJ) correction to an ASE calculator.

    The named functional selects the damping parameters associated with the
    level of theory used to train the wrapped calculator. The default 15 Å
    neighbor list is rebuilt for each changed atomic configuration, including
    cell changes, and nvalchemiops supplies analytic energy, forces, and stress.

    Args:
        calculator: Base ASE calculator whose predictions receive the D3 term.
        functional: DFT-D3(BJ) parameterization, either ``"r2scan"`` or
            ``"pbe"``.
        device: Torch device for D3. Defaults to CUDA when available, otherwise
            CPU.
        cutoff: D3 neighbor cutoff in Angstrom. Defaults to 15 Å.
        smoothing_fraction: Fraction of the cutoff over which C5 smoothing is
            applied. Defaults to 0.2 (the outer 20% of the cutoff).
        param_file: Optional local D3 parameter-table file.
        auto_download: Allow FairChem to download and cache the parameter
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
        super().__init__()
        self.base_calculator = calculator
        self.dispersion_calculator = _NValChemiDFTD3Calculator(
            functional,
            device=device,
            cutoff=cutoff,
            smoothing_fraction=smoothing_fraction,
            param_file=param_file,
            auto_download=auto_download,
        )
        dispersion_properties = set(self.dispersion_calculator.implemented_properties)
        self.implemented_properties = [
            prop
            for prop in self.base_calculator.implemented_properties
            if prop in dispersion_properties
        ]

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        """Evaluate each calculator once and directly add common results."""

        super().calculate(atoms, properties, system_changes)
        if self.atoms is None:
            raise ValueError("An Atoms object is required.")
        if properties is None:
            properties = self.implemented_properties

        self.base_calculator.calculate(self.atoms, properties, system_changes)
        self.dispersion_calculator.calculate(self.atoms, properties, system_changes)

        self.results = {}
        for prop in self.implemented_properties:
            if not (
                prop in self.base_calculator.results
                and prop in self.dispersion_calculator.results
            ):
                continue
            base_result = self.base_calculator.results[prop]
            dispersion_result = self.dispersion_calculator.results[prop]
            if prop == "stress" and np.shape(base_result) != np.shape(
                dispersion_result
            ):
                if np.shape(base_result) == (3, 3):
                    base_result = full_3x3_to_voigt_6_stress(base_result)
                if np.shape(dispersion_result) == (3, 3):
                    dispersion_result = full_3x3_to_voigt_6_stress(dispersion_result)
            self.results[prop] = base_result + dispersion_result
            self.results[f"{prop}_contributions"] = [
                base_result,
                dispersion_result,
            ]

    @property
    def functional(self) -> str:
        """DFT functional associated with the selected damping parameters."""

        return self.dispersion_calculator.functional

    @property
    def damping_parameters(self) -> _DFTD3Parameters:
        """Selected DFT-D3(BJ) damping parameters."""

        return self.dispersion_calculator.damping_parameters


__all__ = ["DFTD3Calculator"]
