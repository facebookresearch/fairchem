"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
import time

import numpy as np
from ase.io import read, write
from ase.io.trajectory import Trajectory

# WatAnalysis
sys.path.insert(0, "/storage/home/mshuaibi/projects/WatAnalysis")
from WatAnalysis.analysis import WaterAnalysis
from WatAnalysis.utils import guess_surface_indices

# SimSoliqTools
sys.path.insert(0, "/storage/home/mshuaibi/projects/SimSoliqTools")
# MDAnalysis
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import (
    HydrogenBondAnalysis,
)
from simsoliq.analyze.density import isolate_solvent_density
from simsoliq.io import init_mdtraj

# Local utilities
sys.path.insert(
    0,
    "/storage/home/mshuaibi/fairchem/src/fairchem/experimental/"
    "foundation_models/data/solvents/release",
)
from utils.constants import (
    PROD_BASE,
    SELECTED_RUNS_8X8,
    STATES,
    SURFACE_TRAJ_8X8_PATH,
)

# ── Constants ─────────────────────────────────────────────────────────────
FRAME_STRIDE = 1
INTERFACE_INTERVAL = (0, 4.5)
DZ = 0.1
ION_Z = {"Li": 3, "K": 19, "Cs": 55}

OUTPUT_BASE = "/checkpoint/ocp/shared/oc25_demos_md/paper/analysis/structural"

_ref = read(SURFACE_TRAJ_8X8_PATH)
SURFACE_AREA = float(np.linalg.norm(np.cross(_ref.cell[0], _ref.cell[1])))

E_CHARGE = 1.60218e-19
ANG2_TO_CM2 = 1e-16


def compute_sigma(ion_count):
    if ion_count == 0:
        return 0.0
    area_cm2 = SURFACE_AREA * ANG2_TO_CM2
    return -ion_count * E_CHARGE * 1e6 / area_cm2


def find_prod_dir(ic, it, b, p):
    matches = sorted(
        glob.glob(
            os.path.join(
                PROD_BASE,
                f"opes-barrier-{b}-pace-{p}-Cu100_8x8" f"-{ic}{it}-production_*",
            )
        )
    )
    return matches[-1] if matches else None


def get_traj_path(ic, it, b, p):
    pd = find_prod_dir(ic, it, b, p)
    if pd is None:
        return None
    tp = os.path.join(pd, "run.traj")
    return tp if os.path.exists(tp) else None


# ── State classification helpers ──────────────────────────────────────────


def get_cc_distance(atoms):
    """
    Compute C-C distance of the adsorbate dimer.

    Identifies the two adsorbate carbon atoms (tag=2, Z=6) and returns
    their minimum-image distance.
    """
    nums = atoms.get_atomic_numbers()
    tags = atoms.get_tags()
    c_idx = np.where((tags == 2) & (nums == 6))[0]
    if len(c_idx) < 2:
        return np.inf
    return atoms.get_distance(c_idx[0], c_idx[1], mic=True)


def load_and_filter_frames(traj_path, state=None, step=FRAME_STRIDE):
    """
    Load trajectory frames with stride, optionally filtering by state.

    Args:
        traj_path: Path to the ASE trajectory file.
        state: If None, return all frames. Otherwise must be a key in
            STATES; only frames whose CC distance falls within the
            state window are returned.
        step: Frame stride for loading.

    Returns:
        List of ASE Atoms objects passing the state filter.
    """
    traj = Trajectory(traj_path)
    n_total = len(traj)
    frame_indices = list(range(0, n_total, step))
    atoms_list = [traj[i] for i in frame_indices]
    print(f"  Loaded {len(atoms_list)}/{n_total} frames (step={step})")

    if state is None:
        return atoms_list

    lo, hi = STATES[state]
    filtered = []
    for atoms in atoms_list:
        d_cc = get_cc_distance(atoms)
        if lo <= d_cc <= hi:
            filtered.append(atoms)
    print(
        f"  State '{state}' [{lo}, {hi}] A: "
        f"{len(filtered)}/{len(atoms_list)} frames"
    )
    return filtered


# ── MDAnalysis helpers ────────────────────────────────────────────────────


def atoms_list_to_mda_universe(atoms_list):
    """
    Convert a list of ASE Atoms to an MDAnalysis Universe.

    Returns:
        (u, tmpfile_path): The Universe and the path to the temp file
        (caller is responsible for cleanup).
    """
    tmpfile = tempfile.NamedTemporaryFile(suffix=".xyz", delete=False)
    write(tmpfile.name, atoms_list, format="extxyz")
    u = mda.Universe(tmpfile.name, format="XYZ")
    cell = atoms_list[0].cell.array
    u.dimensions = [
        cell[0, 0],
        cell[1, 1],
        cell[2, 2],
        90.0,
        90.0,
        90.0,
    ]
    return u, tmpfile.name


def _solvent_selections(atoms):
    """
    Build MDAnalysis index-based selections for solvent O and H only.

    Uses ASE tags to exclude adsorbate atoms (tag != 3) that would
    otherwise be picked up by a generic ``"name O"`` selection.
    """
    nums = atoms.get_atomic_numbers()
    tags = atoms.get_tags()
    solvent_mask = tags == 3
    water_o_idx = np.where(solvent_mask & (nums == 8))[0]
    water_h_idx = np.where(solvent_mask & (nums == 1))[0]
    o_sel = "index " + " ".join(str(i) for i in water_o_idx)
    h_sel = "index " + " ".join(str(i) for i in water_h_idx)
    return o_sel, h_sel


def _adsorbate_oxygen_selection(atoms):
    """
    Build MDAnalysis index-based selection for adsorbate oxygens only.
    """
    nums = atoms.get_atomic_numbers()
    tags = atoms.get_tags()
    ads_o_idx = np.where((tags == 2) & (nums == 8))[0]
    return "index " + " ".join(str(i) for i in ads_o_idx)


def get_surface_indices_from_atoms(atoms, element="Cu", tolerance=1.4):
    try:
        return guess_surface_indices(atoms, element=element, tolerance=tolerance)
    except Exception:
        tags = atoms.get_tags()
        surface_mask = tags == 1
        surface_idx = np.where(surface_mask)[0]
        z_med = np.median(atoms.positions[surface_idx, 2])
        top = surface_idx[atoms.positions[surface_idx, 2] >= z_med]
        bot = surface_idx[atoms.positions[surface_idx, 2] < z_med]
        return [top, bot]


# ── Analysis functions ────────────────────────────────────────────────────


def run_density_simsoliq(traj_path, state=None):
    if state is None:
        a = init_mdtraj(traj_path, fmat="ase")
        densdata = a.get_density_profile(height_axis=2, savepkl=False)
        return isolate_solvent_density(densdata)

    atoms_list = load_and_filter_frames(traj_path, state=state, step=FRAME_STRIDE)
    if len(atoms_list) == 0:
        return None
    tmpfile = tempfile.NamedTemporaryFile(suffix=".traj", delete=False)
    write(tmpfile.name, atoms_list, format="traj")
    try:
        a = init_mdtraj(tmpfile.name, fmat="ase")
        densdata = a.get_density_profile(height_axis=2, savepkl=False)
        return isolate_solvent_density(densdata)
    finally:
        os.unlink(tmpfile.name)


def run_watanalysis(traj_path, ion_type, state=None, n_blocks=10):
    atoms_list = load_and_filter_frames(traj_path, state=state, step=FRAME_STRIDE)
    if len(atoms_list) == 0:
        return None

    atoms0 = atoms_list[0]
    surf_ids = get_surface_indices_from_atoms(atoms0, element="Cu")
    o_sel, h_sel = _solvent_selections(atoms0)
    u, tmpfile = atoms_list_to_mda_universe(atoms_list)

    ion_sel = f"name {ion_type}"
    # Check if ions exist in the universe before tracking
    ion_ag = u.select_atoms(ion_sel)
    species_sels = [ion_sel] if len(ion_ag) > 0 else []

    wa = WaterAnalysis(
        u,
        surf_ids=surf_ids,
        oxygen_sel=o_sel,
        hydrogen_sel=h_sel,
        dz=DZ,
        oh_cutoff=1.3,
        verbose=True,
        species_sels=species_sels,
    )
    wa.run()

    results = {}

    z_dens, rho_mean, rho_err = wa.density_profile(n_blocks=n_blocks)
    results["density_z"] = z_dens
    results["density_rho_mean"] = rho_mean
    results["density_rho_err"] = rho_err

    z_cos, cos_theta = wa.costheta_profile()
    results["costheta_z"] = z_cos
    results["costheta_val"] = cos_theta

    z_orient, rho_cos, rho_cos_err = wa.orientation_profile(n_blocks=n_blocks)
    results["orientation_z"] = z_orient
    results["orientation_rho_cos_theta"] = rho_cos
    results["orientation_rho_cos_theta_err"] = rho_cos_err

    if species_sels:
        try:
            z_ion, rho_ion, rho_ion_err = wa.species_density_profile(
                ion_sel, n_blocks=n_blocks
            )
            results["ion_density_z"] = z_ion
            results["ion_density_rho_mean"] = rho_ion
            results["ion_density_rho_err"] = rho_ion_err
        except Exception as e:
            print(f"    Ion density failed: {e}")

    wc = wa.count_in_region(INTERFACE_INTERVAL)
    results["water_count"] = np.array(wc)

    os.unlink(tmpfile)
    return results


def run_dynamics(traj_path, ion_type, state=None):
    atoms_list = load_and_filter_frames(traj_path, state=state, step=FRAME_STRIDE)
    if len(atoms_list) == 0:
        return None

    atoms0 = atoms_list[0]
    surf_ids = get_surface_indices_from_atoms(atoms0, element="Cu")
    o_sel, h_sel = _solvent_selections(atoms0)
    u, tmpfile = atoms_list_to_mda_universe(atoms_list)

    wa = WaterAnalysis(
        u,
        surf_ids=surf_ids,
        oxygen_sel=o_sel,
        hydrogen_sel=h_sel,
        dz=DZ,
        oh_cutoff=1.3,
        verbose=True,
    )
    wa.run()

    results = {}

    try:
        tau_dip, acf_dip = wa.dipole_autocorrelation(
            max_tau=250,
            delta_tau=1,
            interval=INTERFACE_INTERVAL,
            step=1,
        )
        results["dipole_acf_tau"] = tau_dip
        results["dipole_acf_val"] = acf_dip
    except Exception as e:
        print(f"    Dipole ACF failed: {e}")

    try:
        tau_surv, sp = wa.survival_probability(
            max_tau=500,
            delta_tau=5,
            interval=INTERFACE_INTERVAL,
            step=1,
        )
        results["survival_tau"] = tau_surv
        results["survival_val"] = sp
    except Exception as e:
        print(f"    Survival probability failed: {e}")

    os.unlink(tmpfile)
    return results


def run_angular(traj_path, ion_type, state=None, n_bins=90):
    atoms_list = load_and_filter_frames(traj_path, state=state, step=FRAME_STRIDE)
    if len(atoms_list) == 0:
        return None

    atoms0 = atoms_list[0]
    surf_ids = get_surface_indices_from_atoms(atoms0, element="Cu")
    o_sel, h_sel = _solvent_selections(atoms0)
    u, tmpfile = atoms_list_to_mda_universe(atoms_list)

    wa = WaterAnalysis(
        u,
        surf_ids=surf_ids,
        oxygen_sel=o_sel,
        hydrogen_sel=h_sel,
        dz=DZ,
        oh_cutoff=1.3,
    )
    wa.run()

    grid, hist = wa.angular_distribution(interval=INTERFACE_INTERVAL, n_bins=n_bins)
    os.unlink(tmpfile)
    return {"angular_grid": grid, "angular_hist": hist}


def run_hbond(traj_path, ion_type, state=None):
    """
    Compute hydrogen bond analysis using MDAnalysis.

    Computes two sets of H-bonds:
      1. water-water: H-bonds among solvent water molecules in the
         interfacial layer (z within INTERFACE_INTERVAL of surface).
         Uses update_selections=True so the interfacial subset is
         re-evaluated each frame.
      2. water-adsorbate: H-bonds from water (donor) to adsorbate
         CO oxygens (acceptor).

    Returns count-per-frame time series for each.
    """
    atoms_list = load_and_filter_frames(traj_path, state=state, step=FRAME_STRIDE)
    if len(atoms_list) == 0:
        return None

    atoms0 = atoms_list[0]
    o_sel, h_sel = _solvent_selections(atoms0)
    ads_o_sel = _adsorbate_oxygen_selection(atoms0)
    u, tmpfile = atoms_list_to_mda_universe(atoms_list)

    # Compute surface z for interfacial region bounds
    surf_ids = get_surface_indices_from_atoms(atoms0, element="Cu")
    top_surf_z = float(atoms0.positions[surf_ids[0], 2].max())
    z_lo = top_surf_z + INTERFACE_INTERVAL[0]
    z_hi = top_surf_z + INTERFACE_INTERVAL[1]

    # Interfacial water selections (re-evaluated each frame)
    iface_o_sel = f"({o_sel}) and prop z >= {z_lo:.2f} and prop z <= {z_hi:.2f}"
    iface_h_sel = f"({h_sel}) and prop z >= {z_lo:.2f} and prop z <= {z_hi:.2f}"

    results = {}

    # Water-water H-bonds in interfacial layer
    print(
        f"  Computing interfacial water-water H-bonds "
        f"(z in [{z_lo:.1f}, {z_hi:.1f}])..."
    )
    hba_ww = HydrogenBondAnalysis(
        u,
        donors_sel=iface_o_sel,
        hydrogens_sel=iface_h_sel,
        acceptors_sel=iface_o_sel,
        d_a_cutoff=3.5,
        d_h_a_angle_cutoff=150,
        update_selections=True,
    )
    hba_ww.run(verbose=True)
    results["ww_counts"] = hba_ww.count_by_time()
    results["ww_times"] = hba_ww.times

    # Water -> adsorbate H-bonds (water donates to CO oxygens only)
    print("  Computing water->adsorbate(O) H-bonds...")
    hba_wa = HydrogenBondAnalysis(
        u,
        donors_sel=o_sel,
        hydrogens_sel=h_sel,
        acceptors_sel=ads_o_sel,
        d_a_cutoff=3.5,
        d_h_a_angle_cutoff=150,
        update_selections=False,
    )
    hba_wa.run(verbose=True)
    results["wa_counts"] = hba_wa.count_by_time()
    results["wa_times"] = hba_wa.times

    os.unlink(tmpfile)
    return results


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=("Compute structural analysis for a single OPES config.")
    )
    parser.add_argument(
        "--analysis",
        required=True,
        choices=[
            "density",
            "watanalysis",
            "dynamics",
            "angular",
            "hbond",
        ],
        help="Which analysis to run.",
    )
    parser.add_argument("--ion-type", required=True, choices=["Li", "K", "Cs"])
    parser.add_argument("--ion-count", required=True, type=int)
    parser.add_argument(
        "--state",
        default=None,
        choices=list(STATES.keys()),
        help=(
            "Reaction state to filter frames by C-C distance. "
            "If omitted, all frames are used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_BASE,
        help="Root output directory.",
    )
    args = parser.parse_args()

    # Look up config via flat (ion_count, ion_type) -> (barrier, pace) dict
    key = (args.ion_count, args.ion_type)
    if key not in SELECTED_RUNS_8X8:
        print(f"No config for {args.ion_count}{args.ion_type}")
        sys.exit(1)

    b, p = SELECTED_RUNS_8X8[key]
    ic, it = args.ion_count, args.ion_type
    traj_path = get_traj_path(ic, it, b, p)
    if traj_path is None:
        print(f"No trajectory found for {ic}{it} b={b} p={p}")
        sys.exit(1)

    sigma = compute_sigma(ic)
    state_str = args.state or "all"
    print(f"Analysis: {args.analysis}")
    print(f"State: {state_str}")
    print(f"Config: {ic}{it} (barrier={b}, pace={p})")
    print(f"Trajectory: {traj_path}")
    print(f"Sigma: {sigma:.1f} uC/cm^2")

    out_subdir = os.path.join(args.output_dir, args.analysis)
    os.makedirs(out_subdir, exist_ok=True)
    out_path = os.path.join(out_subdir, f"{it}_{ic}_{state_str}.npz")

    t0 = time.time()

    if args.analysis == "density":
        dens = run_density_simsoliq(traj_path, state=args.state)
        if dens is None:
            print(f"No frames for state '{state_str}', skipping")
            sys.exit(0)
        np.savez(
            out_path,
            sigma=sigma,
            state=state_str,
            binc=dens["binc"],
            **{f"hist_{k}": v for k, v in dens["hists"].items()},
        )

    elif args.analysis == "watanalysis":
        res = run_watanalysis(traj_path, it, state=args.state)
        if res is None:
            print(f"No frames for state '{state_str}', skipping")
            sys.exit(0)
        np.savez(out_path, sigma=sigma, state=state_str, **res)

    elif args.analysis == "dynamics":
        res = run_dynamics(traj_path, it, state=args.state)
        if res is None:
            print(f"No frames for state '{state_str}', skipping")
            sys.exit(0)
        np.savez(out_path, sigma=sigma, state=state_str, **res)

    elif args.analysis == "angular":
        res = run_angular(traj_path, it, state=args.state)
        if res is None:
            print(f"No frames for state '{state_str}', skipping")
            sys.exit(0)
        np.savez(out_path, sigma=sigma, state=state_str, **res)

    elif args.analysis == "hbond":
        res = run_hbond(traj_path, it, state=args.state)
        if res is None:
            print(f"No frames for state '{state_str}', skipping")
            sys.exit(0)
        np.savez(out_path, sigma=sigma, state=state_str, **res)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
