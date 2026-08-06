"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

# =============================================================================
# Compaction fidelity check for the OMol sphere/per-m Pareto-front candidates.
#
# For each candidate run we load its final EMA inference checkpoint (the trained
# pruned model at full width C with the dropped channels zeroed), physically
# compact it to a native sphere_channels=K model (scripts/compact_channels.py),
# and measure how far the compacted model's energies/forces drift from the pruned
# model's on a batch of REAL OMol val structures. Compaction is viable when the
# drift is ~0 (Route-B / norm-stats-exact) -- i.e. the deployed native checkpoint
# reproduces the trained pruned model, so its val metrics are unchanged.
#
#   python scripts/compaction_fidelity.py --n 64 [--device cpu|cuda]
# =============================================================================
import argparse
import copy
import importlib.util
from pathlib import Path

import torch
from omegaconf import OmegaConf

from fairchem.core.datasets.ase_datasets import AseDBDataset
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit.utils import load_inference_model

_CC = Path(__file__).parent / "compact_channels.py"
_spec = importlib.util.spec_from_file_location("compact_channels", _CC)
compact_channels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compact_channels)

RB = "/checkpoint/amaia/explore/lvj/uma_pat_runs"
VAL = "/checkpoint/amaia/explore/lvj/datasets/omol/250430-release/val"

# label -> (run_dir, recorded pre-compaction val E MAE, F MAE) for context
CANDIDATES = {
    "C=112 (s0.125)": ("202607-2903-2146-c796", 0.1487, 0.0133),
    "C=96 (s0.25)": ("202607-2904-0544-b1cd", 0.1558, 0.0138),
    "C=64+m1 (s0.5,mmax1)": ("202607-2914-5757-364a", 0.1748, 0.0157),
    "C=112+m1 (192->112)": ("202608-0316-0804-add8", 0.1491, 0.0138),
    "C=96+m1 (192->96)": ("202608-0316-0825-a0fe", 0.1551, 0.0141),
    "C=80+m1 (192->80)": ("202608-0316-0844-bd50", 0.1626, 0.0148),
    "C=64+m1 (192->64)": ("202607-3015-2616-49e3", None, None),
}


def load_structs(n, device):
    ds = AseDBDataset(config={"src": VAL})
    step = max(1, len(ds) // n)
    idx = list(range(0, len(ds), step))[:n]
    out = []
    for i in idx:
        a = ds.get_atoms(i)
        a.info.setdefault("charge", 0)
        a.info.setdefault("spin", 1)
        d = AtomicData.from_ase(
            a,
            radius=6,
            max_neigh=300,
            task_name="omol",
            r_edges=False,
            r_data_keys=["spin", "charge"],
        )
        out.append(d.to(device))
    return out


def predict(model, structs):
    es, fs, nat = [], [], []
    for d in structs:
        o = model(d)
        es.append(float(o["omol_energy"]["energy"].detach().reshape(-1)[0]))
        fs.append(o["omol_forces"]["forces"].detach())
        nat.append(d.pos.shape[0])
    return es, fs, nat


def run_candidate(label, run_dir, device, structs):
    ckpt_path = f"{RB}/{run_dir}/checkpoints/final/inference_ckpt.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ema, _ = load_inference_model(ckpt_path, use_ema=True, strict=True)
    raw = ema.module
    raw.eval().to(device)

    keep = compact_channels.infer_keep_channels(raw, "sphere")
    cfg = OmegaConf.create(copy.deepcopy(ckpt.model_config))
    compacted, report = compact_channels.compact(raw, cfg, mode="sphere", keep=keep)
    compacted.eval().to(device)

    e0, f0, nat = predict(raw, structs)
    e1, f1, _ = predict(compacted, structs)

    # per-structure energy drift (total eV and meV/atom); force drift (max comp)
    de = [abs(a - b) for a, b in zip(e0, e1)]
    de_pa = [d / n * 1000.0 for d, n in zip(de, nat)]
    dfrc = [float((a - b).abs().max()) for a, b in zip(f0, f1)]
    return {
        "label": label,
        "C_full": report["params_before"],
        "kept": report["kept"],
        "param_reduction": report["param_reduction"],
        "maxdE_eV": max(de),
        "meandE_meV_atom": sum(de_pa) / len(de_pa),
        "maxdF": max(dfrc),
        "meandF": sum(dfrc) / len(dfrc),
        "sphere_full": raw.backbone.sphere_channels,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", default=None, help="substring filter on label")
    args = ap.parse_args()

    structs = load_structs(args.n, args.device)
    print(f"loaded {len(structs)} val structures on {args.device}\n")
    print(
        f"{'candidate':<24}{'K':>5}{'param_red':>11}"
        f"{'maxΔE(eV)':>12}{'ΔE(meV/at)':>12}{'maxΔF':>11}{'meanΔF':>11}"
    )
    rows = []
    for label, (rd, _e, _f) in CANDIDATES.items():
        if args.only and args.only not in label:
            continue
        r = run_candidate(label, rd, args.device, structs)
        rows.append(r)
        print(
            f"{r['label']:<24}{r['kept']:>5}{r['param_reduction']:>10.1%}"
            f"{r['maxdE_eV']:>12.2e}{r['meandE_meV_atom']:>12.3f}"
            f"{r['maxdF']:>11.2e}{r['meandF']:>11.2e}"
        )
    print(
        "\nviable if maxΔF << recorded F MAE (~0.013-0.016) and ΔE(meV/atom) small "
        "(sub-meV = exact); a large ΔE/ΔF flags a Route-A norm-centering gap."
    )


if __name__ == "__main__":
    main()
