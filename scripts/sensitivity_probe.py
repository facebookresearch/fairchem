"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

# =============================================================================
# Structured per-module sensitivity probe (one-shot, NO training).
#
# For a trained dense checkpoint, ablate a fixed fraction of the LOWEST-L2 output
# units of each module (one module at a time) and measure how far the model's
# predictions move (ΔE, mean|ΔF|) on a small calibration set. Modules with a small
# delta tolerate sparsity; large-delta modules are sensitive. This produces a
# per-module sensitivity map to guide non-uniform structured pruning ratios.
#
# Caveats (see the write-up): (1) this measures OUTPUT PERTURBATION vs the dense
# model (a label-free proxy for the val-loss increase pruning would cause), not
# true val loss; (2) it ablates output units by L2, an indicative ranking -- the
# only *compactable* structured cut is a shared width (global sphere_channels C,
# per-block hidden_channels H), so read this as "where is the redundancy", not a
# drop-in ratio. The `sphere-global` row uses the real channel-correct sphere mask
# as an anchor.
#
#   python scripts/sensitivity_probe.py dense_inference.pt [--frac 0.5]
# =============================================================================
import argparse

import torch
from ase.build import molecule as get_molecule

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.models.uma import channel_pruning as CP
from fairchem.core.units.mlip_unit.utils import load_inference_model


def calib_structures():
    names = ["CH3CH2OH", "C6H6", "CH3COOH", "CH3CN"]
    out = []
    for n in names:
        a = get_molecule(n)
        a.info["charge"] = 0
        a.info["spin"] = 1
        out.append(
            AtomicData.from_ase(
                a,
                radius=6,
                max_neigh=300,
                task_name="omol",
                r_edges=False,
                r_data_keys=["spin", "charge"],
            )
        )
    return out


def predict(model, structs):
    """Return (energies[list], forces[list]) for the calibration structures."""
    es, fs = [], []
    for data in structs:
        out = model(data)
        es.append(out["omol_energy"]["energy"].detach().reshape(-1))
        fs.append(out["omol_forces"]["forces"].detach())
    return es, fs


def delta(base, new):
    (e0, f0), (e1, f1) = base, new
    de = torch.stack([(a - b).abs().mean() for a, b in zip(e0, e1)]).mean()
    dforce = torch.stack([(a - b).abs().mean() for a, b in zip(f0, f1)]).mean()
    return float(de), float(dforce)


def lowest_l2_slices(w: torch.Tensor, axis: int, frac: float) -> torch.Tensor:
    """Indices of the lowest-L2 `frac` slices along `axis`."""
    dims = [d for d in range(w.dim()) if d != axis]
    score = w.detach().pow(2).sum(dim=dims).sqrt()  # [size(axis)]
    k = max(1, int(round(frac * score.numel())))
    return torch.argsort(score)[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument(
        "--composite",
        action="store_true",
        help="also evaluate mixed sensitivity-informed allocations (one-shot)",
    )
    ap.add_argument(
        "--sph-axes",
        action="store_true",
        help="also ablate angular axes: per-degree (l) and per-m-order sensitivity",
    )
    args = ap.parse_args()

    model, _ = load_inference_model(args.ckpt, use_ema=True, strict=True)
    raw = model.module  # HydraModel
    raw.eval()
    bb = raw.backbone
    nlayers = bb.num_layers

    structs = calib_structures()
    base = predict(raw, structs)
    print(f"calibration: {len(structs)} molecules; ablation frac={args.frac}")
    print(f"{'module':<26}{'ΔE (eV)':>12}{'mean|ΔF| (eV/Å)':>18}")

    # --- per-module output-unit ablation targets ---
    params = dict(raw.named_parameters())
    targets = []
    for i in range(nlayers):
        p = f"backbone.blocks.{i}.edge_wise"
        aw = f"backbone.blocks.{i}.atom_wise"
        targets += [
            (f"blk{i}.conv1_out(H)", f"{p}.so2_conv_1.fc_m0.weight", 0),
            (f"blk{i}.conv2_out(C)", f"{p}.so2_conv_2.fc_m0.weight", 0),
            (f"blk{i}.radial_out", f"{p}.so2_conv_1.rad_func.net.6.weight", 0),
            (f"blk{i}.atomwise", f"{aw}.so3_linear_1.weight", 1),
        ]
    targets += [
        ("mix_csd_out(C)", "backbone.mix_csd.weight", 0),
        ("sphere_embed(C)", "backbone.sphere_embedding.weight", 1),
        (
            "edge_degree_radial",
            "backbone.edge_degree_embedding.rad_func.net.6.weight",
            0,
        ),
    ]

    rows = []
    for name, pname, axis in targets:
        if pname not in params:
            continue
        w = params[pname]
        orig = w.detach().clone()
        idx = lowest_l2_slices(w, axis, args.frac)
        with torch.no_grad():
            w.index_fill_(axis, idx, 0.0)
        de, df = delta(base, predict(raw, structs))
        with torch.no_grad():
            w.copy_(orig)
        rows.append((name, de, df))

    # --- anchor: channel-correct GLOBAL sphere ablation (the real compactable cut) ---
    imp = CP.channel_importance(raw, "sphere")
    drop = torch.argsort(imp)[: max(1, int(round(args.frac * imp.numel())))]
    saved = {n: p.detach().clone() for n, p in raw.named_parameters()}
    CP.apply_channel_mask(raw, "sphere", drop)
    de, df = delta(base, predict(raw, structs))
    with torch.no_grad():
        for n, p in raw.named_parameters():
            p.copy_(saved[n])
    rows.append(("sphere-GLOBAL(C) [anchor]", de, df))

    for name, de, df in sorted(rows, key=lambda r: r[2]):
        print(f"{name:<26}{de:>12.4f}{df:>18.5f}")
    print("\n(sorted least→most sensitive by mean|ΔF|)")

    if args.composite:
        run_composite(raw, structs, base, nlayers)
    if args.sph_axes:
        natoms = [int(s.pos.shape[0]) for s in structs]
        run_sph_axes(raw, structs, base, bb.lmax, bb.mmax, natoms)


def _ff_ops(nlayers, frac):
    """Aggressive trim of the spectral-FF (atom_wise) output width, all blocks."""
    return [
        (f"backbone.blocks.{i}.atom_wise.so3_linear_1.weight", 1, frac)
        for i in range(nlayers)
    ]


@torch.no_grad()
def _apply_ops(raw, ops):
    """Apply a list of ops; op = ('sphere', frac) or (param_name, axis, frac)."""
    params = dict(raw.named_parameters())
    for op in ops:
        if op[0] == "sphere":
            imp = CP.channel_importance(raw, "sphere")
            drop = torch.argsort(imp)[: max(1, int(round(op[1] * imp.numel())))]
            CP.apply_channel_mask(raw, "sphere", drop)
        else:
            pname, axis, frac = op
            w = params[pname]
            w.index_fill_(axis, lowest_l2_slices(w, axis, frac), 0.0)


def run_composite(raw, structs, base, nlayers):
    """Compare sensitivity-informed mixed allocations vs uniform C cuts (one-shot)."""
    allocs = {
        "C-only @0.25": [("sphere", 0.25)],
        "C-only @0.50": [("sphere", 0.50)],
        "FF-only @0.60": _ff_ops(nlayers, 0.60),
        "mixed: C@0.25 + FF@0.60": [("sphere", 0.25), *_ff_ops(nlayers, 0.60)],
        "mixed: C@0.35 + FF@0.60": [("sphere", 0.35), *_ff_ops(nlayers, 0.60)],
    }
    snapshot = {n: p.detach().clone() for n, p in raw.named_parameters()}
    print(f"\n{'composite allocation':<28}{'ΔE (eV)':>12}{'mean|ΔF| (eV/Å)':>18}")
    for name, ops in allocs.items():
        _apply_ops(raw, ops)
        de, df = delta(base, predict(raw, structs))
        with torch.no_grad():
            for n, p in raw.named_parameters():
                p.copy_(snapshot[n])
        print(f"{name:<28}{de:>12.4f}{df:>18.5f}")
    print(
        "\n(one-shot, pre-heal proxy: compare mixed vs C-only at matched C cut — "
        "does the FF trim add ~free accuracy cost on top of the latency-relevant C cut?)"
    )


@torch.no_grad()
def _zero_degree(raw, lmax, l):
    """Zero degree-l everywhere via the RMSNorm affine_weight (per-degree scale)."""
    saved = {}
    for n, p in raw.named_parameters():
        if not n.endswith("affine_weight") or p.dim() < 2:
            continue
        saved[n] = p.detach().clone()
        if p.shape[0] == lmax + 1:  # [lmax+1, C] per-degree affine
            p[l].zero_()
        elif p.shape[0] == (lmax + 1) ** 2:  # [L^2, C] per-coefficient
            p[l * l : (l + 1) * (l + 1)].zero_()
    return saved


@torch.no_grad()
def _zero_m(raw, m):
    """Zero m-order m>=1 in every SO2 conv (conv_1 + conv_2): drop that angular order."""
    saved = {}
    key = f"so2_m_conv.{m - 1}.fc.weight"
    for n, p in raw.named_parameters():
        if key in n:
            saved[n] = p.detach().clone()
            p.zero_()
    return saved


@torch.no_grad()
def _restore(raw, saved):
    params = dict(raw.named_parameters())
    for n, v in saved.items():
        params[n].copy_(v)


def _per_struct_de(base, new):
    (e0, _), (e1, _) = base, new
    return [float((a - b).abs().mean()) for a, b in zip(e0, e1)]


def run_sph_axes(raw, structs, base, lmax, mmax, natoms):
    """One-shot sensitivity to dropping a whole degree (l) or angular order (m)."""
    print(f"\n{'angular ablation':<24}{'ΔE (eV)':>12}{'mean|ΔF| (eV/Å)':>18}")
    for l in range(1, lmax + 1):
        saved = _zero_degree(raw, lmax, l)
        de, df = delta(base, predict(raw, structs))
        _restore(raw, saved)
        print(f"{'drop degree l=' + str(l):<24}{de:>12.4f}{df:>18.5f}")
    for m in range(1, mmax + 1):
        saved = _zero_m(raw, m)
        out = predict(raw, structs)
        de, df = delta(base, out)
        de_per_atom = [d / n for d, n in zip(_per_struct_de(base, out), natoms)]
        _restore(raw, saved)
        mean = sum(de_per_atom) / len(de_per_atom)
        std = (sum((x - mean) ** 2 for x in de_per_atom) / len(de_per_atom)) ** 0.5
        print(
            f"{'drop m-order m=' + str(m):<24}{de:>12.4f}{df:>18.5f}"
            f"   ΔE/atom={mean * 1e3:.2f}±{std * 1e3:.2f} meV (rel std {std / mean:.0%})"
        )
    print(
        "\n(m-drops: low mean|ΔF| + LOW ΔE/atom rel-std ⇒ the energy shift is a ~constant "
        "per-atom offset that element-refs / a short re-heal should absorb)"
    )


if __name__ == "__main__":
    main()
