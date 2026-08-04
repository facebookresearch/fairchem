"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

# =============================================================================
# Prototype: per-m-order ("adaptive mmax") grouper + its compaction.
#
# (1) m-order importance: per block, aggregate L2 of the so2_m_conv[m] weights
#     (conv_1 + conv_2) — which angular order is least used.
# (2) Compaction of the highest m-order: the m=2 SO2 blocks are the ONLY tensors
#     that differ between an mmax=2 and an mmax=1 model, so "dropping m=2" compacts
#     to a standard `mmax=1` model (a config knob) — no custom kernels. We rebuild
#     at mmax-1, transfer the retained weights (same-shape copy; leading-slice the
#     radial net.6 whose per-m segments shrink), and PARITY-CHECK against the model
#     with m=2 masked to zero. Exact parity ⇒ the compaction path is correct.
#
#   python scripts/per_m_compact_probe.py dense_inference.pt
# =============================================================================
import argparse
import copy

import torch
from ase.build import molecule as get_molecule
from omegaconf import OmegaConf

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit.api.inference import (
    MLIPInferenceCheckpoint,  # noqa: F401
)
from fairchem.core.units.mlip_unit.utils import load_inference_model


def calib():
    out = []
    for n in ["CH3CH2OH", "C6H6", "CH3COOH", "CH3CN"]:
        a = get_molecule(n)
        a.info["charge"], a.info["spin"] = 0, 1
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
    es, fs = [], []
    for d in structs:
        o = model(d)
        es.append(o["omol_energy"]["energy"].detach().reshape(-1))
        fs.append(o["omol_forces"]["forces"].detach())
    return es, fs


def delta(base, new):
    (e0, f0), (e1, f1) = base, new
    de = torch.stack([(a - b).abs().max() for a, b in zip(e0, e1)]).max()
    df = torch.stack([(a - b).abs().max() for a, b in zip(f0, f1)]).max()
    return float(de), float(df)


def m_importance(raw, mmax, nlayers):
    """Per-block per-m aggregate L2 of so2_m_conv weights (conv_1 + conv_2)."""
    params = dict(raw.named_parameters())
    print(f"{'block':<8}" + "".join(f"{'m=' + str(m):>10}" for m in range(1, mmax + 1)))
    for i in range(nlayers):
        row = []
        for m in range(1, mmax + 1):
            s = 0.0
            for conv in ("so2_conv_1", "so2_conv_2"):
                k = f"backbone.blocks.{i}.edge_wise.{conv}.so2_m_conv.{m - 1}.fc.weight"
                if k in params:
                    s += float(params[k].detach().pow(2).sum())
            row.append(s**0.5)
        print(f"blk{i:<5}" + "".join(f"{v:>10.3f}" for v in row))


@torch.no_grad()
def mask_top_m(raw, mmax):
    """Zero the highest m-order (m=mmax) everywhere -> reference for parity."""
    for n, p in raw.named_parameters():
        if f"so2_m_conv.{mmax - 1}.fc.weight" in n:
            p.zero_()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, _ = load_inference_model(args.ckpt, use_ema=True, strict=True)
    raw = model.module
    raw.eval()
    bb = raw.backbone
    mmax, nlayers = bb.mmax, bb.num_layers
    structs = calib()

    print(f"=== m-order importance (mmax={mmax}) ===")
    m_importance(raw, mmax, nlayers)

    # reference: highest m-order masked to zero on the full (mmax) model
    ref = predict(raw, structs)  # baseline first
    mask_top_m(raw, mmax)
    ref_masked = predict(raw, structs)
    print(
        f"\ndrop m={mmax} (mask) vs full: {delta(ref, ref_masked)} (maxΔE eV, maxΔF eV/Å)"
    )

    # === compaction: rebuild at mmax-1 and transfer retained weights ===
    cfg = OmegaConf.create(copy.deepcopy(ckpt.model_config))
    OmegaConf.update(cfg, "backbone.mmax", mmax - 1, force_add=True)
    import hydra

    compact = hydra.utils.instantiate(cfg)
    compact.eval()
    src = dict(raw.named_parameters())  # raw currently has m=mmax zeroed (harmless)
    # reload clean source weights (undo the mask) for an honest transfer
    src_clean, _ = load_inference_model(args.ckpt, use_ema=True, strict=True)
    src = dict(src_clean.module.named_parameters())

    copied, sliced, missing = 0, 0, []
    with torch.no_grad():
        for n, tgt in compact.named_parameters():
            if n not in src:
                missing.append(n)
                continue
            s = src[n]
            if s.shape == tgt.shape:
                tgt.copy_(s)
                copied += 1
            else:
                # per-m radial segments are trailing -> leading-slice to target shape
                idx = tuple(slice(0, d) for d in tgt.shape)
                if all(t <= so for t, so in zip(tgt.shape, s.shape)):
                    tgt.copy_(s[idx])
                    sliced += 1
                else:
                    missing.append(f"{n} (shape {tuple(s.shape)}->{tuple(tgt.shape)})")
    print(
        f"\ntransfer: copied={copied} leading-sliced={sliced} unresolved={len(missing)}"
    )
    for m in missing[:12]:
        print("  UNRESOLVED:", m)

    comp_out = predict(compact, structs)
    de, df = delta(ref_masked, comp_out)
    p_before = sum(p.numel() for p in raw.parameters())
    p_after = sum(p.numel() for p in compact.parameters())
    print(
        f"\nPARITY compact(mmax={mmax - 1}) vs m={mmax}-masked: "
        f"maxΔE={de:.2e} eV  maxΔF={df:.2e} eV/Å"
    )
    print(f"params {p_before} -> {p_after} ({1 - p_after / p_before:.1%})")
    print("parity ~0 ⇒ dropping the top m-order compacts cleanly to an mmax-1 model.")


if __name__ == "__main__":
    main()
