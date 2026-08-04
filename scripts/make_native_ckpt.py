"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

# =============================================================================
# Write an UNTRAINED native-shape inference checkpoint at a target sphere/hidden/
# edge width, for LATENCY + MEMORY profiling only (weights are random; accuracy is
# irrelevant and measured separately via real training). This mirrors the
# collaborator's eSEN profile method of instantiating native target shapes.
#
#   python scripts/make_native_ckpt.py <dense_inference.pt> <out.pt> --sphere 96
# =============================================================================
import argparse
import copy

import torch
from omegaconf import OmegaConf

from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint
from fairchem.core.units.mlip_unit.utils import load_inference_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dense")
    ap.add_argument("out")
    ap.add_argument("--sphere", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--edge", type=int, default=None)
    ap.add_argument("--mmax", type=int, default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.dense, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(copy.deepcopy(ckpt.model_config))
    if args.sphere is not None:
        OmegaConf.update(cfg, "backbone.sphere_channels", args.sphere, force_add=True)
        # Route-B style: norm stats over the kept width (near-exact for real runs).
        OmegaConf.update(
            cfg, "backbone.norm_stats_num_channels", args.sphere, force_add=True
        )
    if args.hidden is not None:
        OmegaConf.update(cfg, "backbone.hidden_channels", args.hidden, force_add=True)
    if args.edge is not None:
        OmegaConf.update(cfg, "backbone.edge_channels", args.edge, force_add=True)
    if args.mmax is not None:
        OmegaConf.update(cfg, "backbone.mmax", args.mmax, force_add=True)

    import hydra

    model = hydra.utils.instantiate(cfg)
    model.eval()
    nparam = sum(p.numel() for p in model.parameters())
    print(
        f"native shape S={args.sphere} H={args.hidden} E={args.edge} "
        f"mmax={args.mmax} params={nparam}"
    )

    ema = torch.optim.swa_utils.AveragedModel(model)
    out_ckpt = MLIPInferenceCheckpoint(
        model_state_dict=model.state_dict(),
        ema_state_dict=ema.state_dict(),
        model_config=OmegaConf.to_container(cfg, resolve=True),
        tasks_config=ckpt.tasks_config,
    )
    torch.save(out_ckpt, args.out)
    reloaded, _ = load_inference_model(args.out, use_ema=True, strict=True)
    print(
        "wrote", args.out, "sphere_channels=", reloaded.module.backbone.sphere_channels
    )


if __name__ == "__main__":
    main()
