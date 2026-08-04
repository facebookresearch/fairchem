"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

# =============================================================================
# Compact a channel-pruned inference checkpoint (zeroed shared channels physically
# removed) and save a NEW inference .pt that loads through the standard
# MLIPPredictUnit / FAIRChemCalculator path at the reduced width.
#
#   python scripts/compact_save_ckpt.py <pruned_inference.pt> <out.pt>
# =============================================================================
import argparse
import copy
import importlib.util
from pathlib import Path

import torch
from omegaconf import OmegaConf

from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint
from fairchem.core.units.mlip_unit.utils import load_inference_model

_CC = Path(__file__).parent / "compact_channels.py"
_spec = importlib.util.spec_from_file_location("compact_channels", _CC)
compact_channels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compact_channels)
infer_keep_channels = compact_channels.infer_keep_channels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pruned")
    ap.add_argument("out")
    args = ap.parse_args()

    # load the EMA model (what inference actually uses) as a HydraModel
    ckpt = torch.load(args.pruned, map_location="cpu", weights_only=False)
    ema_model, _ = load_inference_model(args.pruned, use_ema=True, strict=True)
    raw = ema_model.module  # AveragedModel -> HydraModel

    keep = infer_keep_channels(raw, "sphere")
    full = raw.backbone.sphere_channels
    print(
        f"sphere: kept {keep.numel()}/{full} channels (dropped {full - keep.numel()})"
    )

    # model_config is a plain dict in the checkpoint; compact() needs a DictConfig
    model_cfg = OmegaConf.create(copy.deepcopy(ckpt.model_config))
    compacted, report = compact_channels.compact(raw, model_cfg, keep=keep)
    compacted.eval()
    print("compaction report:", report)

    reduced_cfg = copy.deepcopy(model_cfg)
    OmegaConf.update(
        reduced_cfg, "backbone.sphere_channels", int(keep.numel()), force_add=True
    )
    # persist the RMSNorm over-channel stats divisor so the reloaded model
    # reproduces the original width's centering/normalization (Route A) = the pruned
    # model's stats (its original sphere_channels, or a Route-B target).
    ns = getattr(raw.backbone, "norm_stats_num_channels", None)
    if ns is None:
        ns = getattr(raw.backbone.norm, "stats_num_channels", None)
    if ns is not None:
        OmegaConf.update(
            reduced_cfg, "backbone.norm_stats_num_channels", int(ns), force_add=True
        )
    reduced_cfg = OmegaConf.to_container(
        reduced_cfg, resolve=True
    )  # save as plain dict

    ema_wrapped = torch.optim.swa_utils.AveragedModel(compacted)
    out_ckpt = MLIPInferenceCheckpoint(
        model_state_dict=compacted.state_dict(),
        ema_state_dict=ema_wrapped.state_dict(),
        model_config=reduced_cfg,
        tasks_config=ckpt.tasks_config,
    )
    torch.save(out_ckpt, args.out)
    print("wrote", args.out)

    # round-trip check: reload through the standard path
    reloaded, _ = load_inference_model(args.out, use_ema=True, strict=True)
    print(
        "reloaded OK; backbone sphere_channels =",
        reloaded.module.backbone.sphere_channels,
    )


if __name__ == "__main__":
    main()
