"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

# =============================================================================
# Compact a sphere-channel-pruned eSCN model: physically remove the pruned
# sphere_channels and rebuild at reduced width C -> K for real dense speedup.
#
# Strategy: re-instantiate the model from its (Hydra) config with sphere_channels
# reduced to K, then copy each parameter, index_select-ing the KEPT channels along
# every AxisSpec axis. Non-spec params are width-independent and copied as-is.
#
# The csd front-end (charge/spin/dataset embeddings + mix_csd INPUT) needs a small
# post-instantiation surgery: for rand_emb models it is sliced to K (uniform-K,
# round-trippable); for pos_emb it is kept full-width to avoid the sin/cos coupling
# (mix_csd OUTPUT rows + sphere_embedding + everything downstream shrink).
# =============================================================================
import copy

import torch
from omegaconf import OmegaConf

from fairchem.core.models.uma.channel_pruning import (
    channel_importance,
    get_spec,
)


def infer_keep_channels(model, mode: str, tol: float = 1e-12) -> torch.Tensor:
    """Channels whose aggregate importance is > tol (i.e. not fully zeroed)."""
    imp = channel_importance(model, mode)
    return (imp > tol).nonzero(as_tuple=True)[0]


def _reduced_model(
    model_cfg, keep: torch.Tensor, norm_stats=None, centering_stats=None
):
    """Instantiate a fresh model with sphere_channels reduced to len(keep)."""
    import hydra

    K = int(keep.numel())
    cfg = copy.deepcopy(model_cfg)
    OmegaConf.update(cfg, "backbone.sphere_channels", K, force_add=True)
    # keep the RMSNorm over-channel statistics matched to the original width
    # (so centering + normalization reproduce the pre-compaction shift/scale).
    if norm_stats is not None:
        OmegaConf.update(
            cfg, "backbone.norm_stats_num_channels", int(norm_stats), force_add=True
        )
    # the RMSNorm normalization must still account for the centering energy of the
    # channels we physically removed -> tell the reduced norms the ORIGINAL width so
    # they add that contribution back exactly (see layer_norm.py).
    if centering_stats is not None:
        OmegaConf.update(
            cfg,
            "backbone.norm_stats_centering_channels",
            int(centering_stats),
            force_add=True,
        )
    return hydra.utils.instantiate(cfg)


@torch.no_grad()
def compact(
    model,
    model_cfg,
    mode: str = "sphere",
    keep: torch.Tensor | None = None,
    frontend_mode: str = "auto",
):
    """
    Return (compact_model, report). `model` is the pruned (zeroed-channel) model;
    `model_cfg` is its Hydra model config node (for rebuild at reduced width).

    frontend_mode (sphere only): "auto" -> slice csd front-end for rand_emb models
    (uniform-K, round-trippable), keep full width for pos_emb; "slice"/"restore"
    force one path (for diagnostics).
    """
    if keep is None:
        keep = infer_keep_channels(model, mode)
    keep, _ = keep.sort()
    spec = get_spec(model, mode)
    src = dict(model.named_parameters())

    # (sphere) if the RMSNorm supports the over-channel stats override, build the
    # compacted norms with the SAME divisor as the pruned model (= its original
    # sphere_channels for a standard run, or the trained target for a Route-B run).
    # This reproduces the norm centering + normalization EXACTLY, so no approximate
    # sqrt(C/K) affine rescale is needed. `norm_stats` None => old norm => fall back.
    norm_stats = None
    centering_stats = None
    if mode in ("sphere", "sphere_channels"):
        bb = model.backbone if hasattr(model, "backbone") else model
        norm_stats = getattr(bb.norm, "stats_num_channels", None)
        # original physical width -> the reduced norms add back the dropped
        # channels' centering energy exactly (makes compaction output-preserving).
        # Only meaningful for a norm that supports the over-channel stats override;
        # an ancient norm without it falls back to the approximate sqrt(C/K) rescale.
        if norm_stats is not None:
            centering_stats = bb.sphere_channels

    new = _reduced_model(
        model_cfg, keep, norm_stats=norm_stats, centering_stats=centering_stats
    )
    dst = dict(new.named_parameters())

    # sphere: how the csd front-end (charge/spin/dataset embeddings + mix_csd) is
    # handled depends on the charge/spin embedding type:
    #   rand_emb  -> plain lookup table, channel axis is cleanly sliceable, and the
    #                mix_csd INPUT columns for pruned channels were already zeroed in
    #                training (they are in sphere_spec), so those csd channels feed
    #                nothing. We slice the whole front-end -> a UNIFORM sphere_channels=K
    #                model that round-trips as a standard checkpoint.
    #   pos_emb   -> sin/cos coupled (W[i] -> out i and i+C/2); not channel-sliceable,
    #                so keep the front-end at full width (mixed-width model).
    frontend = set()
    rand_frontend = False
    if mode in ("sphere", "sphere_channels"):
        is_rand = any("charge_embedding" in pn and "rand_emb" in pn for pn in src)
        if frontend_mode == "slice":
            rand_frontend = True
        elif frontend_mode == "restore":
            rand_frontend = False
        else:  # auto
            rand_frontend = is_rand
        if rand_frontend:
            # slice csd embeddings manually below; mix_csd / sphere_embedding /
            # dataset_embedding are in sphere_spec -> sliced by the main loop.
            frontend = {
                pn for pn in src if "charge_embedding" in pn or "spin_embedding" in pn
            }
        else:
            frontend = {
                pn
                for pn in src
                if any(
                    t in pn
                    for t in (
                        "mix_csd",
                        "sphere_embedding",
                        "charge_embedding",
                        "spin_embedding",
                        "dataset_embedding",
                    )
                )
            }

    copied_sliced = 0
    for name, dp in dst.items():
        sp = src.get(name)
        if sp is None or name in frontend:
            continue
        if name in spec:
            w = sp.clone()
            for axspec in spec[name]:
                w = w.index_select(axspec.axis, axspec.keep_index(keep))
            assert (
                w.shape == dp.shape
            ), f"{name}: sliced {tuple(w.shape)} != rebuilt {tuple(dp.shape)}"
            dp.copy_(w)
            copied_sliced += 1
        elif sp.shape == dp.shape:
            dp.copy_(sp)
        else:
            # a width-dependent param we failed to spec -> loud error, not silent random init
            raise AssertionError(
                f"unhandled width-dependent param (add to spec): {name} "
                f"{tuple(sp.shape)} -> {tuple(dp.shape)}"
            )

    if mode in ("sphere", "sphere_channels"):
        if rand_frontend:
            _slice_sphere_frontend(model, new, keep)  # uniform-K, round-trippable
        else:
            _restore_sphere_frontend(model, new, keep)  # full-width (pos_emb coupling)
        if norm_stats is None:
            # Fallback for an OLD norm without the over-channel stats override:
            # the RMSNorm normalizes over sphere_channels, so dropping (zeroed)
            # channels shrinks the denominator C->K, rescaling survivors by
            # sqrt(C/K). Undo the multiplicative part via affine_weight. NOTE this
            # does NOT correct the centering shift (an additive mean-over-channels)
            # -> ~0.08 eV/A residual; the stats override (above) fixes it exactly.
            C = model.backbone.sphere_channels
            scale = (C / int(keep.numel())) ** 0.5
            for n, p in new.named_parameters():
                if n.endswith(".affine_weight"):
                    p.mul_(scale)

    report = {
        "mode": mode,
        "kept": int(keep.numel()),
        "copied_sliced": copied_sliced,
        "params_before": sum(p.numel() for p in model.parameters()),
        "params_after": sum(p.numel() for p in new.parameters()),
    }
    report["param_reduction"] = 1 - report["params_after"] / report["params_before"]
    return new, report


@torch.no_grad()
def _slice_sphere_frontend(old, new, keep: torch.Tensor):
    """
    rand_emb csd front-end: slice the charge/spin embedding tables to the kept
    channels (axis 1), giving a UNIFORM sphere_channels=K model. mix_csd,
    sphere_embedding and dataset_embedding are sliced by the main spec loop.

    Correctness: mix_csd's input columns for pruned channels are zeroed during
    training (they are in sphere_spec), so pruned csd channels feed nothing —
    dropping them is output-preserving.
    """
    obb, nbb = old.backbone, new.backbone
    for attr in ("charge_embedding", "spin_embedding"):
        if hasattr(obb, attr):
            oe, ne = getattr(obb, attr), getattr(nbb, attr)
            ne.rand_emb.weight.copy_(oe.rand_emb.weight.data.index_select(1, keep))


@torch.no_grad()
def _restore_sphere_frontend(old, new, keep: torch.Tensor):
    """
    Keep the csd front-end at full width (avoids pos_emb sin/cos coupling):
    charge/spin/dataset embeddings full C, mix_csd = [K, 3C] (output rows sliced,
    input cols full). sphere_embedding is sliced to K (writes the K-wide l=0 slot).
    """
    obb, nbb = old.backbone, new.backbone
    # restore full-width csd embeddings by deep-copying the originals
    for attr in ("charge_embedding", "spin_embedding", "dataset_embedding"):
        if hasattr(obb, attr):
            setattr(nbb, attr, copy.deepcopy(getattr(obb, attr)))
    # rebuild mix_csd as [K, in_full] with sliced output rows + full input cols
    om = obb.mix_csd
    nm = torch.nn.Linear(om.in_features, int(keep.numel()), bias=om.bias is not None)
    nm.weight.copy_(om.weight.data.index_select(0, keep))
    if om.bias is not None:
        nm.bias.copy_(om.bias.data.index_select(0, keep))
    nbb.mix_csd = nm
    # sphere_embedding: [num, K] sliced on dim1 (already rebuilt at K, copy sliced)
    nbb.sphere_embedding.weight.copy_(obb.sphere_embedding.weight.data[:, keep])


@torch.no_grad()
def validate_compaction(orig, compact_m, sample_inputs) -> dict:
    """Output delta between original (pruned) and compacted model on the same input."""
    orig.eval()
    compact_m.eval()
    o = orig(*sample_inputs)
    c = compact_m(*sample_inputs)
    o = o if isinstance(o, torch.Tensor) else next(iter(o.values()))
    c = c if isinstance(c, torch.Tensor) else next(iter(c.values()))
    d = (o - c).abs()
    return {"max_abs_delta": float(d.max()), "mean_abs_delta": float(d.mean())}
