"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

# =============================================================================
# Shared-channel structured pruning for eSCN (eSEN/UMA) backbones.
#
# PAT's per-layer Dim0 pruning zeros DIFFERENT rows per layer, so shared feature
# widths can't be physically shrunk. This module instead prunes the SAME feature
# channels everywhere they appear, so a width can later be compacted (see
# scripts/compact_channels.py) for real dense speedup.
#
# We prune the "sphere" width: the sphere_channels residual stream C, with one
# global mask consistent across all blocks / embeddings / norms / SO2 convs, so C
# can be physically compacted. (An earlier `hidden`-width variant was evaluated but
# dropped -- it is strictly dominated on latency and memory; see the write-up.)
#
# CORE ABSTRACTION -- AxisSpec(axis, width, bases): channel c occupies the flat
# index set {base + c for base in bases} along tensor `axis`. This expresses
# every layout uniformly:
#   * plain axis of size W               -> bases=[0]
#   * SO2 conv1 input (l, [src|tgt], c)  -> bases=[l*2C + r*C], r in {0,1}
#   * SO2 conv2 fc_m0 output (l, c)       -> bases=[l*C]
#   * SO2 conv2 so2_m output (cos/sin, coeff, c) -> bases=[h*out_half + g*C]
#   * radial net.6 output (per-m segments coupled to conv1 inputs) -> concat bases
# A param may carry >1 AxisSpec (e.g. mix_csd: out rows AND in columns).
#
# Equivariance: removing a whole channel (all its l,m components) is SO(3)-safe
# (Wigner rotation treats the channel axis as a batch dim); this module only ever
# removes whole channels.
# =============================================================================
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn
from torchtnt.framework.callback import Callback

from fairchem.core.common import distutils

if TYPE_CHECKING:
    from torchtnt.framework.state import State
    from torchtnt.framework.unit import TTrainUnit


@dataclass
class AxisSpec:
    """How feature channels are laid out along one axis of one parameter."""

    axis: int
    width: int  # number of feature channels (sphere_channels C)
    bases: list[int]  # channel c -> flat indices {b + c for b in bases}

    def flat_indices(self, channels: torch.Tensor) -> torch.Tensor:
        """Flat indices along `axis` for the given channel ids (LongTensor)."""
        b = torch.tensor(self.bases, dtype=torch.long, device=channels.device).view(
            -1, 1
        )  # [B,1]
        return (b + channels.view(1, -1)).reshape(-1)  # [B*len(channels)]

    def channel_scores(self, w: torch.Tensor) -> torch.Tensor:
        """Per-channel L2 (rms-normalized) importance -> [width]."""
        idx = self.flat_indices(torch.arange(self.width, device=w.device))
        sel = w.detach().index_select(self.axis, idx)  # [.., B*width, ..]
        # reshape the selected axis to (B, width) and reduce everything else
        shape = list(sel.shape)
        shape[self.axis : self.axis + 1] = [len(self.bases), self.width]
        sel = sel.reshape(shape)
        dims = [d for d in range(sel.dim()) if d != self.axis + 1]  # keep width axis
        sq = sel.pow(2).sum(dim=dims)  # [width]
        n = sel.numel() / self.width
        return (sq / n).sqrt()

    def keep_index(self, keep_channels: torch.Tensor) -> torch.Tensor:
        """Flat indices along `axis` to KEEP (for compaction index_select)."""
        return self.flat_indices(keep_channels.sort().values)

    def zero_(self, w: torch.Tensor, drop_channels: torch.Tensor) -> None:
        """Zero the dropped channels' slices along `axis` (in place)."""
        if drop_channels.numel() == 0:
            return
        idx = self.flat_indices(drop_channels)
        w.detach().index_fill_(self.axis, idx, 0.0)


def _backbone(model: nn.Module) -> nn.Module:
    return model.backbone if hasattr(model, "backbone") else model


# ------------------------------------------------------------------ specs -----
def sphere_spec(model: nn.Module) -> dict[str, list[AxisSpec]]:
    """param_name -> [AxisSpec] for the sphere_channels (C) residual width."""
    bb = _backbone(model)
    C = bb.sphere_channels
    lmax, mmax = bb.lmax, bb.mmax
    spec: dict[str, list[AxisSpec]] = {}

    def add(name, s):
        spec.setdefault(name, []).append(s)

    # --- clean single-axis params ---
    add("backbone.sphere_embedding.weight", AxisSpec(1, C, [0]))
    for pn, _ in model.named_parameters():
        if "dataset_embedding.dataset_emb_dict." in pn and pn.endswith(".weight"):
            add(pn, AxisSpec(1, C, [0]))
        # RMSNorm affine on every norm_1/norm_2 + final norm
        if pn.endswith(".affine_weight"):  # [lmax+1, C]
            add(pn, AxisSpec(1, C, [0]))
        if pn.endswith(".affine_bias"):  # [C]
            add(pn, AxisSpec(0, C, [0]))
    # mix_csd: out rows [C], in cols = k blocks each [C]
    mix = bb.mix_csd
    add("backbone.mix_csd.weight", AxisSpec(0, C, [0]))
    add("backbone.mix_csd.bias", AxisSpec(0, C, [0]))
    k_in = mix.in_features // C
    add("backbone.mix_csd.weight", AxisSpec(1, C, [b * C for b in range(k_in)]))

    # --- per-block SO2 + atom_wise ---
    for bi in range(bb.num_layers):
        p = f"backbone.blocks.{bi}.edge_wise"
        # conv_1: input carries C as (l, [src|tgt], c) with internal width 2C
        add(
            f"{p}.so2_conv_1.fc_m0.weight",
            AxisSpec(
                1, C, [l * 2 * C + r * C for l in range(lmax + 1) for r in (0, 1)]
            ),
        )
        for m in range(1, mmax + 1):
            g = lmax - m + 1
            add(
                f"{p}.so2_conv_1.so2_m_conv.{m-1}.fc.weight",
                AxisSpec(1, C, [j * 2 * C + r * C for j in range(g) for r in (0, 1)]),
            )
        # conv_1 radial net.6 output: segments coupled to conv_1 fc inputs (width C)
        seg_bases, off = [], 0
        # segment 0 = fc_m0 input layout
        seg_bases += [off + l * 2 * C + r * C for l in range(lmax + 1) for r in (0, 1)]
        off += (lmax + 1) * 2 * C
        for m in range(1, mmax + 1):
            g = lmax - m + 1
            seg_bases += [off + j * 2 * C + r * C for j in range(g) for r in (0, 1)]
            off += g * 2 * C
        add(f"{p}.so2_conv_1.rad_func.net.6.weight", AxisSpec(0, C, seg_bases))
        add(f"{p}.so2_conv_1.rad_func.net.6.bias", AxisSpec(0, C, seg_bases))
        # conv_2: output carries C
        add(
            f"{p}.so2_conv_2.fc_m0.weight",
            AxisSpec(0, C, [l * C for l in range(lmax + 1)]),
        )
        add(
            f"{p}.so2_conv_2.fc_m0.bias",
            AxisSpec(0, C, [l * C for l in range(lmax + 1)]),
        )
        for m in range(1, mmax + 1):
            g = lmax - m + 1
            out_half = g * C
            add(
                f"{p}.so2_conv_2.so2_m_conv.{m-1}.fc.weight",
                AxisSpec(
                    0, C, [h * out_half + j * C for h in (0, 1) for j in range(g)]
                ),
            )
        # atom_wise (grid ff): grid_mlp.0 in [C], grid_mlp.4 out [C]
        aw = f"backbone.blocks.{bi}.atom_wise"
        if hasattr(bb.blocks[bi].atom_wise, "grid_mlp"):
            add(f"{aw}.grid_mlp.0.weight", AxisSpec(1, C, [0]))
            add(f"{aw}.grid_mlp.4.weight", AxisSpec(0, C, [0]))
        else:  # spectral ff
            add(f"{aw}.scalar_mlp.0.weight", AxisSpec(1, C, [0]))
            add(f"{aw}.so3_linear_1.weight", AxisSpec(2, C, [0]))
            add(f"{aw}.so3_linear_2.weight", AxisSpec(1, C, [0]))
            add(f"{aw}.so3_linear_2.bias", AxisSpec(0, C, [0]))

    # edge_degree_embedding radial net.6 output: (m0-coeff, c), m0 coeffs = lmax+1
    add(
        "backbone.edge_degree_embedding.rad_func.net.6.weight",
        AxisSpec(0, C, [k * C for k in range(lmax + 1)]),
    )
    add(
        "backbone.edge_degree_embedding.rad_func.net.6.bias",
        AxisSpec(0, C, [k * C for k in range(lmax + 1)]),
    )

    # heads: their input reads the sphere_channels stream -> slice input axis.
    for pn, pt in model.named_parameters():
        if "output_heads" not in pn:
            continue
        if pn.endswith(("energy_block.0.weight", "energy_block.weight")):
            add(pn, AxisSpec(1, C, [0]))  # Linear(sphere_channels, hidden|1)
        elif pn.endswith(".linear.weight") and pt.dim() == 3:
            add(pn, AxisSpec(2, C, [0]))  # Linear_Force_Head SO3_Linear [l, out, C]

    # final norm handled by the .affine_* loop above.
    # NOTE: charge/spin pos_emb .W are intentionally NOT sliced (sin/cos coupled);
    # their C-wide output is consumed by mix_csd, whose input columns are kept full
    # (mix_csd output rows are sliced instead) -- see compact_channels.py.
    return spec


def get_spec(model: nn.Module, mode: str = "sphere") -> dict[str, list[AxisSpec]]:
    if mode in ("sphere", "sphere_channels"):
        return sphere_spec(model)
    raise ValueError(f"unsupported mode {mode!r} (only 'sphere' is supported)")


# --------------------------------------------------------------- utilities ----
def validate_spec(model: nn.Module, mode: str) -> dict:
    """
    Self-check: every AxisSpec's channel index set must be in-bounds and the
    per-channel index sets must PARTITION exactly the channel-related slots along
    that axis (no overlap, none out of range). Returns a report; raises on error.
    """
    params = dict(model.named_parameters())
    spec = get_spec(model, mode)
    report = {"params": 0, "specs": 0, "width": None}
    for name, axspecs in spec.items():
        assert name in params, f"spec references missing param {name}"
        w = params[name]
        report["params"] += 1
        for s in axspecs:
            report["specs"] += 1
            report["width"] = s.width
            size = w.shape[s.axis]
            idx = s.flat_indices(torch.arange(s.width))
            assert (
                idx.max().item() < size
            ), f"{name} axis{s.axis}: index {idx.max().item()} >= size {size}"
            assert idx.min().item() >= 0
            # no overlap: each covered slot used exactly once
            assert idx.unique().numel() == idx.numel(), f"{name} overlapping indices"
    return report


def channel_importance(model: nn.Module, mode: str) -> torch.Tensor:
    """
    Aggregate per-channel importance across all spec params -> [width].
    """
    params = dict(model.named_parameters())
    spec = get_spec(model, mode)
    width = next(iter(spec.values()))[0].width
    device = next(iter(params.values())).device
    agg = torch.zeros(width, device=device)
    for name, axspecs in spec.items():
        for s in axspecs:
            agg = agg + s.channel_scores(params[name])
    return agg


@torch.no_grad()
def apply_channel_mask(model: nn.Module, mode: str, drop_channels: torch.Tensor) -> int:
    """Zero the given channels across all spec params (in place). Returns count of zeroed slots."""
    params = dict(model.named_parameters())
    spec = get_spec(model, mode)
    for name, axspecs in spec.items():
        for s in axspecs:
            s.zero_(params[name], drop_channels)
    return int(drop_channels.numel())


def _unwrap(model: nn.Module) -> nn.Module:
    """Peel DDP/FSDP/EMA `.module` wrappers until the HydraModel (has .backbone)."""
    while not hasattr(model, "backbone") and hasattr(model, "module"):
        model = model.module
    return model


def cubic_target(step: int, warmup: int, healing_start: int, final: float) -> float:
    """
    Cubic sparsity ramp 0 -> ``final`` over (warmup, healing_start); flat at
    ``final`` afterwards (Zhu & Gupta). Returns the target channel sparsity.
    """
    if step <= warmup:
        return 0.0
    if step >= healing_start:
        return final
    frac = (step - warmup) / (healing_start - warmup)
    return final * (1.0 - (1.0 - frac) ** 3)


class ChannelPruningCallback(Callback):
    """
    Shared-channel structured pruning during training, so a feature width can
    later be physically compacted (``scripts/compact_channels.py``).

    Schedule mirrors PAT's cubic min-sparsity ramp:
      * step < warmup            : no pruning (dense warmup).
      * warmup <= step < heal    : ramp target 0 -> ``target_sparsity``; each step
        drop the globally least-important channels to hit the current target.
      * step >= heal             : FREEZE the dropped set; re-zero it every step so
        survivors heal while dropped channels stay exactly zero (mirrors the
        ``p.ne(0)`` heal-freeze in the PAT optimizer).

    Fractions resolve against total training steps in ``on_train_start`` (same
    total-step derivation as ``MLIPTrainEvalUnit.load_scheduler``); absolute
    ``warmup_steps`` / ``healing_start_step`` override the fractions.
    """

    def __init__(
        self,
        mode: str,
        target_sparsity: float,
        warmup_frac: float | None = 0.05,
        healing_start_frac: float | None = 0.6,
        warmup_steps: int | None = None,
        healing_start_step: int | None = None,
        log_every_n_steps: int = 100,
    ) -> None:
        if mode not in ("sphere", "sphere_channels"):
            raise ValueError(f"unsupported mode {mode!r} (only 'sphere' is supported)")
        if not 0.0 <= target_sparsity < 1.0:
            raise ValueError("target_sparsity must be in [0, 1)")
        self.mode = mode
        self.target_sparsity = target_sparsity
        self._warmup_frac = warmup_frac
        self._healing_start_frac = healing_start_frac
        self.warmup_steps = warmup_steps
        self.healing_start_step = healing_start_step
        self.log_every_n_steps = log_every_n_steps
        self.drop_channels: torch.Tensor | None = None  # frozen at heal start
        self._resolved = False

    # -- schedule resolution -------------------------------------------------
    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        if self._resolved:
            return
        total_steps = self._total_steps(state, unit)
        if self.warmup_steps is None:
            self.warmup_steps = int((self._warmup_frac or 0.0) * total_steps)
        if self.healing_start_step is None:
            self.healing_start_step = int(
                (self._healing_start_frac or 0.0) * total_steps
            )
        assert self.warmup_steps < self.healing_start_step, (
            f"channel_prune: warmup_steps={self.warmup_steps} must be < "
            f"healing_start_step={self.healing_start_step} (total={total_steps})"
        )
        self._resolved = True
        logging.info(
            "ChannelPruningCallback[%s]: total_steps=%d warmup=%d healing_start=%d "
            "target_sparsity=%.3f",
            self.mode,
            total_steps,
            self.warmup_steps,
            self.healing_start_step,
            self.target_sparsity,
        )

    @staticmethod
    def _total_steps(state: State, unit: TTrainUnit) -> int:
        sched_kw = getattr(
            getattr(unit, "cosine_lr_scheduler_fn", None), "keywords", {}
        )
        steps = sched_kw.get("steps")
        if steps is not None:
            return int(steps)
        epochs = sched_kw.get("epochs")
        dl = state.train_state.dataloader
        dl_size = len(dl)
        return int(epochs * dl_size)

    # -- pruning step --------------------------------------------------------
    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        step = unit.train_progress.num_steps_completed
        if step <= self.warmup_steps:
            return
        model = _unwrap(unit.model)
        width = self._width(model)

        if step < self.healing_start_step:
            # prune phase: recompute the least-important channels for this target
            target = cubic_target(
                step, self.warmup_steps, self.healing_start_step, self.target_sparsity
            )
            n_drop = int(round(target * width))
            if n_drop == 0:
                return
            imp = channel_importance(model, self.mode)
            self.drop_channels = torch.argsort(imp)[:n_drop].to(imp.device)
        elif self.drop_channels is None:
            # heal phase entered fresh (e.g. resumed checkpoint): reconstruct the
            # frozen set from the already-zeroed channels.
            n_drop = int(round(self.target_sparsity * width))
            imp = channel_importance(model, self.mode)
            self.drop_channels = torch.argsort(imp)[:n_drop].to(imp.device)

        # zero the dropped channels on the live model (and EMA, so eval + later
        # compaction see exact zeros).
        apply_channel_mask(model, self.mode, self.drop_channels)
        ema = getattr(unit, "ema_model", None)
        if ema is not None:
            apply_channel_mask(_unwrap(ema), self.mode, self.drop_channels)

        if step % self.log_every_n_steps == 0:
            sparsity = self.drop_channels.numel() / width
            if getattr(unit, "logger", None) is not None:
                unit.logger.log(
                    {"train/channel_sparsity": sparsity}, step=step, commit=False
                )
            if distutils.is_master():
                logging.info(
                    "channel_prune[%s] step=%d dropped=%d/%d sparsity=%.3f",
                    self.mode,
                    step,
                    self.drop_channels.numel(),
                    width,
                    sparsity,
                )

    def _width(self, model: nn.Module) -> int:
        return _backbone(model).sphere_channels
