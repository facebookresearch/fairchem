"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import torch
from torch.optim import AdamW
from torchtnt.framework.callback import Callback

from fairchem.core.common import distutils
from fairchem.core.common.logger import WandBSingletonLogger
from fairchem.core.components.train.train_runner import TrainEvalRunner
from fairchem.core.units.mlip_unit.mlip_unit import (
    MLIPTrainEvalUnit,
    Task,
    TrainStrategy,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torchtnt.framework.state import State

# torchao's PARQ lives in a prototype namespace. It is an optional dependency:
# install with `pip install fairchem-core[quant]` (or a torch-compatible torchao
# build that ships `torchao.prototype.parq`). We import lazily-friendly names at
# module load and raise a clear error if the package is missing.
try:
    from torchao.prototype.parq.optim import ProxPARQ, QuantOptimizer
    from torchao.prototype.parq.quant import UnifTorchaoQuantizer

    _HAS_PARQ = True
except ImportError as exc:  # pragma: no cover - exercised only when dep missing
    _HAS_PARQ = False
    _PARQ_IMPORT_ERROR = exc


def _require_parq() -> None:
    """Raise a helpful error if torchao's PARQ prototype is unavailable."""
    if not _HAS_PARQ:
        raise ImportError(
            "PARQ quantization requires torchao with the `torchao.prototype.parq` "
            "module. Install the quant extra: `pip install fairchem-core[quant]`, "
            "or install a torch-compatible torchao build that ships prototype.parq."
        ) from _PARQ_IMPORT_ERROR


# ---------------------------------------------------------------------------
# Quantizable-layer selection for the UMA / eSCN-MD backbone.
#
# Weight-only quantization applies an elementwise scale-and-round. With a single
# per-tensor scale this commutes with every group operation in the network, so it
# is equivariance-safe everywhere. Finer (per-row / per-channel) granularities are
# NOT safe on the SO(2) `fc` weights: `SO2_m_Conv.forward` reshapes the linear
# output into [R0, I0, R1, I1] and recombines it as (R0 - I1, R1 + I0); assigning
# different scales across those slots breaks rotational equivariance. See
# `fairchem/core/models/uma/nn/so2_layers.py`. We therefore quantize per-tensor.
#
# The pool below covers the linear/spectral weights of the backbone (SO(2) fc /
# fc_m0, SO(3) so3_linear, the radial MLP projections, and the scalar MLPs). The
# atom embeddings, all norm affine parameters, and the entire energy/force head
# are intentionally excluded: forces are computed as -dE/dpos via autograd, so
# quantization noise in the head couples straight into the force error.
# ---------------------------------------------------------------------------

QUANTIZABLE_PATTERNS: list[re.Pattern] = [
    # SO(2) edge-wise linears (m=0 branch, m>0 branch, and the radial MLP).
    re.compile(r"(?:^|\.)blocks\.\d+\.edge_wise\.so2_conv_[12]\.fc_m0\.weight$"),
    re.compile(
        r"(?:^|\.)blocks\.\d+\.edge_wise\.so2_conv_[12]\.so2_m_conv\.\d+\.fc\.weight$"
    ),
    re.compile(
        r"(?:^|\.)blocks\.\d+\.edge_wise\.so2_conv_[12]\.rad_func\.net\.(?:0|3|6)\.weight$"
    ),
    # SO(3) atom-wise linears.
    re.compile(r"(?:^|\.)blocks\.\d+\.atom_wise\.scalar_mlp\.0\.weight$"),
    re.compile(r"(?:^|\.)blocks\.\d+\.atom_wise\.so3_linear_[12]\.weight$"),
    # Edge-degree embedding radial MLP.
    re.compile(r"(?:^|\.)edge_degree_embedding\.rad_func\.net\.(?:0|3|6)\.weight$"),
    # Charge/spin/dataset mixing linear.
    re.compile(r"(?:^|\.)mix_csd\.weight$"),
    # NOTE: energy_block.* (the head) is intentionally NOT quantized.
]


def is_quantizable(name: str) -> bool:
    """Return True if the parameter ``name`` is in the equivariance-safe pool."""
    return any(rx.search(name) for rx in QUANTIZABLE_PATTERNS)


def _is_bias_norm_or_embedding(name: str) -> bool:
    """Return True for params that should never receive weight decay and are not
    quantized: biases, norm affine params, embeddings, and the LayerNorm weights
    inside the radial MLPs (positions net.1 / net.4)."""
    return (
        name.endswith(".bias")
        or ".affine_weight" in name  # covers norm.affine_weight
        or ".affine_bias" in name  # covers norm.affine_bias
        or "embedding" in name.lower()
        or re.search(r"\.rad_func\.net\.(?:1|4)\.weight$", name) is not None
    )


def build_parq_param_groups(
    model: torch.nn.Module,
    quant_bits: int,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Split model parameters into optimizer groups for PARQ.

    Three groups are produced:
        - quant: the equivariance-safe quantizable weights, tagged with
          ``quant_bits`` so ``QuantOptimizer`` regularizes them.
        - no_decay: biases, norms, and embeddings (weight decay disabled,
          not quantized).
        - decay: any remaining trainable weights (weight decay applied, not
          quantized) — notably the energy/force head, kept in full precision.

    Args:
        model: the model to partition.
        quant_bits: bit-width for the quantizable group.
        weight_decay: weight decay for the quant and decay groups.

    Returns:
        A list of parameter-group dicts suitable for a torch optimizer.
    """
    no_weight_decay: set[str] = set()
    for module in model.modules():
        if hasattr(module, "no_weight_decay") and callable(module.no_weight_decay):
            try:
                skip = module.no_weight_decay()
            except (AttributeError, NotImplementedError):
                skip = set()
            if isinstance(skip, (set, list, tuple)):
                no_weight_decay.update(skip)

    quant_params, quant_names = [], []
    no_decay_params, no_decay_names = [], []
    decay_params, decay_names = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if is_quantizable(name):
            quant_params.append(p)
            quant_names.append(name)
        elif any(name.endswith(sfx) for sfx in no_weight_decay) or (
            _is_bias_norm_or_embedding(name)
        ):
            no_decay_params.append(p)
            no_decay_names.append(name)
        else:
            decay_params.append(p)
            decay_names.append(name)

    if distutils.is_master():
        n_quant = sum(p.numel() for p in quant_params)
        n_no_decay = sum(p.numel() for p in no_decay_params)
        n_decay = sum(p.numel() for p in decay_params)
        total = n_quant + n_no_decay + n_decay
        logging.info(
            f"[PARQ] W{quant_bits} param split — "
            f"quant={n_quant:,} ({100 * n_quant / total:.2f}%), "
            f"no_decay={n_no_decay:,} ({100 * n_no_decay / total:.2f}%), "
            f"decay={n_decay:,} ({100 * n_decay / total:.2f}%); "
            f"{len(quant_names)} quantizable tensors"
        )

    groups: list[dict[str, Any]] = []
    if quant_params:
        # No `quant_block_size`: per-tensor scale (equivariance-safe on SO(2) fc).
        groups.append(
            {
                "params": quant_params,
                "weight_decay": weight_decay,
                "quant_bits": int(quant_bits),
            }
        )
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    return groups


def build_parq_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    quant_bits: int,
    total_steps: int,
    anneal_start_frac: float = 0.10,
    anneal_end_frac: float = 0.90,
    anneal_steepness: float = 10.0,
    warmup_steps: int = 0,
    quant_period: int = 10,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> QuantOptimizer:
    """Wrap AdamW in PARQ's ``QuantOptimizer`` for weight-only QAT.

    PARQ regularizes the quantizable group toward a discrete grid, annealing its
    soft-to-hard inverse-slope schedule from ``anneal_start_frac`` to
    ``anneal_end_frac`` of ``total_steps``. Quantization is per-tensor.

    Args:
        model: the model to optimize.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay (quant + decay groups).
        quant_bits: bit-width for the quantizable weights (e.g. 8, 6, 4).
        total_steps: total number of training steps; sets the anneal horizon.
        anneal_start_frac: fraction of ``total_steps`` where soft->hard begins.
        anneal_end_frac: fraction of ``total_steps`` where hard quantization is reached.
        anneal_steepness: sigmoid steepness of the inverse-slope schedule.
        warmup_steps: pure base-optimizer steps before regularization starts.
        quant_period: how often (in steps) quantization targets are recomputed.
        betas: AdamW betas.
        eps: AdamW epsilon.

    Returns:
        A ``QuantOptimizer`` wrapping AdamW.
    """
    _require_parq()
    param_groups = build_parq_param_groups(model, quant_bits, weight_decay)
    base = AdamW(param_groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    # A fresh quantizer per optimizer: UnifTorchaoQuantizer caches its
    # quant_min/max on first use, so a shared instance would leak one
    # bit-width's bounds into another.
    quantizer = UnifTorchaoQuantizer()

    anneal_start = max(warmup_steps + 1, int(total_steps * anneal_start_frac))
    anneal_end = min(total_steps - 1, int(total_steps * anneal_end_frac))
    prox_map = ProxPARQ(
        anneal_start=anneal_start,
        anneal_end=anneal_end,
        steepness=anneal_steepness,
    )
    if distutils.is_master():
        logging.info(
            f"[PARQ] W{quant_bits} schedule: total_steps={total_steps}, "
            f"warmup={warmup_steps}, anneal_start={anneal_start}, "
            f"anneal_end={anneal_end}, steepness={anneal_steepness}, "
            f"quant_period={quant_period}"
        )

    return QuantOptimizer(
        base,
        quantizer,
        prox_map,
        warmup_steps=warmup_steps,
        quant_period=quant_period,
        quant_per_channel=False,  # per-tensor: equivariance-safe on SO(2) fc
        quant_shrink=False,
    )


@torch.no_grad()
def quantize_model_weights_in_place(
    model: torch.nn.Module, bits: int
) -> tuple[int, int]:
    """Hard-quantize every quantizable weight of ``model`` in place (per-tensor).

    This is the post-training-quantization primitive: it snaps the equivariance-safe
    weights to a ``bits``-bit grid without any fine-tuning. Useful for measuring the
    no-recovery accuracy ceiling of a trained dense model.

    Args:
        model: the model whose weights are quantized in place.
        bits: bit-width of the discrete grid.

    Returns:
        (num_tensors_quantized, num_params_quantized).
    """
    _require_parq()
    quantizer = UnifTorchaoQuantizer()
    n_tensors = 0
    n_params = 0
    for name, p in model.named_parameters():
        if not p.requires_grad or not is_quantizable(name):
            continue
        q, _ = quantizer.quantize(p.data, bits, dim=None)  # dim=None -> per-tensor
        p.data.copy_(q)
        n_tensors += 1
        n_params += p.numel()
    if distutils.is_master():
        logging.info(
            f"[PARQ] hard-quantized {n_tensors} tensors ({n_params:,} params) "
            f"to W{bits} per-tensor"
        )
    return n_tensors, n_params


class PARQMLIPTrainEvalUnit(MLIPTrainEvalUnit):
    """``MLIPTrainEvalUnit`` that builds a PARQ ``QuantOptimizer`` for QAT.

    The parent constructs its optimizer through the module-level
    ``_get_optimizer_wd`` helper, which flattens parameters into two groups and
    would drop the per-group ``quant_bits`` metadata PARQ needs. Rather than edit
    the parent, this subclass temporarily redirects that helper to a PARQ factory
    for the duration of ``super().__init__`` only, then restores it. All PARQ code
    is contained here.
    """

    def __init__(
        self,
        job_config: DictConfig,
        model: torch.nn.Module,
        cosine_lr_scheduler_fn: callable,
        tasks: list[Task],
        parq_bits: int,
        parq_lr: float,
        parq_weight_decay: float,
        parq_total_steps: int,
        parq_anneal_start_frac: float = 0.10,
        parq_anneal_end_frac: float = 0.90,
        parq_anneal_steepness: float = 10.0,
        parq_warmup_steps: int = 0,
        parq_quant_period: int = 10,
        optimizer_fn: callable | None = None,
        bf16: bool = False,
        print_every: int = 10,
        clip_grad_norm: float | None = None,
        ema_decay: float = 0.999,
        train_strategy: TrainStrategy = TrainStrategy.DDP,
        debug_checksums_save_path: str | None = None,
        profile_flops: bool = False,
        save_inference_ckpt: bool = True,
    ):
        # optimizer_fn is accepted only so a config can inherit the base
        # train_eval_unit block (which sets it) without erroring; PARQ ignores
        # it and constructs its own QuantOptimizer below.
        _require_parq()
        from fairchem.core.units.mlip_unit import mlip_unit as _mlip_mod

        def _parq_get_optimizer_wd(_optimizer_fn, m):
            # _optimizer_fn is ignored: PARQ builds its own QuantOptimizer.
            return build_parq_optimizer(
                m,
                lr=float(parq_lr),
                weight_decay=float(parq_weight_decay),
                quant_bits=int(parq_bits),
                total_steps=int(parq_total_steps),
                anneal_start_frac=float(parq_anneal_start_frac),
                anneal_end_frac=float(parq_anneal_end_frac),
                anneal_steepness=float(parq_anneal_steepness),
                warmup_steps=int(parq_warmup_steps),
                quant_period=int(parq_quant_period),
            )

        original = _mlip_mod._get_optimizer_wd
        _mlip_mod._get_optimizer_wd = _parq_get_optimizer_wd
        try:
            super().__init__(
                job_config=job_config,
                model=model,
                optimizer_fn=None,  # ignored; PARQ builds its own optimizer
                cosine_lr_scheduler_fn=cosine_lr_scheduler_fn,
                tasks=tasks,
                bf16=bf16,
                print_every=print_every,
                clip_grad_norm=clip_grad_norm,
                ema_decay=ema_decay,
                train_strategy=train_strategy,
                debug_checksums_save_path=debug_checksums_save_path,
                profile_flops=profile_flops,
                save_inference_ckpt=save_inference_ckpt,
            )
        finally:
            _mlip_mod._get_optimizer_wd = original


class PARQMonitorCallback(Callback):
    """Log PARQ optimizer state (inverse slope, bit-width) every ``log_every`` steps."""

    def __init__(self, log_every: int = 100) -> None:
        super().__init__()
        self.log_every = int(log_every)

    def on_train_step_end(self, state: State, unit: MLIPTrainEvalUnit) -> None:  # type: ignore[override]
        tp = getattr(unit, "train_progress", None)
        step = int(tp.num_steps_completed) if tp is not None else -1
        if step % self.log_every != 0:
            return
        opt = getattr(unit, "optimizer", None)
        if opt is None:
            return
        payload: dict[str, float] = {}
        for i, g in enumerate(getattr(opt, "param_groups", [])):
            if "inv_slope" in g:
                payload[f"parq/group{i}_inv_slope"] = float(g["inv_slope"])
                payload[f"parq/group{i}_quant_bits"] = int(g.get("quant_bits", -1))
                break  # only the single quantized group carries inv_slope
        if hasattr(opt, "num_steps"):
            payload["parq/optimizer_num_steps"] = int(opt.num_steps)
        if not payload:
            return
        if distutils.is_master() and unit.logger is not None:
            WandBSingletonLogger.get_instance().log(payload, step=step, commit=False)
        if step % (self.log_every * 5) == 0 and distutils.is_master():
            inv = payload.get("parq/group0_inv_slope")
            inv_str = f"{inv:.4f}" if inv is not None else "n/a"
            logging.info(f"[PARQ] step={step} inv_slope={inv_str}")


class PARQTrainEvalRunner(TrainEvalRunner):
    """``TrainEvalRunner`` that installs a ``PARQMonitorCallback``."""

    def __init__(self, *args: Any, parq_log_every: int = 100, **kwargs: Any) -> None:
        callbacks = list(kwargs.pop("callbacks", None) or [])
        callbacks.append(PARQMonitorCallback(log_every=parq_log_every))
        kwargs["callbacks"] = callbacks
        super().__init__(*args, **kwargs)
