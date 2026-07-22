"""Robust Exponential Moving Average utilities for fairchem.

`NameMatchedAveragedModel` is a drop-in `torch.optim.swa_utils.AveragedModel`
subclass whose `update_parameters` matches live and EMA state by NAME instead
of by positional zip.

Motivation
----------
`torch.optim.swa_utils.AveragedModel.update_parameters` iterates parameters
and buffers with `zip(self.module.buffers(), model.buffers())`. Positional
zip is fragile whenever the wrapped module's buffer tree can drift between
the moment of `AveragedModel.__init__` (which deep-copies the module) and a
later `update_parameters` call. Common causes of drift in modern training
setups:

- `nn.utils.parametrize.register_parametrization` adds `parametrizations.*.mask`
  buffers under wrapped modules. Multi-parametrization or shared-mask setups
  interact with `deepcopy`'s memoization in ways that can shuffle the
  iteration order of `named_buffers` between the live and EMA modules.
- Any code that replaces a buffer registration after `AveragedModel` was
  constructed (e.g. moving a shared mask tensor to a new device by re-pointing
  the buffer entry) alters what `named_buffers` yields on the live side while
  the EMA copy keeps its original layout.
- DDP / FSDP wrapping adds their own bookkeeping buffers, which interact
  poorly with `deepcopy` in some PyTorch releases.

Symptomatic failure: `RuntimeError: The size of tensor a (N) must match the
size of tensor b (M)` at the positional buffer-copy line of `update_parameters`.

There is also a second, subtler failure mode driven by
`named_buffers()` defaulting to `remove_duplicate=True`: shared buffer
tensors (e.g. one project-owned mask referenced by many parametrizations)
are yielded once, under their FIRST-seen name. When those references are
re-pointed post-`deepcopy`, the "first name" that survives dedup can
differ between the live and EMA copies — even though the underlying
tensors would sync fine. A name-matched implementation that used dedup
would then either crash on the spurious "buffer name sets differ" or,
worse, silently drop the sync for shared buffers, letting the EMA copy
hold stale references while the live model evolves.

Fix
---
`AveragedModel.__init__` uses `self.module = copy.deepcopy(model)`, which
guarantees that live and EMA share identical `named_parameters()` /
`named_buffers()` name trees AT CONSTRUCTION TIME. Matching by name at
every `update_parameters` call is robust to any of the reorderings above.

To also cover shared buffers correctly, buffers are iterated with
`remove_duplicate=False`, so every registered path is walked and shared
tensors get synced under all their names. Redundant writes to the same
underlying tensor are cheap and correct.

In the healthy case where positional and name-matched iteration would produce
identical results, `NameMatchedAveragedModel` is bit-exact with the vanilla
`AveragedModel` (verified via a parity test at `max_abs_diff = 0.0`). It adds
no observable overhead — the extra dict construction is dwarfed by the tensor
copies it performs.

Usage
-----
Drop-in replacement wherever you'd use `torch.optim.swa_utils.AveragedModel`:

    from fairchem.core.common.ema import NameMatchedAveragedModel

    ema = NameMatchedAveragedModel(
        model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999),
    )
    # ... training loop, calling ema.update_parameters(model) each step ...
    # ema.module(x) returns the EMA-weighted forward pass at eval time.

Failure modes
-------------
Because same-name buffers are expected to have identical shapes (deep-copy
guarantees this at construction), any post-construction shape drift is
reported as a `RuntimeError` naming the buffer. Same for name-set differences
(a buffer disappearing from one side but not the other). This is intentional:
silent skipping would let the EMA slowly diverge from the live model in ways
that are very hard to detect downstream. Prefer to fail loudly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from torch.optim import swa_utils

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module


class NameMatchedAveragedModel(swa_utils.AveragedModel):
    """`AveragedModel` variant that syncs parameters/buffers by NAME.

    Identical semantics to `torch.optim.swa_utils.AveragedModel` in the
    healthy case where positional and name-matched iteration agree. Robust
    against `nn.utils.parametrize` + `deepcopy` and DDP interactions that
    can misalign the positional buffer order in the vanilla implementation.

    See module docstring for the failure mode this fixes.
    """

    def update_parameters(self, model: Module) -> None:  # type: ignore[override]
        # ---- Parameters: match by name ------------------------------------
        ema_pd = dict(self.module.named_parameters())
        live_pd = dict(model.named_parameters())
        only_ema = sorted(set(ema_pd) - set(live_pd))
        only_live = sorted(set(live_pd) - set(ema_pd))
        if only_ema or only_live:
            raise RuntimeError(
                "NameMatchedAveragedModel: parameter name sets differ between "
                f"live and EMA. only_ema ({len(only_ema)}): {only_ema}. "
                f"only_live ({len(only_live)}): {only_live}."
            )
        common_params = sorted(ema_pd)

        self_param_detached: list[Optional[Tensor]] = []
        model_param_detached: list[Optional[Tensor]] = []
        for name in common_params:
            p_averaged = ema_pd[name]
            p_model = live_pd[name]
            p_model_ = p_model.detach().to(p_averaged.device)
            self_param_detached.append(p_averaged.detach())
            model_param_detached.append(p_model_)
            if self.n_averaged == 0:
                # First call: direct copy (matches AveragedModel's behavior).
                p_averaged.detach().copy_(p_model_)

        if self.n_averaged > 0:
            if self.multi_avg_fn is not None or self.avg_fn is None:
                grouped = swa_utils._group_tensors_by_device_and_dtype(  # type: ignore[attr-defined]
                    [self_param_detached, model_param_detached]
                )
                for (device, _), (
                    [self_params, model_params],
                    _,
                ) in grouped.items():
                    if self.multi_avg_fn:
                        self.multi_avg_fn(
                            self_params,
                            model_params,
                            self.n_averaged.to(device),
                        )
                    else:
                        multi_avg_fn = swa_utils.get_swa_multi_avg_fn()
                        multi_avg_fn(
                            self_params,
                            model_params,
                            self.n_averaged.to(device),
                        )
            else:
                for p_avg, p_mod in zip(self_param_detached, model_param_detached):
                    n_averaged = self.n_averaged.to(p_avg.device)
                    p_avg.detach().copy_(
                        self.avg_fn(p_avg.detach(), p_mod, n_averaged),
                    )

        # ---- Buffers: match by name, INCLUDING shared duplicates ----------
        if not self.use_buffers:
            # remove_duplicate=False so a buffer tensor shared by multiple
            # parametrizations appears under EVERY name it's registered as.
            # Otherwise dedup can pick different "first names" for shared
            # tensors on live vs EMA when buffer references are re-pointed
            # post-deepcopy (e.g. a sparsifier that consolidates per-parametrization
            # mask buffers to a single project-owned tensor), producing
            # spurious name-set mismatches AND, in a silent-skip variant,
            # missing the sync entirely.
            ema_bd = dict(self.module.named_buffers(remove_duplicate=False))
            live_bd = dict(model.named_buffers(remove_duplicate=False))
            only_ema_b = sorted(set(ema_bd) - set(live_bd))
            only_live_b = sorted(set(live_bd) - set(ema_bd))
            if only_ema_b or only_live_b:
                # Full list to help debug — silent truncation obscures the real diff.
                raise RuntimeError(
                    "NameMatchedAveragedModel: buffer name sets differ between "
                    f"live and EMA. only_ema ({len(only_ema_b)}): {only_ema_b}. "
                    f"only_live ({len(only_live_b)}): {only_live_b}."
                )
            for name in sorted(ema_bd):
                b_swa = ema_bd[name]
                b_model = live_bd[name]
                if b_swa.shape != b_model.shape:
                    # Deep-copy at construction time guarantees identical
                    # shapes. Post-construction shape drift is a bug we
                    # surface instead of silently skipping.
                    raise RuntimeError(
                        f"NameMatchedAveragedModel: buffer '{name}' shape "
                        f"mismatch between live and EMA "
                        f"(live={tuple(b_model.shape)}, ema={tuple(b_swa.shape)}). "
                        "Something has mutated the buffer tree after "
                        "AveragedModel construction — investigate.",
                    )
                b_swa.detach().copy_(b_model.detach().to(b_swa.device))

        self.n_averaged += 1
