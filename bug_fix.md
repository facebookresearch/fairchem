# UMA checkpoint identity & `include_self` — bug fix plan (v2, post-review)

> v2 incorporates 9 adversarial reviews. Changes from v1 are marked **[v2]**.

## Background

UMA checkpoints are immutable (already shipped). Each generation must load with a
specific `include_self` behavior in the MoE composition reduction
(`eSCNMDMoeBackbone.set_MOLE_coefficients`, only reached when `num_experts > 0`):

| Generation | `include_self` |
|---|---|
| 1.0 | True (but deprecated → rejected on load) |
| 1.1 | False |
| 1.2 | True |
| 1.2.1+ | False |

Today `include_self = np.isclose(self.model_version, 1.0)` with
`model_version: float = 1.0`. Two defects:

1. **Regression — untagged collides with 1.0.** A freshly-trained model sets
   neither `model_id` nor `model_version`; the shim reads `model_version` from the
   config as `None`, classifies it as deprecated 1.0, and **load hard-fails**.
   Confirmed: `test_direct_mole_inference_modes` (CPU) trains a normal model and
   dies with `RuntimeError: UMA 1.0 ... no longer supported`. All real training
   backbones (`K4L2/K10L4/K16L6`) and the test backbone (`K2L2`) set neither field.
2. **`isclose(model_version, 1.0)` is a coincidence, not a rule.** It is correct
   for shipped 1.0/1.1/1.2 only because of what each happened to store; it cannot
   express the real per-generation mapping (1.2 True, neighbors False).

## Ground truth (measured from shipped checkpoints)

| Checkpoint | `model_id` | `backbone.model_version` |
|---|---|---|
| `uma-s-1.pt` (1.0) | None | absent |
| `uma-s-1p1` / `uma-m-1p1` | None | `1.1` |
| `uma-s-1p2` | `"UMA-S-1.2"` | absent |

1.1 is identified by `model_version`; 1.2 by `model_id`; 1.0 / untagged carry
neither and are indistinguishable.

## Principle

`model_id` is the single source of truth for generation. The compat shim does
three things: **classify, back-fill `model_id`, reject**, and then **authoritatively
set** the derived behavior flag `include_self_bug` on the backbone config. The
backbone consumes a plain bool and knows nothing about version strings.

**[v2] `model_version` is left unchanged** (default stays as-is). It is no longer
read for behavior, only by the shim's classifier as the legacy fallback that
identifies shipped 1.1. Flipping its default is NOT part of this fix — the
regression is fixed by the shim's classification of the config, not by the class
default (the shim sees `None` from the config regardless of the default).

## Behavior contract

| Generation | `model_id` in ckpt | `model_version` in ckpt | Classified `version` | `include_self_bug` = `(version == "1.2")` |
|---|---|---|---|---|
| **1.1** | absent → back-fill `"UMA-1.1"` (loud warn) | `1.1` | `"1.1"` | `False` |
| **1.2** (any size: S/M/L/bare) | `"UMA-*-1.2"` | `None` | `"1.2"` | `True` |
| **1.2.1+** | `"UMA-1.2.1"` (training sets it) | `None` | `"unknown_uma"` | `False` |
| **untagged / 1.0** | absent | absent | `"unidentified"` | n/a → **raise** "identity required" |

This preserves exactly how shipped 1.1/1.2 run today → numeric no-op for releases.

## Canonical mapping (single source) — **[v2]**

In `compat.py`, one table + helper, referenced by the shim AND the tests:

```python
# include_self for the MoE composition reduction, per UMA generation.
# Only 1.2 uses True; 1.1 and 1.2.1+ use False; 1.0 is rejected before lookup.
# WHEN A NEW GENERATION SHIPS WITH DIFFERENT BEHAVIOR: add it here + update the
# docstring table + changelog + tests.
_UMA_INCLUDE_SELF: dict[str, bool] = {"1.2": True}

def uma_include_self_bug(version: str) -> bool:
    return _UMA_INCLUDE_SELF.get(version, False)
```

## Implementation

### 1. `escn_moe.py` (backbone)
- Add `include_self_bug: bool = False` as an **explicit** `__init__` param
  (stored as `self.include_self_bug`, **not** forwarded via `**kwargs` — the
  parent `eSCNMDBackbone.__init__` has no `**kwargs` and would `TypeError`). **[v2]**
- In `set_MOLE_coefficients`: `include_self = self.include_self_bug`
  (remove `np.isclose(self.model_version, 1.0)`).
- Add docstring + a comment at the `index_reduce_` call site pointing to `compat.py`. **[v2]**
- `model_version` param/default **unchanged**. **[v2]**

### 2. `compat.py` (shim)
- `get_uma_version`: `model_id` first; else `model_version == 1.1` → `"1.1"`;
  else neither → `"unidentified"`; explicit `model_version == 1.0` → `"1.0"`.
- `apply_uma_compat_fixups`:
  - `"1.1"` → back-fill `model_id="UMA-1.1"` (loud warning).
  - `"unidentified"` / `"1.0"` → **raise**, with **distinct** messages: **[v2]**
    - unidentified → "UMA checkpoint identity required: set `model_id` …" (+ note
      it may be a deprecated 1.0 → `pip install 'fairchem-core<=2.21.0'`).
    - explicit 1.0 → the existing 1.0-deprecation message.
  - For every UMA backbone that survives (1.1/1.2/unknown_uma): **authoritatively
    set** `backbone.include_self_bug = uma_include_self_bug(version)`, using
    `open_dict` + `force_add` for struct-mode DictConfig (same as the `model_id`
    back-fill). Always overwrite (don't "set only if absent") so re-saved
    finetune configs can't carry a stale value. **[v2]**

### 3. `base.py` (`HydraModel`)
- No change.

### 4. Configs
- `model_id` on the **`model:` (HydraModel)** level; `include_self_bug` on the
  **`backbone:`** level. **[v2]** Release backbones (`K4L2/K10L4/K16L6`,
  perf_check) set `model_id`.
- **Shared test fixtures are tagged via training overrides, not new yaml** **[v3]**
  (see "Test harness" below). Because the shim derives `include_self_bug`
  authoritatively from `model_id`, a fixture gets the right flag purely by its
  `model_id` — no need to set the bool at train time (it would be overwritten on
  load anyway).

### 5. Migration for already-trained untagged checkpoints — **[v2]**
- These now fail to load. Ship a documented one-liner to tag in place, e.g.
  `ckpt = torch.load(p); ckpt.model_config["model_id"] = "UMA-1.2.1"; torch.save(ckpt, p)`,
  in the error message and changelog.

## Test harness — real configs + overrides (not synthetic dicts) **[v3]**

Behavior tests are centered on the UMA models we already have configs for, driven
by overrides — not hand-built `model_config` dicts. The pieces:

- **Existing session fixtures** (`tests/core/conftest.py`) train a real `K2L2`
  (`eSCNMDMoeBackbone`) from `tests/core/units/mlip_unit/test_mlip_train.yaml` via
  `launch_main` and load through the shim path
  (`convert_train_checkpoint_to_inference_checkpoint` → `load_inference_model`):
  - `direct_checkpoint` — `num_experts=0`
  - `direct_mole_checkpoint` — `num_experts=8`
  - `conserving_mole_checkpoint` — `num_experts=8`, conserving head
- **Training override** to bake identity into a checkpoint:
  - `+runner.train_eval_unit.model.model_id=UMA-S-1.2` (HydraModel level)
  - `+runner.train_eval_unit.model.backbone.model_version=1.1` (backbone level, to
    simulate a shipped-1.1 checkpoint with no model_id)
- **Load override** (merged *after* the shim, `utils.py:55→58`) to flip behavior on
  an already-trained checkpoint:
  - `load_predict_unit(path, overrides={"backbone": {"include_self_bug": True}})`
  - `load_predict_unit(path, overrides={"model_id": "..."} )`

**Fixture tagging (regression fix for all existing consumers):** the three shared
fixtures currently produce untagged checkpoints, which is exactly what breaks
`test_direct_mole_inference_modes` et al. Add a `model_id` training override so they
load. Choosing the `model_id` chooses `include_self_bug` (shim derives it): use a
`1.2` id to keep the MoE fixtures' current `include_self=True` (no snapshot churn),
or a non-1.2 id and refresh snapshots. Decide by running D.

## Testing plan

### A. Classification edge cases — fast unit (CPU)
Keep the cheap pure-function tests in `test_compat.py` (they cover what configs
can't easily reach):
- no `model_id` + no `model_version` → raises, message contains "identity required". **[v2]**
- explicit `model_version == 1.0` → raises, message contains "UMA 1.0" (distinct). **[v2]**
- 1.1 (`model_version=1.1`) → back-fills `model_id="UMA-1.1"` + loud warning.
- 1.2 (`UMA-S-1.2` **and** `UMA-M-1.2`) → classified `"1.2"`, no back-fill.
- struct-mode DictConfig write path (`open_dict`) works. **[v2]**

### B. Shim sets the flag on a REAL checkpoint (the linchpin) (CPU) — **[v3]**
Train `K2L2` (num_experts=8) via `launch_main` with a chosen identity, load it, and
assert the **instantiated backbone** carries the right flag:
- train with `+...model.model_id=UMA-S-1.2` → loaded `backbone.include_self_bug is True`.
- train with `+...model.backbone.model_version=1.1` (no model_id) → shim back-fills
  `model_id="UMA-1.1"`, loaded `backbone.include_self_bug is False`.
- train with `+...model.model_id=UMA-1.2.1` → loaded flag `False`.
Assert via `predictor.model.module.backbone.include_self_bug`.

### C. The flag has teeth — REAL MoE checkpoint + load override (CPU) — **[v3]**
Reuse `direct_mole_checkpoint` (num_experts=8). Load it twice and compare a forward
pass:
- `load_predict_unit(ckpt, overrides={"backbone": {"include_self_bug": True}})`
- `load_predict_unit(ckpt, overrides={"backbone": {"include_self_bug": False}})`
Assert the predicted energies/forces **differ** → proves the flag is wired into
`set_MOLE_coefficients`, on a real model, with zero synthetic construction.

### D. Round-trip regression — existing fixtures (CPU)
- `direct_checkpoint` (num_experts=0): train (now tagged) → load **succeeds**
  (the regression that currently raises "UMA 1.0").
- `direct_mole`/`conserving_mole`: load succeeds; `test_direct_mole_inference_modes`
  + inference-mode consistency tests pass.
- **Untagged still raises:** load a fixture checkpoint, strip `model_id` from its
  saved `model_config`, re-save to a temp path, and assert `load_predict_unit`
  raises "identity required". (Exercises the real load path, not a synthetic dict.)

### E. Numeric-equivalence guard — pretrained, real (GPU)
- `@pytest.mark.pretrained("uma-s-1p1","uma-s-1p2")`: load via
  `get_predict_unit_for_test`, assert resolved `backbone.include_self_bug`
  (1.1→False, 1.2→True) and snapshot energies/forces on the
  `test_multiple_dataset_predict` systems (in-repo baseline so "unchanged" is
  mechanically checkable). **[v2]**

### F. Deprecation — pretrained (GPU / opt-in)
- `uma-s-1.pt` → raises (`test_uma_1p0_predict_unit_raises`).

**Gating:** A runs on CPU here (fast, ms); B/C/D train the `K2L2` fixtures
(~minutes, CPU) — but reuse the session-scoped fixtures so cost is amortized; E/F
need the H100. Pre-commit on every modified file.

## Documentation — **[v2]**
- `compat.py` module docstring: add the `include_self_bug` behavior + the canonical
  table + a "when a new generation ships, change these lines" checklist.
- `set_MOLE_coefficients` docstring + inline comment at `index_reduce_`:
  `include_self=self.include_self_bug  # per-generation, injected by compat.py`.
- `uma_changelog.md`: note (a) 1.0 deprecated, (b) identity (`model_id`) now
  required to load + how to tag an old checkpoint, (c) `include_self` is handled
  automatically per generation.

## Notes
- `include_self_bug` only matters for `num_experts > 0`; `model_id` is required at
  load for all UMA models.
- `1.2.1+` classifies as `unknown_uma` today (the regex matches only `1.<minor>`,
  not patch versions). That yields the correct `include_self_bug=False` via the
  default, but logs an "unclassified" warning. Making 1.2.1 first-class (richer
  regex + classifier entry) is a small, separate follow-up. **[v2]**
- Decision kept from review: shim-writes-bool (hardened) over backbone-reads-`model_id`,
  to keep the backbone free of version-string logic. The hardening (explicit param,
  `open_dict`, authoritative overwrite) neutralizes the re-save / override footguns. **[v2]**
