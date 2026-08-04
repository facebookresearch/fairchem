# Sphere channel pruning — reproduction guide

Structured **sphere-channel** pruning of the UMA-small conserving OMol model: prune the
`sphere_channels` residual width **C (=128)** down to **K** with one global mask shared across
every tensor that uses C, then physically compact to a standard `sphere_channels=K` checkpoint
that runs dense and narrower (real latency + memory win, no sparse kernels).

Full analysis and results: the accompanying `channel-pruning-inference-report.md` write-up.

> Only the **sphere** width is supported. `hidden` / combined variants were evaluated and dropped
> (strictly dominated); see the write-up.

## Target sparsity → kept width

`K = round(128 * (1 - target_sparsity))`. Pick `target_sparsity` (and matching `K`) from:

| target_sparsity | K (`sphere_channels`) | notes |
|---:|---:|---|
| 0.25 | 96 | best all-round: ~dense accuracy, −17% mem, ~1.15× |
| 0.30 | 90 | conservative |
| 0.50 | 64 | max compression: −34% mem, up to 1.45×; accuracy **frontier** |
| 0.625 | 48 | too far — does not recover even at 30 ep (don't use) |

K must be even (charge/spin embedding requirement); the values above already are.

## Reference accuracy (OMol val)

What to expect at the recommended budget. `val loss` is the combined weighted objective logged as
`val/loss` (energy_coef·per-atom-E + force_coef·L2-force); `F cos-sim` is `omol_forces` cosine
similarity.

All rows are trained at the same **30-epoch** budget (fair comparison). `ΔE` is vs dense@30ep.

| config | E MAE (eV) | F MAE (eV/Å) | F cos-sim | val loss | ΔE vs dense |
|---|---:|---:|---:|---:|---:|
| dense · C=128 | 0.1298 | 0.0120 | 0.9942 | 0.1065 | — |
| s0.25 · C=96 | 0.1357 | 0.0126 | 0.9938 | 0.1103 | +4.5% |
| s0.50 · C=64 | 0.1486 | 0.0136 | 0.9931 | 0.1184 | +14.5% |
| s0.625 · C=48 | 0.1667 | 0.0147 | 0.9922 | 0.1287 | +28.4% |

Pruning has a **real, monotonic accuracy cost** at matched budget — it does not reach dense
accuracy at equal compute; the payoff is memory + large-system latency (see the write-up). `C=96`
is the mild-cut sweet spot (+4.5% E for −24% params / −17% mem / ~1.15×); `C=64` trades +14.5% E for
−34% mem / up to 1.45×. `C=48` (+28%) is past the useful frontier — don't ship it. **Always train
the dense baseline at the same `epochs=`**: +10 epochs improves every width ~12%, so a longer budget
raises the whole curve without shrinking the pruning gap.

## Pipeline

### 1. Train (prune + heal) — sphere at a chosen sparsity

```bash
fairchem -c configs/esen/uma_sm_conserving_omol_4M_chanprune.yaml \
    channel_target_sparsity=0.5 \
    channel_norm_stats_num_channels=64 \   # = K; Route B → near-exact compaction
    epochs=30 \                            # ~1.5× the dense budget (closes the prune gap)
    job.scheduler.num_nodes=4
```

- `channel_target_sparsity` — the only knob you normally change. Set `channel_norm_stats_num_channels`
  to the matching **K** (see table) so the RMSNorm keeps its over-channel statistics at the kept
  width during training (**Route B**) — the compacted model then matches near-exactly.
- **`epochs=30`** matters: pruned models need ~1.5× the dense epoch budget to re-converge after the
  channels are removed. At the default 20 ep the s0.5 gap is ~+15%; at 30 ep it is small.
- The `ChannelPruningCallback` runs the schedule automatically: dense warmup
  (`channel_warmup_frac=0.05`) → cubic prune ramp → **heal-freeze** from `channel_healing_start_frac`
  (0.5; its exact value is noise). Uses a plain AdamW — no custom optimizer.
- Not on the `fair_amaia_cw` cluster? Override `override /cluster: <yours>` (or run locally with
  `job.device_type=CPU` for a smoke test).

Output: `<run_dir>/checkpoints/final/inference_ckpt.pt` (the trained, still-full-width-but-zeroed
model).

To sweep sparsities, launch one job per level, varying `channel_target_sparsity` + the matching
`channel_norm_stats_num_channels`.

### 2. Compact — physically remove the zeroed channels

```bash
python scripts/compact_save_ckpt.py \
    <run_dir>/checkpoints/final/inference_ckpt.pt \
    sphere_sXX_compact_inference.pt
```

Auto-detects the kept channels (the non-zeroed ones), rebuilds at `sphere_channels=K`, copies the
sliced weights, persists the norm-stats divisor, and round-trips through the standard
`MLIPPredictUnit` / `FAIRChemCalculator` path. The result is a **standard reduced-width checkpoint**
(deployable as-is). `compact_channels.py` holds the reusable compaction logic.

### 3. (Optional) Re-heal — close the last compaction residual

Only needed if you did *not* train Route B, or want to squeeze the last ~1e-3 eV/Å:

```bash
fairchem -c configs/esen/uma_sm_conserving_omol_4M_reheal.yaml \
    reheal_ckpt=sphere_sXX_compact_inference.pt \
    job.scheduler.num_nodes=4
# 2000 steps, LR 1e-4, ~15 min on 4×8 H100
```

### 4. Benchmark (optional) — latency + memory + dynamics

```bash
# latency + peak memory vs a dense baseline, swept over system size:
python scripts/bench_prod_latency.py dense_inference.pt sphere_sXX_compact_inference.pt \
    --natoms 255 1023 2049

# latency/memory of an arbitrary native width WITHOUT training (untrained shape):
python scripts/make_native_ckpt.py dense_inference.pt native_S96.pt --sphere 96

# MD dynamics fidelity (energy drift, dense vs compacted) over a VelocityVerlet trajectory:
python scripts/bench_md_latency.py dense_inference.pt sphere_sXX_compact_inference.pt \
    --system water --natoms 300 600 1200
```

## Files

| file | role |
|---|---|
| `../src/fairchem/core/models/uma/channel_pruning.py` | `ChannelPruningCallback` + sphere spec / mask logic |
| `../configs/esen/uma_sm_conserving_omol_4M_chanprune.yaml` | training config (prune + heal) |
| `../configs/esen/uma_sm_conserving_omol_4M_reheal.yaml` | optional post-compaction re-heal |
| `../configs/esen/uma_sm_conserving_omol_4M_dense.yaml` | dense baseline (train at matched `epochs=` for a fair comparison) |
| `compact_channels.py` | compaction logic (importlib-loaded by the driver) |
| `compact_save_ckpt.py` | driver: trained → compacted standard checkpoint |
| `bench_prod_latency.py` | latency + peak-memory sweep (`umas_fast_pytorch`, TF32) |
| `make_native_ckpt.py` | build an untrained native-width checkpoint for latency/memory only |
| `bench_md_latency.py` | ASE `VelocityVerlet` dynamics-fidelity check |

> Fair comparisons: always train the dense baseline at the **same `epochs=`** as the pruned run —
> a longer budget improves both, so cross-budget comparisons are misleading.
