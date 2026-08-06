# Sphere channel pruning — reproduction guide

Structured **sphere-channel** pruning of the UMA-small conserving OMol model: prune the
`sphere_channels` residual width **C (=128)** down to **K** with one global mask shared across
every tensor that uses C, then physically compact to a standard `sphere_channels=K` checkpoint
that runs dense and narrower (real latency + memory win, no sparse kernels).

Full analysis and results: the accompanying `channel-pruning-inference-report.md` write-up.

> **Status (2026-08-06): project closed out.** Recommended recipe:
> **SOAP, 10 epochs** — `../configs/esen/uma_sm_conserving_omol_4M_chanprune_soap.yaml`.
> The earlier AdamW pruning config was retired in favour of it: SOAP/10ep beats AdamW/20ep
> across the whole Pareto front in both energy and forces at half the compute (see
> **Best results** below).

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

## Best results (matched Pareto sweep, 2026-08-06)

The recommended recipe — **SOAP, 10 epochs** — dominates the prior **AdamW, 20 epochs** recipe
across the entire front: lower energy AND force error at every width, using **half the training
steps** (~50k vs ~101k). `F cos-sim` is `omol_forces` cosine similarity; `val loss` is the combined
weighted objective logged as `val/loss` (energy_coef·per-atom-E + force_coef·L2-force). All values
are aggregate OMol val at end of training.

| width | speedup¹ | recipe | E MAE (eV) | F MAE (eV/Å) | F cos-sim | val loss |
|---|---:|---|---:|---:|---:|---:|
| dense · C=128 | 1.00× | **SOAP 10ep** | **0.1163** | **0.0110** | **0.9948** | **0.0956** |
|               |       | AdamW 20ep    | 0.1476     | 0.0131     | 0.9935     | 0.1117     |
| s0.25 · C=96  | 1.17× | **SOAP 10ep** | **0.1240** | **0.0116** | **0.9944** | **0.1008** |
|               |       | AdamW 20ep    | 0.1558     | 0.0138     | 0.9929     | 0.1176     |
| s0.50 · C=64  | 1.45× | **SOAP 10ep** | **0.1443** | **0.0130** | **0.9933** | **0.1113** |
|               |       | AdamW 20ep    | 0.1705     | 0.0148     | 0.9922     | 0.1269     |

¹ dense-inference latency speedup vs C=128 @2049 atoms; depends only on the compacted width.

Findings from the front:
- **SOAP/10ep is the Pareto frontier** — strictly below AdamW/20ep in E, F, cos-sim and val loss at
  every width, at ~½ the compute. SOAP/5ep (not shown) is *undertrained*, not a bad SOAP×pruning
  interaction: 5 epochs starves the healing phase (schedule is in step-fractions), so its pruning
  penalty is inflated (+42 meV at s0.5 vs +28 meV at 10ep). Do not judge SOAP at 5 epochs.
- Pruning still has a **real, monotonic accuracy cost** at matched budget — it does not reach dense
  accuracy at equal compute; the payoff is memory + large-system latency (see the write-up). `C=96`
  is the mild-cut sweet spot (+7.7 meV E for −24% params / −17% mem / ~1.15×); `C=64` is the
  compression frontier (+28 meV E for −34% mem / up to 1.45×). `C=48` does not recover even at 30 ep
  — past the useful frontier, don't ship it.
- **Always train the dense baseline at the same optimizer + `epochs=`** as the pruned run: a
  longer/stronger budget lifts the whole curve without shrinking the pruning gap.

> These are pre-compaction OMol val numbers; the RMSNorm centering-leak fix makes sphere compaction
> output-exact, so the compacted checkpoints reproduce them.

## Pipeline

### 1. Train (prune + heal) — sphere at a chosen sparsity

```bash
fairchem -c configs/esen/uma_sm_conserving_omol_4M_chanprune_soap.yaml \
    channel_target_sparsity=0.5 \          # 0.25 -> C=96, 0.5 -> C=64
    job.scheduler.num_nodes=4
```

- `channel_target_sparsity` — the only knob you normally change.
- **Optimizer + budget are baked into the config: SOAP, 10 epochs** (~50k steps at 4 nodes). This is
  the frontier recipe (beats AdamW/20ep everywhere at half the compute — see Best results). The
  budget is spent mostly on *healing*; do not drop below 10 epochs.
- **Compaction is output-exact.** The RMSNorm centering-leak fix means standard training
  (`channel_norm_stats_num_channels=null`, the default) compacts exactly — you no longer need the
  old Route-B (`channel_norm_stats_num_channels=K`) trick, and step 3 (re-heal) is essentially never
  required.
- The `ChannelPruningCallback` runs the schedule automatically: dense warmup
  (`channel_warmup_frac=0.05`) → cubic prune ramp → **heal-freeze** from `channel_healing_start_frac`
  (0.5; its exact value is noise).
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

### 3. (Optional, rarely needed) Re-heal — close the last compaction residual

Compaction is now output-exact, so this is essentially never required. Kept only for old
checkpoints trained before the RMSNorm fix, or to squeeze a last ~1e-3 eV/Å:

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
| `../configs/esen/uma_sm_conserving_omol_4M_chanprune_soap.yaml` | training config (prune + heal) — **SOAP, 10 ep, default recipe** |
| `../configs/esen/uma_sm_conserving_omol_4M_reheal.yaml` | optional post-compaction re-heal |
| `../configs/esen/uma_sm_conserving_omol_4M_dense.yaml` | dense baseline (train at matched `epochs=` for a fair comparison) |
| `compact_channels.py` | compaction logic (importlib-loaded by the driver) |
| `compact_save_ckpt.py` | driver: trained → compacted standard checkpoint |
| `bench_prod_latency.py` | latency + peak-memory sweep (`umas_fast_pytorch`, TF32) |
| `make_native_ckpt.py` | build an untrained native-width checkpoint for latency/memory only |
| `bench_md_latency.py` | ASE `VelocityVerlet` dynamics-fidelity check |

> Fair comparisons: always train the dense baseline at the **same `epochs=`** as the pruned run —
> a longer budget improves both, so cross-budget comparisons are misleading.
