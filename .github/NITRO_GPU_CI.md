<!--
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-->

# UMA Nitro GPU CI activation

The `test_nitro_gpu (3.14)` job is implemented but **activation is pending**.
Until the `UMA_NITRO_RUNNER` repository variable identifies a provisioned GPU
runner, GitHub skips the job and the supported-GPU coverage gap remains open.

## Activate

1. Provision a runner with one CUDA GPU of compute capability 8.x through 12.x,
   a CUDA-enabled PyTorch 2.13 wheel, a compatible driver, and package/model
   download access. Prefer an H100 to exercise native Hopper CuTe/GEMM kernels;
   SM8 devices use the portable policy and do not establish native H100 coverage.
2. Set repository variable `UMA_NITRO_RUNNER` to that runner's actual label.
   No runner label is assumed or provisioned by this PR.
3. Dispatch `tests` on the branch containing these workflow changes. A
   `workflow_run` uses the default branch's workflow definition, and the existing
   setup action rejects workflow changes to prevent a misleading result.
4. Review `nitro-preflight.log`, `nitro-coverage.json`, the full pytest
   log and JUnit artifact from a successful run. The preflight must report the
   selected GPU and effective Nitro policy; an H100 must report capability 9.0
   and `native_sm90_plus`.
5. Make `test_nitro_gpu (3.14)` a required status for the applicable branches
   only after activation and a successful run. Record the runner and the first
   successful run in the investigation plan before closing the coverage item.

The job uses the existing `external-testing` environment, which requires reviewer
approval before checkout or any test code runs. For `workflow_run`, review the
triggering PR commit: the workflow definition runs from the default branch while
checkout uses the upstream head SHA. Approval also applies to manual dispatches. Keep the environment's required
reviewer protection configured when activating the runner.

The lane preserves the approved checkout SHA, contributor approval gate,
per-step Hugging Face secret handling and manual status setup/cleanup. Its
45-minute job timeout bounds installation and initial kernel compilation.
Tune it from measured successful runs; it is not a measured runtime guarantee.

## Coverage gate

The lane executes the entire `tests/core/models/uma/uma_nitro` GPU subset
serially, including compilation. It does not use an eager-inference override or
a sweep filter: the prediction module pins `uma-s-1p2p1`, which is absent from
the existing 1p1/1p2 model sweeps. Pytest's empty-selection exit remains fatal.
The existing T4 lane retains its valid fallback coverage.
The command sets `--rootdir=.` because the config file is below `packages/`;
this preserves module identities in JUnit for tests under the repository root.

JUnit verification runs even when pytest fails. It requires the specific current
18 prediction cases and 12 supported-GPU fallback cases to pass, including their
pinned checkpoint and parameter IDs. It rejects skipped prediction or required
fallback cases and duplicated identities, and verifies
named calculator, default inference, energy/force/stress agreement, capacity
recovery and runtime fallback groups. The prediction tests' `_assert_runs_nitro`
assertions prove actual `UMANitroModel` execution. New required parametrizations
should update the required case manifest when the suite changes. A different
passing parametrization cannot replace a required scenario.

JUnit, preflight output and the coverage summary are retained for 30 days with
`always()` artifact upload. Pytest output is in the normal GitHub job log.
GNU timeout sends KILL at the phase deadline and returns 137; JUnit can be
missing or incomplete after termination. Other skipped tests appear in the
coverage summary and cannot replace required passed cases. Missing or invalid
XML is fatal to the coverage gate.

## Local validation without model downloads

```bash
python -m pytest tests/core/scripts/test_nitro_ci.py \
  --rootdir=. -c packages/fairchem-core/pyproject.toml
CUDA_VISIBLE_DEVICES=0 python .github/scripts/nitro_ci.py \
  --preflight nitro-preflight.log
```

The helper regression tests need no pretrained checkpoints or GPU imports.
Full acceptance still requires the checksum-accepted pinned checkpoint and
the CI dependency pins on the provisioned supported GPU. An out-of-band random
checkpoint smoke check cannot replace this acceptance run.
