"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

# Each subpackage imports only from the ones listed before it
# (tests/core/regression_suite/test_layers.py fails otherwise):
#   spec        schema, hashing, transforms, plan helpers, and profiles
#   suites      the suites and the table that loads them
#   artifacts   run records, sealed baselines, and storage paths
#   evaluation  pairing with FP64, statistics, calibration, and grading
#   reporting   text, HTML, and W&B reports
#   execution   runners, campaigns, Slurm launches, and smoke runs
#   cli.py      the fairchem-regression command
