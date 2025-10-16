"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

FastCSP - Fast Crystal Structure Prediction Workflow

Main entry point orchestrating the complete FastCSP workflow.
"""

from __future__ import annotations

import argparse

from fairchem.applications.fastcsp.core.workflow.main import main


def cli_main():
    """Main CLI entry point for FastCSP workflow."""
    parser = argparse.ArgumentParser(
        description="FastCSP: Fast Crystal Structure Prediction Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Workflow Stages:
  generate                      Generate crystal structures using Genarris
  process_generated             Process and deduplicate Genarris outputs
  relax                         Perform UMA-based structure relaxations
  filter                        Filtering and duplicate removal for ranking
  evaluate                      Compare against experimental structures
  free_energy                   Compute free energy corrections

Usage:
  fastcsp --config <config.yaml> --stages <stage1> <stage2> ...

Example:
  fastcsp --config configs/example_config.yaml --stages generate process_generated relax filter
        """,
    )

    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to YAML configuration file containing workflow parameters",
    )
    parser.add_argument(
        "-s",
        "--stages",
        type=str,
        nargs="*",
        choices=[
            "generate",  # need Genarris installed
            "process_generated",
            "relax",
            "filter",
            "evaluate",  # optional, can require CSD API License
            "free_energy",  # optional, TODO: implement "free_energy"
        ],
        default=["generate", "process_generated", "relax", "filter"],
        help="Workflow stages to execute (in order). Default: generate process_generated relax filter",
    )

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
