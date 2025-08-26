"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

FastCSP - Fast Crystal Structure Prediction Workflow

Main entry point orchestrating the complete FastCSP crystal structure prediction workflow.
"""

from __future__ import annotations

import argparse

from fairchem.applications.fastcsp.core.workflow.main import main


def cli_main():
    """Main CLI entry point for fastcsp console script."""
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="FastCSP: Fast Crystal Structure Prediction Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow Stages:
  generate                Generate crystal structures using Genarris
  process_generated       Process and deduplicate Genarris outputs
  relax                   Perform UMA-based structure relaxation
  filter                  Energy filtering and duplicate removal for ranking
  evaluate                Compare against experimental data (needs CSD license)
  create_vasp_inputs      Generate DFT input files for validation
  read_vasp_outputs       Process DFT results and compute validation metrics

Usage:
    fastcsp --config <config.yaml> --stages <stage1> <stage2> ...

Example:
  fastcsp --config configs/example_config.yaml --stages generate process_generated relax filter
        """,
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file containing workflow parameters",
    )
    parser.add_argument(
        "--stages",
        type=str,
        nargs="*",
        choices=[
            "generate",  # need Genarris installed
            "process_generated",
            "relax",
            "filter",
            "evaluate",  # need CSD API License
            "free_energy",  # TODO: implement "free_energy"
            "create_vasp_inputs_relaxed",
            "create_vasp_inputs_unrelaxed",  # optional, if you want to create VASP inputs for unrelaxed structures
            "submit_vasp",  # implement your own VASP job submission
            "read_vasp_outputs",
        ],
        default=["generate", "process_generated", "relax", "filter"],
        help="Workflow stages to execute (in order). Default: generate process_generated relax rank",
    )

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
