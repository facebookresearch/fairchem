from __future__ import annotations

import argparse
from typing import Any

import torch


def fix_checkpoint(
    ckpt_path: str,
    max_atoms: int = 1000,
    add_stress: bool = False,
    enable_compile: bool = False,
) -> str:
    """Load a torch checkpoint, adjust backbone config, and save a fixed copy.

    Returns the output path.
    """
    ckpt: Any = torch.load(ckpt_path, weights_only=False)

    # Update config as requested
    ckpt.model_config["backbone"]["use_compile"] = enable_compile
    ckpt.model_config["backbone"]["use_padding"] = enable_compile
    ckpt.model_config["backbone"]["max_atoms"] = int(max_atoms)

    out_path = ckpt_path[:-3] + "_fixed.pt"
    torch.save(ckpt, out_path)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a checkpoint, set backbone.use_compile/use_padding, and "
            "backbone.max_atoms, and save as *_fixed.pt",
        )
    )
    parser.add_argument("ckpt_path", type=str, help="Path to checkpoint .pt file")
    parser.add_argument(
        "--enable-compile",
        action="store_true",
        help="Enable compile for the checkpoint",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=1000,
        dest="max_atoms",
        help="Value for model_config.backbone.max_atoms (default: 1000)",
    )
    parser.add_argument(
        "--add-stress",
        action="store_true",
        help="Add stress task to the checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = fix_checkpoint(
        args.ckpt_path, args.max_atoms, args.add_stress, args.enable_compile
    )
    print("Fixed checkpoint saved to: ", out_path)


if __name__ == "__main__":
    main()
