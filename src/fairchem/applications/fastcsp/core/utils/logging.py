"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

FastCSP Centralized Logging System
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fairchem.applications.fastcsp.core.utils.configs import LoggingConfig


def setup_fastcsp_logger(
    name: str = "fastcsp",
    log_file: str | Path | None = None,
    level: str = "INFO",
    console_output: bool = True,
    append: bool = True,
    format_str: str | None = None,
) -> logging.Logger:
    """
    Set up the centralized FastCSP logger.

    Args:
        name: Logger name (default: "fastcsp").
        log_file: Path to log file (default: None = no file logging).
        level: Logging level (default: "INFO").
        console_output: Whether to output to console (default: True).
        append: Whether to append to existing log file (default: True).
        format_str: Log message format string (default: standard format).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a" if append else "w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def setup_fastcsp_logger_from_config(
    config: LoggingConfig, name: str = "fastcsp"
) -> logging.Logger:
    """
    Set up the centralized FastCSP logger from a LoggingConfig dataclass.

    Args:
        config: LoggingConfig instance with logging parameters.
        name: Logger name (default: "fastcsp").

    Returns:
        Configured logging.Logger instance.
    """
    return setup_fastcsp_logger(
        name=name,
        log_file=config.log_file,
        level=config.level,
        console_output=config.console_output,
        append=config.append,
        format_str=config.format,
    )


def ensure_all_modules_use_central_logger() -> None:
    """Configure all FastCSP modules to use the central logger."""
    central_logger = logging.getLogger("fastcsp")

    # All module patterns to redirect
    # Note: submitit is excluded because its DEBUG output contains carriage returns
    # for progress bars which create long garbled lines in log files
    module_patterns = [
        "fastcsp",
        "fairchem.applications.fastcsp",
        "genarris",
    ]

    # Find and configure all matching modules
    for module_name in list(sys.modules.keys()):
        if any(pattern in module_name for pattern in module_patterns):
            try:
                module_logger = logging.getLogger(module_name)
                module_logger.handlers = central_logger.handlers[:]
                module_logger.setLevel(central_logger.level)
                module_logger.propagate = False
            except Exception:
                continue


def print_fastcsp_header(
    logger: logging.Logger, is_restart: bool = False, stages: list[str] | None = None
) -> None:
    """Print FastCSP header."""
    restart_info = (
        f"[RESTART at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        if is_restart
        else ""
    )
    stage_info = f"- Executing stages: {', '.join(stages)}" if stages else ""

    header = f"""╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                              🔬 FastCSP 🔬                                    ║
║ {restart_info:<77}║
║            Fast Crystal Structure Prediction with Universal Models           ║
║                                                                              ║
║   Developers: Vahe Gharakhanyan¹, Anuroop Sriram¹, Luis Barroso-Luque¹       ║
║   Affiliations: ¹ Meta AI (FAIR)                                             ║
║                                                                              ║
║  📖 Publication: "FastCSP: Accelerated Molecular Crystal Structure           ║
║      Prediction with Universal Model for Atoms" (2025)                       ║
║                                                                              ║
║  💡 Key Features:                                                            ║
║     • End-to-end crystal structure prediction workflow                       ║
║     • Integration with Genarris and Universal Model for Atoms (UMA)          ║
║     • High-performance computing with SLURM support                          ║
║     • Scalable from single molecules to large datasets                       ║
║                                                                              ║
║  🌟 "With the pieces visible, predicting organic crystal structures          ║
║      becomes a dance of arrangement—software choreographs the masterpiece."  ║
║                                                                              ║
║  📄 License: MIT License - Copyright (c) Meta Platforms, Inc. & affiliates   ║
║ {stage_info:<77}║
╚══════════════════════════════════════════════════════════════════════════════╝"""

    for line in header.strip().split("\n"):
        logger.info(line)


def log_config_pretty(logger: logging.Logger, config: dict[str, Any]) -> None:
    """Log configuration in readable format."""
    logger.info("=" * 80)
    logger.info("📋 FastCSP CONFIGURATION:")
    try:
        config_json = json.dumps(config, indent=2, default=str)
        for line in config_json.split("\n"):
            logger.info(f"   {line}")
    except Exception:
        logger.info(f"   {config}")
    logger.info("=" * 80)


def log_stage_start(
    logger: logging.Logger, stage_name: str, description: str = ""
) -> None:
    """Log workflow stage start."""
    logger.info(f"Starting {stage_name}...")
    if description:
        logger.info(f"📋 {description}")


def log_stage_complete(
    logger: logging.Logger, stage_name: str, num_jobs: int = 0
) -> None:
    """Log workflow stage completion."""
    suffix = f" with {num_jobs} jobs" if num_jobs > 0 else ""
    logger.info(f"Finished {stage_name}{suffix}.")


def get_central_logger() -> logging.Logger:
    """Get the central FastCSP logger, auto-configure if needed."""
    logger = logging.getLogger("fastcsp")

    if not logger.handlers:
        log_file = Path.cwd() / "FastCSP.log"
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        try:
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    return logger


def detect_restart(root_dir: Path, log_file: str = "FastCSP.log") -> bool:
    """
    Detect if this is a workflow restart by checking for existing log file.

    Args:
        root_dir: Root directory where the log file would be located
        log_file: Name of the log file to check (default: "FastCSP.log")

    Returns:
        bool: True if this appears to be a restart (log file exists with content),
              False for a fresh start
    """
    log_path = root_dir / log_file
    return log_path.exists() and log_path.stat().st_size > 0
