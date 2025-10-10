"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

FastCSP Centralized Logging System

Key Features:
- Centralized logger configuration across all FastCSP modules
- Structured logging for workflow stages with progress indicators
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def setup_fastcsp_logger(
    name: str = "fastcsp",
    log_file: str | Path | None = None,
    level: str = "INFO",
    console_output: bool = True,
    append: bool = True,
) -> logging.Logger:
    """
    Set up the centralized FastCSP logger with configurable file and console handlers.

    Args:
        name: Logger name identifier (default: "fastcsp")
        log_file: Path to log file for persistent logging (None disables file logging)
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        console_output: Enable console output for interactive monitoring
        append: Append to existing log file (True) or overwrite (False)

    Returns:
        logging.Logger: Configured logger instance ready for use

    Notes:
        - This function should be called once at workflow initialization
        - All FastCSP modules should use get_central_logger() after setup
        - Append mode (default) supports workflow restarts and debugging
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_mode = "a" if append else "w"
        file_handler = logging.FileHandler(log_path, mode=file_mode)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def print_fastcsp_header(
    logger: logging.Logger, is_restart: bool = False, stages: list[str] | None = None
) -> None:
    """Print FastCSP header with project information."""
    restart_info = ""
    if is_restart:
        restart_info = f"[RESTART at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"

    stage_info = ""
    if stages:
        stage_info = f"- Executing stages: {', '.join(stages)}"

    header = f"""╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                              🔬 FastCSP 🔬                                    ║
║ {restart_info:<77}║
║            Fast Crystal Structure Prediction with Universal Models           ║
║                                                                              ║
║   Developers: Vahe Gharakhanyan¹, Anuroop Sriram¹                            ║
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
    """Log configuration in a readable format."""
    logger.info("=" * 80)
    logger.info("📋 FastCSP CONFIGURATION:")
    logger.info("=" * 80)

    try:
        # Pretty print the configuration as formatted JSON
        config_json = json.dumps(config, indent=2, default=str, separators=(",", ": "))
        for line in config_json.split("\n"):
            logger.info(f"   {line}")
    except Exception as e:
        logger.warning(f"Could not serialize config to JSON: {e}")
        # Fallback to basic string representation
        logger.info(f"   {config}")

    logger.info("=" * 80)


def get_fastcsp_logger(
    config: dict[str, Any] | None = None, root_dir: str | Path | None = None
) -> logging.Logger:
    """Get or create FastCSP logger with configuration."""
    # Get logging configuration
    log_config = {}
    if config and "logging" in config:
        log_config = config["logging"]

    # Determine log file path and name
    log_file = log_config.get("log_file", "fastcsp.log")

    if root_dir:
        log_file_path = Path(root_dir) / log_file
    elif config and "root" in config:
        log_file_path = Path(config["root"]) / log_file
    else:
        log_file_path = Path(log_file)  # Use current directory as fallback

    return setup_fastcsp_logger(
        name="fastcsp",
        log_file=log_file_path,
        level=log_config.get("level", "INFO"),
        console_output=log_config.get("console", True),
        append=True,
    )


def ensure_all_modules_use_central_logger() -> None:
    """Configure all FastCSP modules to use the central logger."""
    central_logger = logging.getLogger("fastcsp")

    # List of module names that should use central logging
    module_loggers = [
        "fastcsp.generate",
        "fastcsp.relax",
        "fastcsp.filter",
        "fastcsp.process_generated",
        "fastcsp.eval",
        "genarris",
        "submitit",
    ]

    # Redirect all module loggers to use the central logger's handlers
    for module_name in module_loggers:
        module_logger = logging.getLogger(module_name)
        module_logger.handlers = central_logger.handlers[:]  # Copy handlers
        module_logger.setLevel(central_logger.level)
        module_logger.propagate = False  # Prevent duplicate logging


def log_stage_start(
    logger: logging.Logger, stage_name: str, description: str = ""
) -> None:
    """Log the start of a workflow stage."""
    logger.info(f"Starting {stage_name}...")
    if description:
        logger.info(f"📋 {description}")


def log_stage_complete(
    logger: logging.Logger, stage_name: str, num_jobs: int = 0
) -> None:
    """Log the completion of a workflow stage."""
    if num_jobs > 0:
        logger.info(f"Finished {stage_name} with {num_jobs} jobs.")
    else:
        logger.info(f"Finished {stage_name}.")


def log_error(logger: logging.Logger, error: Exception, context: str = "") -> None:
    """Log error information in a standardized format."""
    import traceback

    logger.error("=" * 80)
    logger.error(f"❌ ERROR{f' in {context}' if context else ''}: {error}")
    logger.error(f"❌ Error type: {type(error).__name__}")

    # Log traceback with proper formatting
    tb_lines = traceback.format_exc().strip().split("\n")
    for line in tb_lines:
        logger.error(f"   {line}")
    logger.error("=" * 80)


def get_central_logger() -> logging.Logger:
    """
    Get the central FastCSP logger instance.
    """
    return logging.getLogger("fastcsp")
