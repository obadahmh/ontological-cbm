#!/usr/bin/env python3
"""Training entrypoint for MedCLIP components."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _dispatch(func_loader: Callable[[], Callable[[Optional[List[str]]], None]], argv: List[str]) -> None:
    func = func_loader()
    func(argv)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Training entrypoint for MedCLIP components.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    commands: Dict[str, Callable[[], Callable[[Optional[List[str]]], None]]] = {
        "cbm": lambda: __import__("src.trainers.cbm", fromlist=["train_main"]).train_main,
        "concept-classifier": lambda: __import__("src.trainers.concept_classifier", fromlist=["train_main"]).train_main,
        "blackbox-classifier": lambda: __import__("src.trainers.blackbox_classifier", fromlist=["train_main"]).train_main,
    }
    help_text = {
        "cbm": "Train the concept bottleneck model.",
        "concept-classifier": "Train the concept classifier.",
        "blackbox-classifier": "Train the direct label classifier (baseline).",
    }

    for name in commands:
        sub = subparsers.add_parser(name, help=help_text.get(name, ""), add_help=False)
        sub.add_argument(
            "args",
            nargs=argparse.REMAINDER,
            help="Arguments forwarded to the underlying trainer; use `--` before them if needed.",
        )

    parsed = parser.parse_args(argv)
    forward_args = parsed.args if hasattr(parsed, "args") else []
    # Allow using a standalone "--" after the subcommand to stop top-level parsing.
    # Argparse stores that sentinel as the first token; strip it before forwarding.
    if forward_args and forward_args[0] == "--":
        forward_args = forward_args[1:]
    _dispatch(commands[parsed.command], forward_args)


if __name__ == "__main__":
    main(sys.argv[1:])
