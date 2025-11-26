#!/usr/bin/env python3
"""Evaluation entrypoint for MedCLIP components."""
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
        description="Evaluation entrypoint for MedCLIP components.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    commands: Dict[str, Callable[[], Callable[[Optional[List[str]]], None]]] = {
        "cbm": lambda: __import__("src.trainers.cbm", fromlist=["eval_main"]).eval_main,
        "concept-classifier": lambda: __import__("src.trainers.concept_classifier", fromlist=["eval_main"]).eval_main,
        "blackbox-classifier": lambda: __import__("src.trainers.blackbox_classifier", fromlist=["eval_main"]).eval_main,
    }
    help_text = {
        "cbm": "Evaluate the concept bottleneck model.",
        "concept-classifier": "Evaluate a concept classifier checkpoint.",
        "blackbox-classifier": "Evaluate the direct label classifier.",
    }

    for name in commands:
        sub = subparsers.add_parser(name, help=help_text.get(name, ""), add_help=False)
        sub.add_argument(
            "args",
            nargs=argparse.REMAINDER,
            help="Arguments forwarded to the underlying evaluator; use `--` before them if needed.",
        )

    parsed = parser.parse_args(argv)
    forward_args = parsed.args if hasattr(parsed, "args") else []
    _dispatch(commands[parsed.command], forward_args)


if __name__ == "__main__":
    main(sys.argv[1:])
