#!/usr/bin/env python3
"""Evaluation entrypoint for MedCLIP components."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _dispatch(
    func_loader: Callable[[], Callable[[Optional[List[str]]], None]],
    argv: List[str],
    extra_args: Optional[Sequence[str]] = None,
) -> None:
    func = func_loader()
    forward = list(argv or [])
    if extra_args:
        forward.extend(extra_args)
    func(forward)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluation entrypoint for MedCLIP components.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    commands: Dict[str, Dict[str, object]] = {
        "cbm": {
            "loader": lambda: __import__("src.training.trainers.cbm", fromlist=["eval_main"]).eval_main,
            "extra": [],
        },
        "concept-classifier": {
            "loader": lambda: __import__("src.training.trainers.feature_classifier", fromlist=["eval_main"]).eval_main,
            "extra": ["--mode", "concept"],
        },
        "blackbox-classifier": {
            "loader": lambda: __import__("src.training.trainers.feature_classifier", fromlist=["eval_main"]).eval_main,
            "extra": ["--mode", "label"],
        },
    }
    help_text = {
        "cbm": "Evaluate the concept bottleneck model.",
        "concept-classifier": "Evaluate a concept classifier checkpoint.",
        "blackbox-classifier": "Evaluate the direct label classifier.",
    }

    for name, meta in commands.items():
        sub = subparsers.add_parser(name, help=help_text.get(name, ""), add_help=False)
        sub.add_argument(
            "args",
            nargs=argparse.REMAINDER,
            help="Arguments forwarded to the underlying evaluator; use `--` before them if needed.",
        )

    parsed = parser.parse_args(argv)
    forward_args = parsed.args if hasattr(parsed, "args") else []
    meta = commands[parsed.command]
    _dispatch(meta["loader"], forward_args, extra_args=meta.get("extra"))


if __name__ == "__main__":
    main(sys.argv[1:])
