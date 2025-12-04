#!/usr/bin/env python3
"""Alias for converting reports to concept CUIs."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concept_extraction.convert_reports import main


if __name__ == "__main__":
    main()
