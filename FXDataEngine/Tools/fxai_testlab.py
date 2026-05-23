#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

OFFLINE_LAB_ROOT = Path(__file__).resolve().parent / "OfflineLab"
if str(OFFLINE_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFLINE_LAB_ROOT))

from testlab import *  # noqa: F401,F403
from testlab.cli import main


if __name__ == "__main__":
    main()
