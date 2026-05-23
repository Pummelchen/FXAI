from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OFFLINE_LAB_ROOT = ROOT / "OfflineLab"
if str(OFFLINE_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFLINE_LAB_ROOT))
