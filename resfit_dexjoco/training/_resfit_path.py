"""Add ResFiT third_party to ``sys.path`` for shared utilities."""

from __future__ import annotations

import sys
from pathlib import Path

_DEXJOco_ROOT = Path(__file__).resolve().parents[2]
_RESFIT_ROOT = _DEXJOco_ROOT / "third_party" / "residual-offpolicy-rl"
_DEXJOco_PKG = _DEXJOco_ROOT / "dexjoco"

for path in (_RESFIT_ROOT, _DEXJOco_PKG, _DEXJOco_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

RESFIT_ROOT = _RESFIT_ROOT
DEXJOco_ROOT = _DEXJOco_ROOT
