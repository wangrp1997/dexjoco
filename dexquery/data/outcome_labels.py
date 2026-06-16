"""Load DexQuery sidecar outcome labels aligned to LeRobot frame indices."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_outcome_labels(label_dir: Path) -> dict[str, np.ndarray]:
    """Load ``outcomes.parquet`` written by ``scripts/label_contact.py``."""
    label_dir = label_dir.expanduser()
    path = label_dir / "outcomes.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing outcome labels: {path}")

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("load_outcome_labels requires pyarrow.") from exc

    table = pq.read_table(path)
    return {name: table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names}


def build_outcome_index_map(labels: dict[str, np.ndarray]) -> dict[int, int]:
    """Map LeRobot global frame ``index`` -> row in ``outcomes.parquet``."""
    indices = labels["index"]
    return {int(idx): row for row, idx in enumerate(indices.tolist())}


def default_outcome_label_dir(dataset_root: Path) -> Path:
    return dataset_root / "dexquery_labels"
