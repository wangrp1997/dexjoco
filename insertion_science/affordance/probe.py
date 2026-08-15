"""Lightweight linear probes (no control policy)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_predict_binary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    C: float = 1.0,
    max_iter: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)
    if len(np.unique(y_train)) < 2:
        # degenerate: predict majority
        pred = np.full_like(y_test, int(np.bincount(y_train).argmax()))
        return {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "degenerate_train": True,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "train_pos_rate": float(y_train.mean()),
            "test_pos_rate": float(y_test.mean()),
        }
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=float(C),
                    max_iter=int(max_iter),
                    random_state=int(seed),
                    class_weight="balanced",
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "degenerate_train": False,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_pos_rate": float(y_train.mean()),
        "test_pos_rate": float(y_test.mean()),
    }
