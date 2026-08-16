"""Anisotropic recoverability boundary from Demo Handoff Perturb Recoverability rows."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _rate(flags: list[bool]) -> float:
    if not flags:
        return float("nan")
    return float(sum(1 for x in flags if x) / len(flags))


def build_cell_rates(
    rows: list[dict[str, Any]],
    *,
    min_n_held_out: int = 2,
) -> dict[str, dict[str, Any]]:
    """Return map pert_name -> scale_str -> {held_out, pooled, n_*, inside candidate fields}."""
    held: dict[tuple[str, float], list[bool]] = defaultdict(list)
    pool: dict[tuple[str, float], list[bool]] = defaultdict(list)
    kind_of: dict[str, str] = {}
    for r in rows:
        name = str(r.get("pert_name") or r.get("kind"))
        kind = str(r.get("kind", "unknown"))
        kind_of[name] = kind
        if kind in ("none", "identity"):
            continue
        scale = float(r["scale"])
        ok = bool(r.get("insert_ok"))
        pool[(name, scale)].append(ok)
        if str(r.get("split")) == "held_out":
            held[(name, scale)].append(ok)

    cells: dict[str, dict[str, Any]] = {}
    for (name, scale), flags in sorted(pool.items()):
        hflags = held.get((name, scale), [])
        use_held = len(hflags) >= int(min_n_held_out)
        rate = _rate(hflags) if use_held else _rate(flags)
        cells.setdefault(name, {})
        cells[name][str(scale)] = {
            "scale": scale,
            "kind": kind_of.get(name, "unknown"),
            "rate": rate,
            "rate_source": "held_out" if use_held else "pooled",
            "n_held_out": len(hflags),
            "n_pooled": len(flags),
            "held_out_rate": _rate(hflags) if hflags else float("nan"),
            "pooled_rate": _rate(flags),
        }
    return cells


def build_direction_boundaries(
    cells: dict[str, dict[str, Any]],
    *,
    inside_rate_min: float,
) -> dict[str, dict[str, Any]]:
    """Per pert_name: max scale with rate >= inside_rate_min (0 if none)."""
    out: dict[str, dict[str, Any]] = {}
    for name, scales in cells.items():
        ok_scales = []
        for _sk, cell in scales.items():
            if float(cell["rate"]) >= float(inside_rate_min):
                ok_scales.append(float(cell["scale"]))
        max_ok = max(ok_scales) if ok_scales else 0.0
        kind = next(iter(scales.values()))["kind"]
        out[name] = {
            "kind": kind,
            "max_recoverable_scale": max_ok,
            "inside_rate_min": float(inside_rate_min),
            "cells": scales,
        }
    return out


def kind_boundary_summary(boundaries: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate max recoverable scale per kind (min across directions = fragile envelope)."""
    by_kind: dict[str, list[float]] = defaultdict(list)
    for name, b in boundaries.items():
        by_kind[str(b["kind"])].append(float(b["max_recoverable_scale"]))
    out = {}
    for kind, vals in sorted(by_kind.items()):
        out[kind] = {
            "max_among_dirs": max(vals) if vals else 0.0,
            "min_among_dirs": min(vals) if vals else 0.0,
            "n_dirs": len(vals),
        }
    return out
