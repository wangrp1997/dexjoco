"""Evaluate the non-overridable P1 route gate and persist the decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retrieval_cerebellum.methodology_gate import (
    decide_methodology_route_from_file,
    write_methodology_decision,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    decision = decide_methodology_route_from_file(args.visual_summary)
    write_methodology_decision(args.output, decision)
    print(json.dumps(decision.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
