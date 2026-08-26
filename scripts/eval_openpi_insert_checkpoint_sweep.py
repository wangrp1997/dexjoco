#!/usr/bin/env python3
"""Compare OpenPI checkpoints from varied demo-replayed insertion handoffs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Checkpoint:
    label: str
    path: Path
    step: int | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("/mnt/hdd/dexjoco/checkpoints/pi05_insert_ft_archive/bimanual_assembly_insert_ft/insert_ft_mix_v1"),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("/mnt/hdd/dexjoco/shared_checkpoints/pi05_dexjoco_ckpt/bimanual_assembly"),
    )
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--config-name", default="bimanual_assembly_insert_ft")
    parser.add_argument("--episode-count", type=int, default=30)
    parser.add_argument("--episode-selection-seed", type=int, default=20260824)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-policy-steps", type=int, default=900)
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument("--replan-ratio", type=float, default=0.8)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--xla-memory-fraction", default="0.40")
    parser.add_argument("--server-timeout", type=float, default=900.0)
    parser.add_argument("--output", type=Path, default=Path("/mnt/hdd/dexjoco/outputs/pi05_insert_ft_handoff_eval"))
    parser.add_argument("--openpi-python", type=Path, default=Path("/home/wangrenpeng/miniconda3/envs/openpi/bin/python"))
    parser.add_argument("--dexjoco-python", type=Path, default=Path("/home/wangrenpeng/miniconda3/envs/dexjoco/bin/python"))
    parser.add_argument("--sidecar-dir", type=Path, default=Path("/mnt/hdd/dexjoco/interaction_sidecar/bimanual_assembly"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _valid_checkpoint(path: Path) -> bool:
    return path.is_dir() and (path / "params" / "_METADATA").is_file() and (path / "assets").is_dir()


def _checkpoints(args: argparse.Namespace) -> list[Checkpoint]:
    rows = []
    if not args.no_baseline:
        rows.append(Checkpoint("baseline", args.baseline.resolve(), None))
    if args.checkpoint_root.is_dir():
        for path in args.checkpoint_root.iterdir():
            if path.name.isdigit():
                rows.append(Checkpoint(f"step_{path.name}", path.resolve(), int(path.name)))
    rows.sort(key=lambda row: (-1 if row.step is None else row.step))
    if not rows:
        raise ValueError("no checkpoints discovered")
    if not args.dry_run:
        invalid = [str(row.path) for row in rows if not _valid_checkpoint(row.path)]
        if invalid:
            raise FileNotFoundError("invalid checkpoints:\n" + "\n".join(invalid))
    return rows


def _select_episodes(sidecar_dir: Path, count: int, seed: int) -> list[int]:
    import random

    manifest = json.loads((sidecar_dir / "manifest.json").read_text(encoding="utf-8"))
    candidates = []
    for entry in manifest["episodes"]:
        timing = entry.get("timing", {})
        if timing.get("peg_lift_start") is not None and timing.get("right_grasp_frame") is not None:
            candidates.append(int(entry["episode_index"]))
    random.Random(seed).shuffle(candidates)
    return sorted(candidates[: min(count, len(candidates))])


def _server_command(args: argparse.Namespace, checkpoint: Checkpoint) -> list[str]:
    return [
        str(args.openpi_python),
        "scripts/serve_policy.py",
        f"--port={args.port}",
        "policy:checkpoint",
        f"--policy.config={args.config_name}",
        f"--policy.dir={checkpoint.path}",
    ]


def _eval_command(args: argparse.Namespace, episodes: list[int], output: Path) -> list[str]:
    return [
        str(args.dexjoco_python),
        "scripts/eval_openpi_demo_handoff_insert.py",
        f"--episodes={','.join(map(str, episodes))}",
        f"--seed={args.seed}",
        "--host=127.0.0.1",
        f"--port={args.port}",
        f"--action-horizon={args.action_horizon}",
        f"--replan-ratio={args.replan_ratio}",
        f"--max-policy-steps={args.max_policy_steps}",
        f"--sidecar-dir={args.sidecar_dir.resolve()}",
        f"--output={output}",
        "--overwrite",
    ]


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _wait_server(process: subprocess.Popen[str], port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"policy server exited with code {process.returncode}")
        if _port_open(port):
            return
        time.sleep(1.0)
    raise TimeoutError(f"server did not start within {timeout}s")


def _stop(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.96
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return center - margin, center + margin


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _summarize(checkpoints: list[Checkpoint], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    ranking = []
    outcomes = {}
    for checkpoint, summary in zip(checkpoints, summaries, strict=True):
        evaluable = [row for row in summary["episodes"] if row["setup_ok"]]
        success = {int(row["episode_index"]): bool(row["success"]) for row in evaluable}
        outcomes[checkpoint.label] = success
        successes = sum(success.values())
        total = len(success)
        lower, upper = _wilson(successes, total)
        failure_counts = {
            str(reason): int(count)
            for reason, count in summary.get("failure_counts", {}).items()
        }
        dominant_failure = max(failure_counts, key=failure_counts.get) if failure_counts else ""
        ranking.append(
            {
                "checkpoint": checkpoint.label,
                "path": str(checkpoint.path),
                "step": checkpoint.step,
                "successes": successes,
                "evaluable_episodes": total,
                "setup_failures": int(summary["setup_failures"]),
                "success_rate": successes / max(1, total),
                "wilson95_low": lower,
                "wilson95_high": upper,
                "dominant_failure": dominant_failure,
                "failure_counts": json.dumps(failure_counts, sort_keys=True),
            }
        )
    ranking.sort(key=lambda row: (-row["success_rate"], math.inf if row["step"] is None else row["step"]))
    pairwise = []
    for left_index, left in enumerate(checkpoints):
        for right in checkpoints[left_index + 1 :]:
            common = sorted(set(outcomes[left.label]) & set(outcomes[right.label]))
            left_wins = sum(outcomes[left.label][ep] and not outcomes[right.label][ep] for ep in common)
            right_wins = sum(outcomes[right.label][ep] and not outcomes[left.label][ep] for ep in common)
            pairwise.append(
                {
                    "left": left.label,
                    "right": right.label,
                    "paired_episodes": len(common),
                    "left_wins": left_wins,
                    "right_wins": right_wins,
                    "ties": len(common) - left_wins - right_wins,
                }
            )
    return {"ranking": ranking, "recommended": ranking[0], "pairwise": pairwise}


def main() -> int:
    args = _parse_args()
    checkpoints = _checkpoints(args)
    episodes = _select_episodes(args.sidecar_dir, args.episode_count, args.episode_selection_seed)
    output = args.output.expanduser().resolve()
    print(f"episodes ({len(episodes)}): {episodes}")
    for checkpoint in checkpoints:
        print(f"{checkpoint.label}: {checkpoint.path}")
        if args.dry_run:
            print("  SERVER", " ".join(_server_command(args, checkpoint)))
            print("  EVAL", " ".join(_eval_command(args, episodes, output / checkpoint.label)))
    if args.dry_run:
        return 0
    if _port_open(args.port):
        raise RuntimeError(f"port {args.port} is already occupied")

    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "varied_demo_replay_handoff_insert_checkpoint_sweep",
        "episodes": episodes,
        "episode_selection_seed": args.episode_selection_seed,
        "seed": args.seed,
        "action_horizon": args.action_horizon,
        "replan_ratio": args.replan_ratio,
        "max_policy_steps": args.max_policy_steps,
        "checkpoints": [{"label": row.label, "path": str(row.path), "step": row.step} for row in checkpoints],
    }
    _atomic_json(output / "protocol.json", protocol)
    server_env = os.environ.copy()
    server_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    server_env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.xla_memory_fraction)
    summaries = []

    for checkpoint in checkpoints:
        checkpoint_output = output / checkpoint.label
        rollout_output = checkpoint_output / "rollouts"
        summary_path = rollout_output / "summary.json"
        if summary_path.is_file() and not args.overwrite:
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            print(f"[{checkpoint.label}] reused completed result", flush=True)
            continue
        checkpoint_output.mkdir(parents=True, exist_ok=True)
        print(f"[{checkpoint.label}] starting server", flush=True)
        with (checkpoint_output / "policy_server.log").open("w", encoding="utf-8") as server_log:
            server = subprocess.Popen(
                _server_command(args, checkpoint),
                cwd=_REPO_ROOT / "openpi",
                env=server_env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                _wait_server(server, args.port, args.server_timeout)
                with (checkpoint_output / "eval.log").open("w", encoding="utf-8") as eval_log:
                    completed = subprocess.run(
                        _eval_command(args, episodes, rollout_output),
                        cwd=_REPO_ROOT,
                        stdout=eval_log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                if completed.returncode != 0:
                    raise RuntimeError(f"evaluation failed; see {checkpoint_output / 'eval.log'}")
                summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
                print(
                    f"[{checkpoint.label}] {summaries[-1]['successes']}/"
                    f"{summaries[-1]['episodes_evaluable']}",
                    flush=True,
                )
            finally:
                _stop(server)

    comparison = _summarize(checkpoints, summaries)
    _atomic_json(output / "summary.json", {**protocol, **comparison})
    with (output / "ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison["ranking"][0]))
        writer.writeheader()
        writer.writerows(comparison["ranking"])
    best = comparison["recommended"]
    (output / "BEST_CHECKPOINT.txt").write_text(f"{best['checkpoint']}\n{best['path']}\n", encoding="utf-8")
    print(f"recommended: {best['checkpoint']} {best['path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
