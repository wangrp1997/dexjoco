#!/usr/bin/env python3
"""Fairly compare multiple OpenPI checkpoints on identical DexJoCo rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SUCCESS_RE = re.compile(r"success_rate_(\d+)_(\d+)\.txt$")


@dataclass(frozen=True)
class Checkpoint:
    label: str
    path: Path
    step: int | None


def _csv_ints(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _checkpoint_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    return label.strip(), Path(raw_path.strip()).expanduser()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("/mnt/hdd/dexjoco/checkpoints/pi05_insert_ft_archive/bimanual_assembly_insert_ft/insert_ft_mix_v1"),
        help="Directory containing numeric checkpoint subdirectories.",
    )
    parser.add_argument("--checkpoint", action="append", type=_checkpoint_arg, default=[])
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("/mnt/hdd/dexjoco/shared_checkpoints/pi05_dexjoco_ckpt/bimanual_assembly"),
        help="Original assembly checkpoint; pass an empty string to disable via --no-baseline.",
    )
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--config-name", default="bimanual_assembly_insert_ft")
    parser.add_argument("--eval-config", type=Path, default=_REPO_ROOT / "configs/rand_obj/bimanual_assembly.yaml")
    parser.add_argument("--seeds", type=_csv_ints, default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument("--replan-ratio", type=float, default=0.8)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--xla-memory-fraction", default="0.85")
    parser.add_argument("--server-timeout", type=float, default=900.0)
    parser.add_argument("--eval-timeout", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("/mnt/hdd/dexjoco/outputs/pi05_insert_ft_checkpoint_eval"))
    parser.add_argument("--openpi-python", type=Path, default=Path("/home/wangrenpeng/miniconda3/envs/openpi/bin/python"))
    parser.add_argument("--dexjoco-python", type=Path, default=Path("/home/wangrenpeng/miniconda3/envs/dexjoco/bin/python"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _valid_policy_checkpoint(path: Path) -> bool:
    return path.is_dir() and (path / "params" / "_METADATA").is_file() and (path / "assets").is_dir()


def _discover_checkpoints(args: argparse.Namespace) -> list[Checkpoint]:
    checkpoints: list[Checkpoint] = []
    if not args.no_baseline:
        checkpoints.append(Checkpoint("baseline", args.baseline.expanduser().resolve(), None))
    root = args.checkpoint_root.expanduser().resolve()
    if root.is_dir():
        for path in root.iterdir():
            if path.name.isdigit():
                checkpoints.append(Checkpoint(f"step_{path.name}", path.resolve(), int(path.name)))
    for label, path in args.checkpoint:
        step_match = re.search(r"(\d+)$", label)
        checkpoints.append(Checkpoint(label, path.resolve(), int(step_match.group(1)) if step_match else None))

    unique: dict[str, Checkpoint] = {}
    for checkpoint in checkpoints:
        unique[checkpoint.label] = checkpoint
    ordered = sorted(unique.values(), key=lambda item: (-1 if item.step is None else item.step, item.label))
    if not ordered:
        raise ValueError("no checkpoints discovered")
    if not args.dry_run:
        invalid = [str(item.path) for item in ordered if not _valid_policy_checkpoint(item.path)]
        if invalid:
            raise FileNotFoundError("invalid policy checkpoints:\n" + "\n".join(invalid))
    return ordered


def _command_text(command: list[str]) -> str:
    return shlex.join(command)


def _server_command(args: argparse.Namespace, checkpoint: Checkpoint) -> list[str]:
    return [
        str(args.openpi_python),
        "scripts/serve_policy.py",
        f"--port={args.port}",
        "policy:checkpoint",
        f"--policy.config={args.config_name}",
        f"--policy.dir={checkpoint.path}",
    ]


def _eval_command(args: argparse.Namespace, checkpoint: Checkpoint, seed: int, output: Path) -> list[str]:
    return [
        str(args.dexjoco_python),
        "-m",
        "dexjoco_openpi_client.cli.evaluate",
        f"--config={args.eval_config.resolve()}",
        f"--seed={seed}",
        "--host=127.0.0.1",
        f"--port={args.port}",
        f"--output={output}",
        f"--checkpoint={checkpoint.path}",
        f"--episodes={args.episodes}",
        f"--action-horizon={args.action_horizon}",
        f"--replan-ratio={args.replan_ratio}",
        "--render-mode=rgb_array",
        "--overwrite",
    ]


def _wait_for_server(process: subprocess.Popen[str], port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"policy server exited with code {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return
        except OSError:
            time.sleep(1.0)
    raise TimeoutError(f"policy server did not become ready on port {port} within {timeout:.0f}s")


def _port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _parse_rollout(output: Path) -> dict[str, Any]:
    markers = []
    for path in output.glob("success_rate_*_*.txt"):
        match = _SUCCESS_RE.match(path.name)
        if match:
            markers.append((int(match.group(1)), int(match.group(2))))
    if len(markers) != 1:
        raise RuntimeError(f"expected one success marker in {output}, found {markers}")
    successes, total = markers[0]
    outcomes: list[bool] = []
    for episode in range(total):
        matches = list(output.glob(f"episode_{episode:02d}_*"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one result directory for episode {episode} in {output}")
        outcomes.append(matches[0].name.endswith("_success"))
    if sum(outcomes) != successes:
        raise RuntimeError(f"success marker disagrees with episode directories in {output}")
    diagnostic_path = output / "evaluation_summary.json"
    if not diagnostic_path.is_file():
        raise RuntimeError(f"missing structured evaluation summary: {diagnostic_path}")
    diagnostics = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    return {
        "successes": successes,
        "episodes": total,
        "outcomes": outcomes,
        "failure_counts": diagnostics["failure_counts"],
        "episode_diagnostics": diagnostics["episode_diagnostics"],
    }


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _summarize(checkpoints: list[Checkpoint], runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for checkpoint in checkpoints:
        selected = [run for run in runs if run["checkpoint"] == checkpoint.label]
        successes = sum(run["successes"] for run in selected)
        total = sum(run["episodes"] for run in selected)
        seed_rates = [run["successes"] / run["episodes"] for run in selected]
        lower, upper = _wilson(successes, total)
        failure_counts: dict[str, int] = {}
        for run in selected:
            for reason, count in run.get("failure_counts", {}).items():
                failure_counts[reason] = failure_counts.get(reason, 0) + int(count)
        dominant_failure = max(failure_counts, key=failure_counts.get) if failure_counts else ""
        rows.append(
            {
                "checkpoint": checkpoint.label,
                "path": str(checkpoint.path),
                "step": checkpoint.step,
                "successes": successes,
                "episodes": total,
                "success_rate": successes / total,
                "worst_seed_rate": min(seed_rates),
                "best_seed_rate": max(seed_rates),
                "wilson95_low": lower,
                "wilson95_high": upper,
                "dominant_failure": dominant_failure,
                "failure_counts": json.dumps(failure_counts, sort_keys=True),
            }
        )
    rows.sort(
        key=lambda row: (
            -row["success_rate"],
            -row["worst_seed_rate"],
            math.inf if row["step"] is None else row["step"],
            row["checkpoint"],
        )
    )

    pairwise = []
    by_checkpoint = {
        checkpoint.label: {
            (run["seed"], episode): outcome
            for run in runs
            if run["checkpoint"] == checkpoint.label
            for episode, outcome in enumerate(run["outcomes"])
        }
        for checkpoint in checkpoints
    }
    for left_index, left in enumerate(checkpoints):
        for right in checkpoints[left_index + 1 :]:
            common = sorted(set(by_checkpoint[left.label]) & set(by_checkpoint[right.label]))
            left_only = sum(by_checkpoint[left.label][key] and not by_checkpoint[right.label][key] for key in common)
            right_only = sum(by_checkpoint[right.label][key] and not by_checkpoint[left.label][key] for key in common)
            pairwise.append(
                {
                    "left": left.label,
                    "right": right.label,
                    "paired_episodes": len(common),
                    "left_wins": left_only,
                    "right_wins": right_only,
                    "ties": len(common) - left_only - right_only,
                }
            )
    return {"ranking": rows, "recommended": rows[0], "pairwise": pairwise}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    checkpoints = _discover_checkpoints(args)
    output_root = args.output.expanduser().resolve()

    print("checkpoints:")
    for checkpoint in checkpoints:
        print(f"  {checkpoint.label}: {checkpoint.path}")
    print(f"paired protocol: seeds={args.seeds}, episodes_per_seed={args.episodes}")

    if args.dry_run:
        for checkpoint in checkpoints:
            print("SERVER", _command_text(_server_command(args, checkpoint)))
            for seed in args.seeds:
                run_output = output_root / checkpoint.label / f"seed_{seed}"
                print("EVAL  ", _command_text(_eval_command(args, checkpoint, seed, run_output)))
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    run_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seeds": args.seeds,
        "episodes_per_seed": args.episodes,
        "action_horizon": args.action_horizon,
        "replan_ratio": args.replan_ratio,
        "eval_config": str(args.eval_config.resolve()),
        "checkpoints": [{"label": item.label, "path": str(item.path), "step": item.step} for item in checkpoints],
    }
    protocol_path = output_root / "protocol.json"
    if protocol_path.is_file() and not args.overwrite:
        existing_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        comparable_keys = (
            "seeds",
            "episodes_per_seed",
            "action_horizon",
            "replan_ratio",
            "eval_config",
            "checkpoints",
        )
        if any(existing_protocol.get(key) != run_manifest.get(key) for key in comparable_keys):
            raise RuntimeError(
                f"existing results use a different protocol: {protocol_path}; "
                "choose another --output or pass --overwrite"
            )
    _atomic_json(protocol_path, run_manifest)

    server_env = os.environ.copy()
    server_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    server_env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.xla_memory_fraction)
    eval_env = os.environ.copy()
    for proxy in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        eval_env.pop(proxy, None)

    for checkpoint in checkpoints:
        checkpoint_output = output_root / checkpoint.label
        checkpoint_output.mkdir(parents=True, exist_ok=True)
        server_log_path = checkpoint_output / "policy_server.log"
        server_command = _server_command(args, checkpoint)
        print(f"\n[{checkpoint.label}] starting policy server", flush=True)
        if _port_is_open(args.port):
            raise RuntimeError(
                f"port {args.port} is already occupied; stop the old policy server or choose another --port"
            )
        with server_log_path.open("a", encoding="utf-8") as server_log:
            server = subprocess.Popen(
                server_command,
                cwd=_REPO_ROOT / "openpi",
                env=server_env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                _wait_for_server(server, args.port, args.server_timeout)
                for seed in args.seeds:
                    run_output = checkpoint_output / f"seed_{seed}"
                    result_path = run_output / "result.json"
                    if result_path.is_file() and not args.overwrite:
                        result = json.loads(result_path.read_text(encoding="utf-8"))
                        runs.append(result)
                        print(f"[{checkpoint.label}] seed={seed}: reused {result['successes']}/{result['episodes']}")
                        continue
                    eval_log_path = checkpoint_output / f"seed_{seed}.log"
                    command = _eval_command(args, checkpoint, seed, run_output)
                    print(f"[{checkpoint.label}] seed={seed}: evaluating", flush=True)
                    with eval_log_path.open("w", encoding="utf-8") as eval_log:
                        completed = subprocess.run(
                            command,
                            cwd=_REPO_ROOT / "dexjoco",
                            env=eval_env,
                            stdout=eval_log,
                            stderr=subprocess.STDOUT,
                            text=True,
                            timeout=args.eval_timeout,
                        )
                    if completed.returncode != 0:
                        raise RuntimeError(f"evaluation failed; see {eval_log_path}")
                    parsed = _parse_rollout(run_output)
                    result = {
                        "checkpoint": checkpoint.label,
                        "checkpoint_path": str(checkpoint.path),
                        "step": checkpoint.step,
                        "seed": seed,
                        **parsed,
                    }
                    _atomic_json(result_path, result)
                    runs.append(result)
                    print(f"[{checkpoint.label}] seed={seed}: {parsed['successes']}/{parsed['episodes']}", flush=True)
            finally:
                _stop_process(server)

    summary = _summarize(checkpoints, runs)
    _atomic_json(output_root / "summary.json", {**run_manifest, **summary, "runs": runs})
    _write_csv(output_root / "ranking.csv", summary["ranking"])
    recommended = summary["recommended"]
    (output_root / "BEST_CHECKPOINT.txt").write_text(
        f"{recommended['checkpoint']}\n{recommended['path']}\n",
        encoding="utf-8",
    )

    print("\nranking:")
    for index, row in enumerate(summary["ranking"], start=1):
        print(
            f"  {index}. {row['checkpoint']}: {row['successes']}/{row['episodes']} "
            f"({100.0 * row['success_rate']:.1f}%, 95% CI {100.0 * row['wilson95_low']:.1f}-"
            f"{100.0 * row['wilson95_high']:.1f}%)"
        )
    print(f"recommended: {summary['recommended']['path']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
