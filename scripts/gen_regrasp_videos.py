#!/usr/bin/env python3
"""Generate short-named regrasp videos: N drop + N grasp_fail.

Output layout:
  /mnt/ssd/datasets/dexjoco_gendata/bimanual_assembly/videos/
    ep010_drop.mp4
    ep010_grasp_fail.mp4
    ...
    summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO), str(_REPO / "dexjoco")]

from dexjoco_datagen.paths import DEFAULT_MANIFEST, video_dir
from dexjoco_datagen.regrasp_pipeline import load_manifest, pick_entry, run_perturb_regrasp_one

MODES = ("drop", "grasp_fail")


def _good_eps(manifest: dict, n: int) -> list[int]:
    out: list[int] = []
    for e in manifest["episodes"]:
        t = e["timing"]
        if (
            t.get("right_grasp_frame") is not None
            and t.get("peg_lift_start") is not None
            and not t.get("right_grasp_fallback", False)
            and int(e["episode_index"]) >= 10
        ):
            out.append(int(e["episode_index"]))
        if len(out) >= n:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="episodes per mode")
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/mnt/ssd/datasets/dexjoco_gendata"),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="delete everything in the videos folder first",
    )
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    manifest = load_manifest(args.manifest)
    eps = args.episodes or _good_eps(manifest, args.n)
    out_dir = video_dir("bimanual_assembly", args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for p in out_dir.iterdir():
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
        print(f"cleaned {out_dir}", flush=True)

    rows: list[dict] = []
    print(f"episodes={eps} modes={MODES} out={out_dir}", flush=True)

    for ep in eps:
        entry = pick_entry(manifest, ep)
        for mode in MODES:
            video_path = out_dir / f"ep{ep:03d}_{mode}.mp4"
            meta_path = out_dir / f"ep{ep:03d}_{mode}.json"
            print(f"\n=== ep{ep} {mode} -> {video_path.name} ===", flush=True)
            try:
                result = run_perturb_regrasp_one(
                    entry,
                    video_path=video_path,
                    mode=mode,
                    record_video=True,
                )
                row = {
                    "episode_index": result.episode_index,
                    "mode": result.mode,
                    "success": result.success,
                    "message": result.message,
                    "diagnostics": result.diagnostics,
                    "video": str(video_path),
                    "error": None,
                }
            except Exception as ex:  # noqa: BLE001
                row = {
                    "episode_index": ep,
                    "mode": mode,
                    "success": False,
                    "message": str(ex),
                    "diagnostics": {"fail_reason": "exception"},
                    "video": str(video_path),
                    "error": repr(ex),
                }
                print(f"[error] ep{ep} {mode}: {ex}", flush=True)
            rows.append(row)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(row, f, indent=2, ensure_ascii=False)

    by_mode: dict[str, dict] = {}
    for mode in MODES:
        sub = [r for r in rows if r["mode"] == mode]
        reasons = Counter(
            (r.get("diagnostics") or {}).get("fail_reason", "unknown") for r in sub
        )
        by_mode[mode] = {
            "n": len(sub),
            "n_ok": sum(1 for r in sub if r["success"]),
            "n_fail": sum(1 for r in sub if not r["success"]),
            "reasons": dict(reasons),
            "rows": sub,
        }

    summary = {"episodes": eps, "by_mode": by_mode}
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n======== SUMMARY ========", flush=True)
    for mode, info in by_mode.items():
        print(
            f"{mode}: {info['n_ok']}/{info['n']} OK  reasons={info['reasons']}",
            flush=True,
        )
        for r in info["rows"]:
            d = r.get("diagnostics") or {}
            tag = "OK" if r["success"] else "FAIL"
            print(
                f"  ep{r['episode_index']:03d}_{mode}.mp4  {tag:4s}  "
                f"reason={d.get('fail_reason')}  "
                f"oriΔ={d.get('ori_delta_deg')}  "
                f"axisΔ={d.get('axis_delta_deg')}  "
                f"dz={d.get('delta_z_m')}  "
                f"tray_xy={d.get('peg_tray_xy_m')}",
                flush=True,
            )
    print(f"summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
