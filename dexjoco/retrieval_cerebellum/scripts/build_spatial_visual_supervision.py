"""Replay episodes and export teacher-only spatial visual supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as parquet

from dexjoco.tasks import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import restore_initial_state
from dexquery.data.action_utils import policy_dual_arm_to_raw, rotvec_dual_arm_to_policy
from dexquery.data.episode_replay import make_assembly_env
from dexquery.data.zarr_io import discover_zarr_demos, load_zarr_episode
from retrieval_cerebellum.privileged import PrivilegedAssemblyPrimitiveProvider
from retrieval_cerebellum.spatial_visual_supervision import (
    CAMERA_KEYS,
    KEYPOINT_NAMES,
    CameraCalibration,
    assembly_keypoints_world,
    keypoint_visibility,
    resize_semantic_mask,
    semantic_mask_from_segmentation,
)
from retrieval_cerebellum.visual_initialization import DEFAULT_CAMERA_KEYS


DEFAULT_DATASET = Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly")
DEFAULT_ZARR = Path("/mnt/ssd/datasets/dexjoco_raw/dexjoco_raw_datasets/bimanual_assembly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--zarr-input-dir", type=Path, default=DEFAULT_ZARR)
    parser.add_argument("--estimation-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--maximum-rows-per-episode", type=int, default=None)
    parser.add_argument("--mask-downsample", type=int, default=4)
    parser.add_argument("--axis-length-m", type=float, default=0.02)
    parser.add_argument("--support-radius-px", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _camera_ids(raw_env) -> tuple[int, int, int]:
    return (
        int(raw_env._front_camera_id),
        int(raw_env._wrist_left_camera_id),
        int(raw_env._wrist_right_camera_id),
    )


def _camera_calibration(raw_env, camera_id: int, width: int, height: int) -> CameraCalibration:
    return CameraCalibration.from_vertical_fov(
        width=width,
        height=height,
        vertical_fov_degrees=float(raw_env._model.cam_fovy[camera_id]),
        position_world=np.asarray(raw_env._data.cam_xpos[camera_id]),
        rotation_world_from_camera=np.asarray(raw_env._data.cam_xmat[camera_id]).reshape(3, 3),
    )


def _video_image_size(dataset_root: Path) -> tuple[int, int]:
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    sizes = set()
    for camera_key in DEFAULT_CAMERA_KEYS:
        feature = info["features"][camera_key]
        sizes.add(
            (
                int(feature["info"]["video.height"]),
                int(feature["info"]["video.width"]),
            )
        )
    if len(sizes) != 1:
        raise ValueError(f"spatial supervision requires equal camera sizes, got {sizes}")
    return next(iter(sizes))


def _body_geom_ids(model, body_id: int) -> frozenset[int]:
    return frozenset(
        int(geom_id)
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == int(body_id)
    )


def _episode_sources(
    estimation_dir: Path,
    *,
    requested: set[int] | None,
    frame_stride: int,
    maximum_rows: int | None,
) -> list[tuple[int, str, np.ndarray]]:
    sources: list[tuple[int, str, np.ndarray]] = []
    for path in sorted((estimation_dir / "episodes").glob("episode_*.parquet")):
        episode_index = int(path.stem.rsplit("_", 1)[-1])
        if requested is not None and episode_index not in requested:
            continue
        table = parquet.read_table(path, columns=["split", "frame_index"])
        split = str(table["split"][0].as_py())
        frames = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)[::frame_stride]
        if maximum_rows is not None:
            frames = frames[:maximum_rows]
        if frames.size:
            sources.append((episode_index, split, frames))
    if not sources:
        raise FileNotFoundError("no estimation episode shards selected")
    return sources


def _write_episode(
    *,
    episode_index: int,
    split: str,
    selected_frames: np.ndarray,
    actions44: np.ndarray,
    initial_state: np.ndarray,
    output_path: Path,
    video_image_size: tuple[int, int],
    mask_downsample: int,
    axis_length_m: float,
    support_radius_px: int,
) -> dict[str, object]:
    env = make_assembly_env(seed=episode_index, randomize=False)
    raw_env = env.unwrapped
    raw_env.hz = 0
    renderer = None
    try:
        env.reset()
        restore_initial_state(
            env,
            "bimanual_assembly",
            CONFIG_MAPPING["bimanual_assembly"](),
            initial_state,
        )
        mujoco.mj_forward(raw_env._model, raw_env._data)
        provider = PrivilegedAssemblyPrimitiveProvider(raw_env)
        renderer = mujoco.Renderer(raw_env._model)
        renderer.enable_segmentation_rendering()
        render_width = int(renderer.width)
        render_height = int(renderer.height)
        image_height, image_width = video_image_size
        camera_ids = _camera_ids(raw_env)
        peg_body_id = int(raw_env._model.body(provider.names.peg_body).id)
        peg_geom_ids = _body_geom_ids(raw_env._model, peg_body_id)
        socket_body_id = int(raw_env._model.body(provider.names.socket_body).id)
        socket_geom_ids = _body_geom_ids(raw_env._model, socket_body_id)
        selected_lookup = {int(frame): row for row, frame in enumerate(selected_frames)}
        row_count = len(selected_frames)
        camera_count = len(camera_ids)
        keypoint_count = len(KEYPOINT_NAMES)
        mask_height = (image_height + mask_downsample - 1) // mask_downsample
        mask_width = (image_width + mask_downsample - 1) // mask_downsample
        keypoints_world = np.empty((row_count, keypoint_count, 3), dtype=np.float32)
        keypoints_uv = np.empty(
            (row_count, camera_count, keypoint_count, 2), dtype=np.float32
        )
        keypoint_depth_m = np.empty(
            (row_count, camera_count, keypoint_count), dtype=np.float32
        )
        keypoint_in_frame = np.empty(
            (row_count, camera_count, keypoint_count), dtype=bool
        )
        keypoint_visible = np.empty_like(keypoint_in_frame)
        semantic_masks = np.empty(
            (row_count, camera_count, mask_height, mask_width), dtype=np.uint8
        )
        intrinsics = np.empty((row_count, camera_count, 3, 3), dtype=np.float32)
        camera_position_world = np.empty((row_count, camera_count, 3), dtype=np.float32)
        camera_rotation_world = np.empty(
            (row_count, camera_count, 3, 3), dtype=np.float32
        )

        written = 0
        for frame_index, action44 in enumerate(actions44):
            action46 = rotvec_dual_arm_to_policy(action44)
            raw_env.step(policy_dual_arm_to_raw(action46))
            row = selected_lookup.get(frame_index)
            if row is None:
                continue
            primitives = provider.snapshot(raw_env)
            points_world = assembly_keypoints_world(
                peg_tip_world=primitives.peg_tip_world,
                peg_axis_world=primitives.peg_axis_world,
                hole_entry_world=primitives.hole_entry_world,
                hole_axis_world=primitives.hole_axis_world,
                axis_length_m=axis_length_m,
            )
            keypoints_world[row] = points_world
            for camera_row, camera_id in enumerate(camera_ids):
                render_calibration = _camera_calibration(
                    raw_env,
                    camera_id,
                    render_width,
                    render_height,
                )
                calibration = render_calibration.rescaled(
                    width=image_width,
                    height=image_height,
                )
                renderer.update_scene(raw_env._data, camera=camera_id)
                segmentation = renderer.render()
                semantic_mask = semantic_mask_from_segmentation(
                    segmentation,
                    peg_geom_ids=peg_geom_ids,
                    socket_geom_ids=socket_geom_ids,
                )
                render_uv, _, render_in_frame = render_calibration.project(points_world)
                uv, depth, in_frame = calibration.project(points_world)
                visible = keypoint_visibility(
                    semantic_mask,
                    render_uv,
                    render_in_frame,
                    support_radius_px=support_radius_px,
                )
                keypoints_uv[row, camera_row] = uv
                keypoint_depth_m[row, camera_row] = depth
                keypoint_in_frame[row, camera_row] = in_frame
                keypoint_visible[row, camera_row] = visible
                semantic_masks[row, camera_row] = resize_semantic_mask(
                    semantic_mask,
                    height=mask_height,
                    width=mask_width,
                )
                intrinsics[row, camera_row] = calibration.intrinsic_matrix
                camera_position_world[row, camera_row] = calibration.position_world
                camera_rotation_world[row, camera_row] = (
                    calibration.rotation_world_from_camera
                )
            written += 1
            if written == row_count:
                break
        if written != row_count:
            missing = sorted(set(selected_lookup) - set(range(len(actions44))))
            raise ValueError(f"selected frames exceed action range: {missing[:8]}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            episode_index=np.asarray(episode_index),
            split=np.asarray(split),
            frame_index=selected_frames,
            camera_keys=np.asarray(CAMERA_KEYS, dtype="U16"),
            keypoint_names=np.asarray(KEYPOINT_NAMES, dtype="U24"),
            image_size=np.asarray([image_height, image_width], dtype=np.int32),
            teacher_render_size=np.asarray(
                [render_height, render_width],
                dtype=np.int32,
            ),
            mask_downsample=np.asarray(mask_downsample, dtype=np.int32),
            keypoints_world=keypoints_world,
            keypoints_uv=keypoints_uv,
            keypoint_depth_m=keypoint_depth_m,
            keypoint_in_frame=keypoint_in_frame,
            keypoint_visible=keypoint_visible,
            semantic_masks=semantic_masks,
            intrinsics=intrinsics,
            camera_position_world=camera_position_world,
            camera_rotation_world=camera_rotation_world,
            axis_length_m=np.asarray(axis_length_m, dtype=np.float32),
        )
        visibility_rate = keypoint_visible.mean(axis=0)
        return {
            "episode_index": episode_index,
            "split": split,
            "num_rows": row_count,
            "frame_start": int(selected_frames[0]),
            "frame_end": int(selected_frames[-1]),
            "visibility_rate": {
                camera: {
                    keypoint: float(visibility_rate[camera_row, keypoint_row])
                    for keypoint_row, keypoint in enumerate(KEYPOINT_NAMES)
                }
                for camera_row, camera in enumerate(CAMERA_KEYS)
            },
        }
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


def main() -> None:
    args = parse_args()
    if args.frame_stride <= 0 or args.mask_downsample <= 0:
        raise ValueError("frame-stride and mask-downsample must be positive")
    estimation_dir = args.estimation_dir or args.dataset_root / "retrieval_cerebellum_estimation"
    output_dir = args.output_dir or args.dataset_root / "retrieval_cerebellum_spatial_visual"
    requested = None if args.episodes is None else set(args.episodes)
    sources = _episode_sources(
        estimation_dir,
        requested=requested,
        frame_stride=args.frame_stride,
        maximum_rows=args.maximum_rows_per_episode,
    )
    if args.max_episodes is not None:
        sources = sources[: args.max_episodes]
    zarr_paths = discover_zarr_demos(args.zarr_input_dir)
    video_image_size = _video_image_size(args.dataset_root)
    summaries: list[dict[str, object]] = []
    for progress, (episode_index, split, frames) in enumerate(sources, start=1):
        output_path = output_dir / "episodes" / f"episode_{episode_index:06d}.npz"
        if output_path.exists() and not args.overwrite:
            print(f"[{progress}/{len(sources)}] skip episode={episode_index}", flush=True)
            continue
        if episode_index >= len(zarr_paths):
            raise IndexError(f"episode {episode_index} has no zarr replay")
        actions44, initial_state = load_zarr_episode(zarr_paths[episode_index])
        if initial_state is None:
            raise ValueError(f"episode {episode_index} has no initial_state")
        print(
            f"[{progress}/{len(sources)}] episode={episode_index} rows={len(frames)}",
            flush=True,
        )
        summaries.append(
            _write_episode(
                episode_index=episode_index,
                split=split,
                selected_frames=frames,
                actions44=actions44,
                initial_state=initial_state,
                output_path=output_path,
                video_image_size=video_image_size,
                mask_downsample=args.mask_downsample,
                axis_length_m=args.axis_length_m,
                support_radius_px=args.support_radius_px,
            )
        )
    summary = {
        "stage": "V2 spatial visual supervision",
        "output_dir": str(output_dir),
        "camera_keys": list(CAMERA_KEYS),
        "keypoint_names": list(KEYPOINT_NAMES),
        "mask_classes": {"0": "background", "1": "peg", "2": "socket"},
        "frame_stride": args.frame_stride,
        "mask_downsample": args.mask_downsample,
        "axis_length_m": args.axis_length_m,
        "num_episodes_written": len(summaries),
        "num_rows_written": sum(int(item["num_rows"]) for item in summaries),
        "teacher_only": True,
        "allowed_online_inputs": False,
        "episodes": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
