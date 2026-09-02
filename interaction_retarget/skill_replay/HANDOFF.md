# Demo → handoff

把 `bimanual_assembly` demo 开环回放到 `peg_lift_end`（右抓 peg、左抓 tray）。  
核心函数：`demo_replay_to_pre_insert`。到此为止不接策略、不插孔。

## 1. 数据准备

需要两样东西：

| 数据 | 作用 |
|------|------|
| demo zarr | 原始双臂演示轨迹（每集一个 `replay.zarr`） |
| sidecar | `manifest.json` + 每集 timing（含 `peg_lift_end` 等） |

**已有 sidecar**：默认目录  
`/mnt/hdd/dexjoco/interaction_sidecar/bimanual_assembly`  
（代码里：`default_sidecar_dir("bimanual_assembly")`）

**没有 sidecar、只有 zarr 时**，先导出：

```bash
cd ~/dexjoco
conda activate dexjoco
export PYTHONPATH=~/dexjoco:~/dexjoco/dexjoco

python scripts/build_interaction_sidecar.py \
  --zarr-root <你的zarr根目录> \
  --out-dir <你的sidecar目录>
```

`zarr` 根目录下应是各集 demo（含 `replay.zarr`）。  
`manifest.json` 里的 `zarr_path` 必须能读到对应文件。

## 2. 只跑到 handoff 并出视频

```bash
export MUJOCO_GL=egl

python scripts/replay_demo_handoff_video.py \
  --episodes 1,5 \
  --output <你的输出目录> \
  --sidecar-dir <你的sidecar目录>
```

`--sidecar-dir` 若省略则用默认 sidecar。  
输出：`epXX_handoff_ego.mp4`。

## 3. 后续接入自己的算法

handoff 之后接管 sim，自己写控制即可。建议：

1. 在仓库旁或仓内**新建目录**（例如 `my_insert/`），不要改 `skill_replay`。
2. 每集开始：回放到 `peg_lift_end`（与上面同一套 API）。
3. 记下当前双臂动作（手指保持 handoff），再从下一帧起发你的腕部/力控命令。

最小接法：

```python
from interaction_retarget.io.zarr_io import load_zarr_episode
from interaction_retarget.sim.replay import make_assembly_env, rotvec_dual_arm_to_policy
from interaction_retarget.sim.settle import read_arm_action
from interaction_retarget.skill_replay.insert import (
    demo_replay_to_pre_insert,
    dual_arm23_to_action44,
)
from pose_insert.pre_insert import resolve_peg_lift_end_frame

env = make_assembly_env(seed=0, randomize=False)
raw = env.unwrapped
peg_lift_end = resolve_peg_lift_end_frame(entry, sidecar_dir)
_, _, init = load_zarr_episode(entry["zarr_path"])

demo_replay_to_pre_insert(
    env, raw,
    zarr_path=entry["zarr_path"],
    stop_frame=int(peg_lift_end),
    initial_state=init,
)
# === handoff 完成；下面换成你的算法 ===
hold44 = dual_arm23_to_action44(
    read_arm_action(raw, "left"),
    read_arm_action(raw, "right"),
)
# 每步：改 hold44 的腕部位姿 → rotvec_dual_arm_to_policy → env.step
```

也可用 `reach_insert_rl.env.handoff_env.InsertHandoffEnv`：`reset()` 已内置同一条回放，之后在 `step` 里接自己的动作。

约定：

- 手指默认锁在 handoff；先动腕部。
- 成功/掉落判定可复用 `hybrid_insert.assembly_contacts.AssemblyContactLabeler`。
- 新算法放独立包，通过 `PYTHONPATH` 挂上 `dexjoco` 与 `dexjoco/dexjoco` 即可。

## 相关入口

| 用途 | 位置 |
|------|------|
| 导出 sidecar | `scripts/build_interaction_sidecar.py` |
| 回放到 handoff | `skill_replay/insert.py` → `demo_replay_to_pre_insert` |
| 停止帧 | `pose_insert.pre_insert.resolve_peg_lift_end_frame` |
| 只出视频 | `scripts/replay_demo_handoff_video.py` |
