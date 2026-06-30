结论：**`/home/wangrenpeng/dexjoco/OpenTrack` 可直接当母版改，通用层（`traj_class`、PPO/DAgger、wrapper）基本复用；主要工作量在 MJX 环境、控制方式、zarr→npz 三条线。**

---

## 总体差距

| 维度 | OpenTrack (G1) | DexJoCo assembly |
|------|----------------|------------------|
| 仿真 | MJX + Brax PPO | Gym + 原生 MuJoCo |
| 控制 | 关节 PD 力矩，`action = Δq` | mocap opspace + Allegro position |
| 参考轨迹 | LAFAN1 已是关节 `qpos` npz | zarr 是 46d action，**无完整 qpos** |
| 随机化 | 机身/地面 DR | `reset(seed)` 随机 peg/tray xy+yaw |
| 成功 | 跟踪误差 | tray/peg 抬升 + 接触 |

**最大工程点**：不能直接把 zarr action 当 ref，要先 **特权回放录 `qpos/qvel/xpos`**，再训 **关节空间 tracker**（和 G1 同范式）。

---

## 工作清单（建议顺序）

### Phase 0 — 环境与基线（0.5 天）

1. `cd OpenTrack && uv sync`，设 `GLI_PATH=/home/wangrenpeng/dexjoco/OpenTrack`
2. 确认 JAX+CUDA 能跑（可先跑 G1 小步 smoke，验证 Brax 链路）
3. 决定：**assembly 训练独立 conda/uv**，还是把 `track_mj` 装进现有 `dexjoco` env（Python 3.12 + jax 0.4.38 可能与现 env 冲突）

### Phase 1 — 资产与 MJX 模型（2–3 天）

4. 新建 `OpenTrack/storage/assets/panda_bimanual_assembly/`
   - 从 `dexjoco/.../xmls/arena_arm_hand_bimanual_assembly.xml` 拷一份
   - 写 `scene_mjx_flat.xml`（平面桌、无随机相机/纹理）
5. **控制改造**（必做）：
   - 训练用模型走 **关节 PD**（双臂 7+16×2），不用 mocap opspace
   - 为 tracking 补 actuator / sensor（参考 G1 的 `g1_mjx.xml`）
6. 新建 `track_mj/envs/assembly_tracking/assembly_tracking_constants.py`
   - `ACTION_JOINT_NAMES`、`Kp/Kd`、`DEFAULT_QPOS`
   - 跟踪 body/site：`target_left/right`、指尖、peg/tray
   - `task_to_xml()` 指向新 scene

### Phase 2 — 数据管线 zarr → OpenTrack npz（2–3 天）

7. 新建 `OpenTrack/scripts/convert_dexjoco_zarr.py`：
   - 读 `manifest.json` + `replay.zarr`
   - **特权回放**（`restore_initial_state` + 逐步 `step(action)`）
   - 每帧录：`qpos, qvel, xpos, xquat, site_xpos...` → `Trajectory.save()`
8. 按 manifest 切段导出（先做一条 demo 冒烟）：
   - `tray_approach` / `tray_squeeze` / `tray_lift`
   - `peg_approach` / `peg_squeeze` / `peg_lift`
   - 或整段 `ep{N}_full`
9. 走一遍 `preprocess_trajectory`（`extend_motion` + 50Hz 插值 + 重算 qvel）
10. 存到 `storage/data/mocap/bimanual_assembly/PandaBimanual/{name}.npz`

**复用**：`interaction_retarget/io/zarr_io.py`、`sim/replay.py` 的 `raw_flat_to_dict`

### Phase 3 — MJX Tracking Env（3–5 天，核心）

11. `assembly_tracking/train/base_env.py` — 仿 `g1_tracking/train/base_env.py`
12. `assembly_env_tracking_general.py` — 从 G1 拷改：
    - `prepare_trajectory()`：路径改为 `PandaBimanual` 子目录
    - **reset**：`rand_obj`（peg/tray free joint 随机）+ RSI 随机 ref 起点
    - **obs**：`joint_pos/vel`、`dif_joint_pos/vel`、可选物体相对位姿
    - **reward**：关节跟踪（主）+ 指尖/物体跟踪（辅）+ lift Δz 奖励
    - **done**：偏离 ref 过大 / 物体飞 / episode 结束
13. `utils/wrapper.py` — 可直接复用 G1 的 vmap/episode wrapper
14. `@registry.register` 注册：
    - `AssemblyTrackingGeneral`
    - `AssemblyTrackingGeneralDR`

### Phase 4 — 域随机化（1–2 天）

15. `dr/domain_randomize_assembly.py`：
    - peg/tray 初始 xy+yaw（对齐 `PandaBimanualAssemblyGymEnv.reset`）
    - 质量/摩擦随机（对齐 `randomize_dynamics`）
    - 可选桌面高度 Δh
16. ref 轨迹：**在 demo 录制 pose 下 warp**（物体系 DITTO 思路只用于生成 ref npz，不再开环执行）

### Phase 5 — 训练与评测（2–3 天）

17. **Specialist**：`python -m track_mj.learning.train.train_ppo_track --task AssemblyTrackingGeneralDR --exp_name ep35_tray_lift`
18. **Generalist**：`train_dagger` 多 ep 混合 + DR
19. 评测脚本 `scripts/eval_assembly_track.py`：
    - 加载 checkpoint，跑 `rand_obj` seed sweep
    - 指标：tray lift ≥30mm、peg lift ≥10mm、跟踪 RMSE
    - 可选录 `outputs/track_replay/ep{N}_seed{S}/`

### Phase 6 — 清理与文档（0.5 天）

20. `traj_process.py` 里去掉对 `g1_tracking_constants` 的硬编码（或 assembly 专用一份）
21. `OpenTrack/REFS.md`：dexjoco 路径、manifest 字段、训练命令
22. `interaction_retarget/` **保持封存**，新链路全在 `OpenTrack/`

---

## 建议目录增量

```
OpenTrack/
├── storage/assets/panda_bimanual_assembly/   # 新 XML
├── storage/data/mocap/bimanual_assembly/PandaBimanual/*.npz
├── track_mj/envs/assembly_tracking/        # 新 env（仿 g1_tracking 结构）
├── track_mj/dr/domain_randomize_assembly.py
├── scripts/convert_dexjoco_zarr.py
└── scripts/eval_assembly_track.py
```

`track_mj/utils/dataset/*`、`learning/train/*`、`learning/policy/*` **先不动**。

---

## 风险点

1. **mocap → 关节 PD**：接触段动力学会变，需调 `Kp/Kd`、`sim_dt`、substeps
2. **自由物体在 ref 里会动**：跟踪的是整场景 `qpos`（含 peg/tray），reset 时只随机物体初值，ref 要按物体位姿 warp
3. **依赖隔离**：OpenTrack 锁 Python 3.12.9 + jax 0.4.38，和 dexjoco conda 可能两套环境
4. **DAgger 放后面**：先单 ep specialist 跑通 lift，再做多 demo generalist

---

## 第一周最小里程碑（MVP）

```
manifest ep35 → convert 出 tray_lift.npz
→ AssemblyTrackingGeneral env 能 reset+step
→ PPO 训 1M step 能在固定 seed 跟住 ref
→ seed=0 时 tray Δz > 0（还不强求 30mm）
```

---

建议 **从 Phase 2（zarr→npz 转换脚本）+ Phase 1（MJX XML）并行开工**；env 和训练都依赖这两块。  
你要的话我可以下一步直接写 `convert_dexjoco_zarr.py` 骨架和 `assembly_tracking_constants.py` 占位。