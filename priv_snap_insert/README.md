# Privileged Snap-Servo Insertion

Demo 回放到抓取+抬起之后（默认 `peg_lift_end`），锁定 peg–手相对位姿（snap o2h），手指不动，用特权 tip/孔几何做腕部伺服插孔。

## 这是什么 / 不是什么

- 是：特权上界 / 正对照。
- 不是：可部署策略、不是训练、不是 residual/gate。

## 流程

1. 回放 demo → handoff（默认 `peg_lift_end`；`peg_lift_start` 测长距运输）
2. 锁定 object_in_hand；双手手指冻结；左臂冻结
3. 右臂：clear → transport → align → insert
4. 每步 snap；终止：insert_ok / timeout / tip 发散

## 快速跑

```bash
conda activate dexjoco
export MUJOCO_GL=egl
cd /home/wangrenpeng/dexjoco
python -m priv_snap_insert.run_p0 --episodes 1 --max-servo-steps 600
python -m priv_snap_insert.run_p0 --episodes 1 --handoff peg_lift_start --max-servo-steps 1000
```

输出：`/mnt/hdd/dexjoco/outputs/priv_snap_insert/`

## 文档

- docs/PROTOCOL.md
- docs/SEARCH_NOTES.md
