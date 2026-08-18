# Protocol: Privileged Snap-Servo Insert P0

## 科学问题

在 DexJoCo bimanual peg-in-hole 上，若抓取后把 peg **刚体锁定**在掌心相对位姿、手指不动，仅用特权几何误差做腕部伺服，能否达到 `insert_ok`？

这回答的是**控制可达上界**，不是「如何从图像学策略」。

## 干预变量

| 变量 | 设定 |
|------|------|
| 上游 | demo raw 回放到 handoff 帧（默认 `peg_lift_start`） |
| o2h | handoff 瞬间锁定；每步 `snap` 回写 freejoint |
| 手指 | 双手 16D 冻结在 handoff |
| 左臂 | 冻结（托盘已抓） |
| 右臂 | 特权 PBVS：侧向纠偏 + 轴对齐 + 条件进给 |
| 观测 | tip / socket / hole_axis / peg_axis（仿真真值） |

## 伺服律（经典）

参考文献中 peg-in-hole 的分阶段笛卡尔伺服：

1. **侧向**：`v_lat = -λ_xy · e_lat`（孔轴法平面）
2. **轴向对齐**：腕旋转使 peg_z ∥ ±hole_axis
3. **进给门控**：仅当 `lat_err ≤ lat_gate` 且 `axis_err ≤ ang_gate` 时允许沿孔轴推进
4. **目标深度**：`target_along_m`（负值 = 深入孔内）

实现复用 `hybrid_insert.geometry` 的 `pbvs_tip_*` / `wrist_rotvec_align_peg_axis`。

## 成功与失败

- **成功**：`AssemblyContactLabeler.insert_ok == True`（peg 与 insert geom 接触）
- **失败**：servo 步数用尽、或 peg 掉出抓取几何（snap 开启时 peg_ok 应几乎恒真）

## P0 生死门（诊断，非训策略）

| 门 | 条件 | 含义 |
|----|------|------|
| G1 | 单 ep smoke `insert_ok` | 伺服律 + snap 链路通 |
| G2 | ≥4/8 held-out ep `insert_ok` | 非偶然单条 demo |
| G3 | 关 snap 对照成功率明显下降 | 证明「固定相对位姿」是必要干预 |

默认 handoff：`peg_lift_end`（抓+抬后）。`peg_lift_start` 为长距压力测试。

不过 G1 → 先调增益/门控。  
G1 过、G2 不过 → 记录近失，不宣称普适上界。  
过 G1/G2 只说明特权上界存在，不授权训策略或复活 residual/PAS。

## 禁止

- 策略训练 / episode 扩采
- residual / gate / skill sewing
- 把本实验写成「可部署插入算法」
