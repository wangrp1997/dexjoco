# Search Notes（特权伺服插孔）

检索时间：2026-08-17。目的：定控制律形态，不是找可训网络。

## 查询

1. `privileged state peg-in-hole Cartesian servo tip socket lateral axis`
2. `peg-in-hole impedance align then insert phase machine`
3. `FixedJoint snap object in hand insertion oracle Isaac`

## 要点（可落地到本仓）

| 来源 | 机制 | 对本实验 |
|------|------|----------|
| 经典 peg-in-hole 状态机（approach / search / align / insert） | 侧向对齐后再深插，避免 jam | 用 `lat_gate` + `ang_gate` 门控进给 |
| PBVS / tip feature error | `e_lat`, `e_along`, `e_axis` → tip 速度 | 直接用仓内 `pbvs_tip_feature_error` / `pbvs_tip_velocity` |
| Franka/Isaac privileged oracle + FixedJoint | 手指与刚体附着打架会伪装成控制误差；纯 snap/FixedJoint 可接近 100% | 每步 `snap_peg` 锁 o2h；手指仍发恒定目标但不靠接触传力 |
| 侧向误差 + 角误差导致 wedging | 三维侧向两分量 | 法平面纠偏 + 轴对齐，不做二维截面假设 |

## 明确不做

- 螺旋搜索 / force-based search（本 P0 先测「无搜索、纯特权对齐」上界）
- 学习阻抗增益 / residual policy
- 视觉估计孔位（特权真值）

## 与 Insertion Science 短名单关系

本目录是**独立特权上界实验**，不进入 `insertion_science` 可训候选短名单；不复活已停的 PAS/PrivHI residual 线。
