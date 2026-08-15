# Controller Compliance Causal P0

- 日期：2026-08-15
- 状态：预注册后执行；禁止事后改门槛或按 episode 调 gain

## 科学问题

固定 operational-space gains 下的 pose/finger 接口，是否掩盖了接触顺应性对
retention、载荷、jam 与插入进展的因果作用。

## 干预

- 相同 MuJoCo snapshot；腕部/手指 action44 序列完全一致。
- 只改变双臂 OSC `pos_gains` / `ori_gains` / `damping_ratio`（运行时 monkeypatch，
  不修改 dexjoco / reach / embodied 源码）。
- 预注册刚度倍率：`1.0`（baseline）、`0.5`、`0.25`；阻尼比固定 `4.0`。
- 动作条件：`hold`（全零 delta）与 `demo_matched`（demo 腕+指）。

## Roots（冻结，执行前）

| 角色 | episode | frame | phase |
|------|---------|-------|-------|
| discovery | 4 | 484 | pre_insert |
| discovery | 6 | 271 | transport |
| held-out | 9 | 407 | pre_insert |
| held-out | 3 | 431 | pre_insert |

来源：既有 Stage-1 / S1b 冻结列表；本 P0 不扩采、不改选。

## 指标

peg retention、o2h 平移/旋转漂移、腕部 FT 范数、接触力代理、tip/lat 进展、
jam 代理（高载荷 + tip 停滞）、数值稳定性（NaN/爆炸）。

## 通过线（预注册）

1. matched restore 近 bit 可复现；
2. ≥2 个 held-out root 上，相对 `1.0×` 的效应超过 replay/noise 门；
3. 至少一个预注册倍率在 held-out 上方向一致；
4. 不得仅表现为 force↓ 且 retention/progress 显著恶化。

失败 → 立即停止本线，不设计动作接口/训练硬门。
