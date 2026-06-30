# skill_graph

VLA eval 里的重抓 recovery 包：抓取掉了 → 模板库选最优 → demo_warp 重抓 → 交还策略继续。

## 1. 导出模板库（一次性）

```bash
conda activate dexjoco
cd ~/dexjoco
export PYTHONPATH=~/dexjoco MUJOCO_GL=egl
python scripts/export_grasp_templates.py
```

输出：`/mnt/hdd/dexjoco/skill_graph/bimanual_assembly/grasp_templates/`

## 2. ForceVLA / π0.5 eval 里开启

在原有 eval 命令上加 `--skill-graph-recovery`：

```bash
# π0.5 示例（按你现有 config / port 改）
python -m dexjoco_openpi_client.cli.evaluate \
  --config <your_config.yaml> \
  --skill-graph-recovery

# ForceVLA 同理，eval 入口相同
```

判定：tray/peg 各自 `dz >= 6cm` 即算抓稳；不要求 peg 世界高度超过 tray。peg 重抓会在物体坐标系绕 Z 轴试 4 个抓取方向，并跳过 demo 长 approach 直接贴近抓取。重抓进度按 `attempt + chunk` 显示。

视频目录名会多 `_skill_graph` 后缀，与普通 eval 分开，例如：
- `outputs/pi0.5/bimanual_assembly_skill_graph_seed0/`
- `outputs/forcevla/bimanual_assembly_skill_graph_seed0_step50000/`

## 目录

| 路径 | 作用 |
|------|------|
| `graphs/bimanual_assembly.json` | 任务图 + recovery 定义（参考） |
| `skills/regrasp/` | 模板选优 + demo_warp 执行 |
| `hooks/vla_recovery.py` | 挂到 eval loop 的 hook |
| `runtime/` | 图加载/执行（后续扩展） |

## 说明

- 不是独立 eval；成功指标看 VLA eval 成功率。
- 插孔仍由 VLA 正常执行；本包只在抓稳失败时重抓，恢复后继续走策略。
- approach 仅 demo_warp；CuRobo 未接。
