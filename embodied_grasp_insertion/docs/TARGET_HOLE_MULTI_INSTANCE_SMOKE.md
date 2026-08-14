# Target Hole Multi-Instance Smoke (P0-S0.3b)

- 日期：2026-08-14T05:54:53Z
- 结论：**pass**
- reason：dual_socket_families_ok=2/2
- 范围：同 family 双 socket 的 instance_id / site / pose 区分（metadata/plumbing）
- 不声称策略知孔；不采集 / 不训练

## Families
- `round_8mm`: passed=True id_distinct=True site_distinct=True pose_sep=0.284m match_pri=True match_sec=False disc=True
- `rectangular_8mm`: passed=True id_distinct=True site_distinct=True pose_sep=0.284m match_pri=True match_sec=False disc=True

下一步：S0.4c 多族物理抓取；仍禁采集/训练。

