# Upstream Handoff Targeting P0 — Result

- UTC: `2026-08-15T18:06:43Z`
- Protocol: `UpstreamHandoffTargetingP0`
- Verdict: `fail_stop_handoff_direction`
- Decision: `stop_handoff_direction`
- Reason: handoff=0.000 basin_hit=0.000 basin_insert=0.000 checks={'handoff_rate_ok': False, 'basin_hit_rate_ok': False, 'basin_insert_rate_ok': False}

## Rates

- seeds: `12`
- handoff_rate: `0.000` (0/12)
- basin_hit_rate (among handoff): `0.000` (0/0)
- basin_insert_rate (among basin hits): `0.000` (0/0)

## Gates

- min_handoff_rate: `0.5`
- min_basin_hit_rate: `0.4`
- min_basin_insert_rate: `0.5`
- checks: `{'handoff_rate_ok': False, 'basin_hit_rate_ok': False, 'basin_insert_rate_ok': False}`

## Per-seed

- seed=`0` demo=`28` handoff=`False` basin=`False` insert=`False` reason=`handoff_never` outside=`None`
- seed=`1` demo=`25` handoff=`False` basin=`False` insert=`False` reason=`tray_lift_hold_unstable` outside=`None`
- seed=`2` demo=`24` handoff=`False` basin=`False` insert=`False` reason=`tray_lift_hold_unstable` outside=`None`
- seed=`3` demo=`10` handoff=`False` basin=`False` insert=`False` reason=`tray_lift_hold_unstable` outside=`None`
- seed=`4` demo=`82` handoff=`False` basin=`False` insert=`False` reason=`tray_lift_hold_unstable` outside=`None`
- seed=`5` demo=`65` handoff=`False` basin=`False` insert=`False` reason=`tray_lift_hold_unstable` outside=`None`
- seed=`6` demo=`21` handoff=`False` basin=`False` insert=`False` reason=`tray_lift_hold_unstable` outside=`None`
- seed=`7` demo=`63` handoff=`False` basin=`False` insert=`False` reason=`tray_lift_hold_unstable` outside=`None`
- seed=`8` demo=`0` handoff=`False` basin=`False` insert=`False` reason=`tray_lift_hold_unstable` outside=`None`
- seed=`9` demo=`34` handoff=`False` basin=`False` insert=`False` reason=`tray_lift_hold_unstable` outside=`None`
- seed=`10` demo=`80` handoff=`False` basin=`False` insert=`False` reason=`tray_lift_hold_unstable` outside=`None`
- seed=`11` demo=`24` handoff=`False` basin=`False` insert=`False` reason=`tray_lift_hold_unstable` outside=`None`

## Note

无 force-demo / restore；不训练。失败则停止整个 handoff 方向。
