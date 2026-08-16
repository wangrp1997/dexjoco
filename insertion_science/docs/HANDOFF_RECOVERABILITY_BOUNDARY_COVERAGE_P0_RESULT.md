# Handoff Recoverability Boundary / Coverage P0 — Result

- UTC: `2026-08-15T16:45:25Z`
- Protocol: `HandoffRecoverabilityBoundaryCoverageP0`
- Verdict: `branch1_fails_mostly_outside_boundary`
- Decision: `redesign_handoff_data_generation`
- Reason: outside_frac=0.929 >= 0.6

## Anisotropic boundary (max recoverable scale, rate≥0.5)

- `axis_tx` (axis): max_scale=`0.0`
- `axis_ty` (axis): max_scale=`0.0`
- `finger_close` (finger): max_scale=`2.0`
- `o2h_shift_x` (o2h): max_scale=`1.0`
- `tip_along_n` (tip_along): max_scale=`1.0`
- `tip_along_p` (tip_along): max_scale=`2.0`
- `tip_lat_nx` (tip_lat): max_scale=`2.0`
- `tip_lat_px` (tip_lat): max_scale=`0.5`
- `tip_lat_py` (tip_lat): max_scale=`0.5`

## Kind envelope (min across directions = fragile)

- `axis`: min_dir=`0.0`, max_dir=`0.0`, n_dirs=`2`
- `finger`: min_dir=`2.0`, max_dir=`2.0`, n_dirs=`1`
- `o2h`: min_dir=`1.0`, max_dir=`1.0`, n_dirs=`1`
- `tip_along`: min_dir=`1.0`, max_dir=`2.0`, n_dirs=`2`
- `tip_lat`: min_dir=`0.5`, max_dir=`2.0`, n_dirs=`3`

## Conservative limits used for coverage

- `tip_lat`: `0.5`
- `tip_along`: `1.0`
- `axis`: `0.0`
- `o2h`: `1.0`
- `finger`: `2.0`

## Coverage

- fails: `14` outside=`13` inside=`1` inside_with_pair=`1`
- outside_frac: `0.929`

## Per-fail (compact)

- src=`archived_eval_fail_traj` ep=`16` inside=`False` outside_axes=`['tip_lat']` scales=`{'s_lat': 2.08, 's_along': 0.8, 's_tip': 0.74}` pair=`True`
- src=`archived_eval_fail_traj` ep=`1` inside=`False` outside_axes=`['tip_lat', 'tip_along']` scales=`{'s_lat': 0.88, 's_along': 1.27, 's_tip': 1.05}` pair=`True`
- src=`archived_eval_fail_traj` ep=`27` inside=`False` outside_axes=`['tip_along']` scales=`{'s_lat': 0.23, 's_along': 7.35, 's_tip': 4.87}` pair=`True`
- src=`archived_eval_fail_traj` ep=`33` inside=`False` outside_axes=`['tip_lat', 'tip_along']` scales=`{'s_lat': 1.9, 's_along': 4.69, 's_tip': 3.38}` pair=`True`
- src=`archived_eval_fail_traj` ep=`40` inside=`False` outside_axes=`['tip_lat', 'tip_along']` scales=`{'s_lat': 27.9, 's_along': 50.95, 's_tip': 41.74}` pair=`False`
- src=`archived_eval_fail_traj` ep=`48` inside=`False` outside_axes=`['tip_lat', 'tip_along']` scales=`{'s_lat': 5.66, 's_along': 2.35, 's_tip': 3.22}` pair=`True`
- src=`archived_eval_fail_traj` ep=`63` inside=`False` outside_axes=`['tip_lat']` scales=`{'s_lat': 2.64, 's_along': 0.85, 's_tip': 1.32}` pair=`True`
- src=`archived_eval_fail_traj` ep=`68` inside=`False` outside_axes=`['tip_lat', 'tip_along']` scales=`{'s_lat': 2.61, 's_along': 1.87, 's_tip': 0.98}` pair=`True`
- src=`archived_eval_fail_traj` ep=`74` inside=`False` outside_axes=`['tip_lat', 'tip_along']` scales=`{'s_lat': 0.88, 's_along': 6.98, 's_tip': 4.72}` pair=`True`
- src=`archived_eval_fail_traj` ep=`88` inside=`False` outside_axes=`['tip_lat', 'tip_along']` scales=`{'s_lat': 12.37, 's_along': 5.55, 's_tip': 7.75}` pair=`True`
- src=`archived_eval_fail_traj` ep=`9` inside=`True` outside_axes=`[]` scales=`{'s_lat': 0.08, 's_along': 0.29, 's_tip': 0.21}` pair=`True`
- src=`demo_identity_fail_handoff` ep=`0` inside=`False` outside_axes=`['tip_lat', 'tip_along', 'axis']` scales=`{'s_lat': 0.95, 's_along': 8.66, 's_tip': 5.82, 's_axis': 1.43}` pair=`True`
- src=`demo_identity_fail_handoff` ep=`2` inside=`False` outside_axes=`['tip_lat', 'tip_along', 'axis']` scales=`{'s_lat': 16.27, 's_along': 15.4, 's_tip': 13.22, 's_axis': 0.38}` pair=`True`
- src=`demo_identity_fail_handoff` ep=`11` inside=`False` outside_axes=`['tip_lat', 'tip_along', 'axis']` scales=`{'s_lat': 5.16, 's_along': 5.66, 's_tip': 4.93, 's_axis': 0.14}` pair=`True`

## Note

归档 fail traj 仅作 handoff 状态样本；不训练、不复活 PrivHI 主线。
axis 在归档 traj 中缺失时，覆盖判定不加 axis 轴。
