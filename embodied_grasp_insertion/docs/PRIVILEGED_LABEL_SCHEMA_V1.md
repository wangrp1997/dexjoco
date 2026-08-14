# Privileged Label Schema v1 (P0-L1)

- protocol: `P0-L1`
- schema_version: `privileged_label_v1`
- privilege_only: true；deployment_input: false
- 非训练数据集；禁止写入 `pilot_micro_demo_v0`

## Included

- object_in_hand_pose_6d
- object_in_hand_velocity (finite-diff contract)
- peg_hand_contact count/by_finger
- finger_force norm/contact_active
- outcome_raw tray_ok/peg_ok/insert_ok
- provenance family/instance/episode/frame/root

## Excluded

- `slip_truth`
- `slip`
- `contact_mode_capture`
- `contact_mode_rim`
- `contact_mode_jam`
- `contact_mode_partial`
- `contact_mode_seated`
- `contact_mode_backout`
- `regrasp_needed`
- `peg_loss_risk`

## Velocity contract

```json
{
  "method": "finite_difference_between_consecutive_control_frames",
  "dt_source": "control_dt_seconds = model.opt.timestep * frame_skip (default skip=10)",
  "linear": "v_lin = (t_k - t_{k-1}) / dt ; t in allegro_palm_right frame (m/s)",
  "angular": "omega = rotvec(R_{k-1}^{-1} R_k) / dt ; R from o2h rotvec (peg relative palm); rad/s; NOT raw rotvec subtraction",
  "first_frame": "velocity fields null (insufficient history)",
  "reference_body": "allegro_palm_right"
}
```

- contact_force_eps_N: 0.05
- finger_order: ['index', 'middle', 'ring', 'thumb']
- reference_body: `allegro_palm_right`

## Notes

- Not a training dataset.
- Never write into pilot_micro_demo_v0.
- Do not name slip without _proxy suffix; L1 omits slip entirely.
- Fine contact modes require a separate contract before generation.
