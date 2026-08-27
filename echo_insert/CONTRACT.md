合规状态: 不合规
符合约束的完整成功: 0/0 (未完成合规评估)

# ECHO-Insert Data Contract

ECHO-Insert is an insertion-only prototype. Runs that use demo replay to create
the handoff state are `insertion_only / non-compliant diagnostic`; they are not
eligible DexJoCo full-task results.

```text
policy_observation:
  state46: float[46], measured right/left TCP pose and hand encoders
  previous_action44: float[44], previous public policy command
  wrist_wrench_local: float[2, 6], named wrist F/T sensors only
  fingertip_load: optional float[2, 4], deployable tactile sensors only
  ego_depth_m: optional float[640,640], fixed calibrated ego RGB-D z-depth
  task_basis_world: float[3,3], frozen runtime depth-plane approach frame
training_feedback: none for the online P0 controller; native reward,
  termination, and info["succeed"] may be recorded outside the action path
forbidden_sources: demos/teachers after handoff; full simulator state; object,
  body, geom, or site pose; xpos/xmat; contact/grasp/phase truth; simulator
  segmentation IDs; insertion depth; target IDs; cfrc_ext fingertip-force
  proxy; shaped reward; success as a policy input
full_success: native DexJoCo/OpenPI info["succeed"]
```

The local insertion controller is valid only after a no-contact handoff with
stable grasps, a visible tray face, and enough clearance for bounded wrist
alignment. It owns peg-axis alignment, coarse plane-center targeting, guarded
surface approach, and local insertion; it is not a global reaching controller.

The runtime task frame uses the 46-D state, fixed robot/peg geometry, and one
metric depth frame from the fixed calibrated ego RGB-D camera. Left-hand
kinematics supplies only a coarse tray ROI. Pure NumPy RANSAC fits a tray-facing
plane in that ROI; the robust finite inlier-footprint center is used as a coarse
hole-center proxy. The fitted inward plane normal defines the insertion axis;
the direct peg-to-center bearing is diagnostic only. The resulting world frame
is frozen so wrench, motion, RLS history, and command offsets remain in one
or inconsistent geometry fails closed. The required `--power-sign {-1,1}`
specifies the named wrist F/T convention and must not be inferred from demo,
reward, simulator contact, or outcome.

Depth is an explicitly approved deployable sensor input for this prototype.
Runtime code uses no simulator segmentation or object/site pose. The plane
footprint center is still only an approximation: occlusion or partial visibility
can bias it, and real deployment requires a calibrated RGB-D camera with the
same frame convention.

The current DexJoCo model has explicit wrist F/T sensors but no deployable
fingertip tactile sensor. The simulation runner must therefore mark tactile as
unavailable; `cfrc_ext` is forbidden even though older ForceVLA code exposes it.

The controller receives a constructed observation object, never an environment
or raw MuJoCo handle. Demo handoff setup and native success accounting live in a
separate diagnostic runner and cannot pass data into action selection.
