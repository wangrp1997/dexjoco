合规状态: 不合规
符合约束的完整成功: 0/0 (未完成合规评估)

# ECHO-Insert

**Energy-Constrained Haptic Optimization for Insertion** is the minimal P0 from
the non-privileged DexJoCo design discussion.

After an externally prepared handoff, the left arm and both hands hold their
handoff commands. The right wrist emits bounded five-dimensional micro-actions:
two tangential translations, axial translation, roll, and pitch. A short
no-contact window estimates wrench bias. An online low-dimensional interaction
model then scores safe candidate actions by axial progress, positive mechanical
work, response uncertainty, lateral load, and tactile slip when real tactile is
available.

Low-load approach samples never update the contact-response model. Interaction
starts only after three consecutive public-wrench threshold crossings, and the
positive-work safety budget uses a 30-step rolling window instead of a lifetime
latch.

All default gains and limits are untuned safety placeholders. The local
controller assumes stable grasps, a visible tray face, no surface contact, and
enough standoff clearance to rotate the peg about its estimated insert end.

This directory deliberately contains no VLA, behavior cloning, world model,
demo-derived action prior, or simulator contact/pose input. See `CONTRACT.md`.

Status: the kinematics-only P0 was falsified. Its depth-RANSAC replacement is
implemented with staged pre-contact control, contact learning, and jam recovery.
The arbitrary 0.20 m setup-distance gate and redundant direct-bearing gate are
removed; alignment motion saturates instead of terminating. Free-space approach
uses an 8 mm target lead, while contact optimization uses 0.8 mm micro-actions.

The latest episode-11, seed-0 diagnostic ran all 1,148 steps remaining after
handoff and ended only when the environment ended. It reached contact, released
two jams, reset the local model, and re-contacted at steps 934 and 1,044.
Diagnostic native success remains 0/1 and eligible full-task success remains
0/0 (not evaluated). Soft approach-workspace exhaustion no longer terminates
the runner; hard force, torque, and positive-work safety remain active. The
runner now defaults to all environment steps remaining after handoff. See
`KINEMATICS_ONLY_P0_RESULT.md` for the superseded baseline.

## Runtime Depth Frame

Once after handoff, the runner reads metric depth from the fixed ego RGB-D
camera. Public Allegro kinematics supplies a coarse left-hand tray ROI; pure
NumPy RANSAC fits the tray-facing plane, and the robust inlier-footprint center
is used as a coarse hole-center proxy. Right-hand kinematics retains the fixed
round-8 peg fit. The resulting world task frame is then frozen for ECHO.
The controller incrementally aligns the signed peg axis to the plane inward
normal while compensating wrist translation about the insert end, re-estimates
wrench bias, centers tangentially on the plane-footprint proxy, and advances
along the normal until public wrist F/T confirms surface interaction.

No MuJoCo object/site pose, segmentation ID, contact truth, demo pose, or
per-episode object transform is consumed. The plane center remains an
occlusion-sensitive approximation rather than hole truth, so degenerate or
inconsistent geometry fails closed. `--power-sign {-1,1}` remains required
because wrist F/T polarity is a hardware convention.

## Verify

From `/home/wangrenpeng/dexjoco` in the `dexjoco` Conda environment:

```bash
python -m pytest -q echo_insert/tests
python -m echo_insert.audit
python echo_insert/run_demo_handoff.py --help
```

The diagnostic runner writes the required compliance header, per-episode JSON,
`summary.json`, and `REPORT.md`. Demo handoff setup is kept outside the
controller but still makes every such run non-compliant.
