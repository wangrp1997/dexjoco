# ECHO-Insert Handoff

## Goal

Implement a real spiral hole search with the energy/safety layer retained. Do
not use true MuJoCo peg/socket poses to align or center the peg.

## Current Status

- Compliant full-task evaluation: `0/0` (not run).
- Tests: `43 passed`.
- The controller now continuously executes one Archimedean spiral candidate;
  the old 2 mm false-entry switch no longer interrupts the spiral.
- Current spiral parameters: 0.2 mm path step, 1.0 mm pitch, 1.5 N axial
  preload, 0.5 mrad maximum torque-compliance step, 12 mm maximum radius.
- Every spiral action still passes the optimizer energy, workspace, force,
  torque, and positive-work safety checks.

## Important Result

`v55` reached native success `1/1` and 30 consecutive bottom-contact steps,
but it is not a valid spiral-search result. It used direct demo handoff plus
true MuJoCo peg/socket geometry to align and center the peg before ECHO took
over. It only proves that insertion works after privileged alignment.

Video:
`/mnt/hdd/dexjoco/outputs/echo_insert_surface_handoff_spiral_video_v55_ep0_20260827/ep00/ego.mp4`

## Main Unresolved Problem

The wrist follows the commanded spiral, but the peg slips and tilts inside the
right hand. Therefore the peg body does not trace the same spiral. A continuous
10.8 mm wrist spiral was observed without native insertion.

## Next Steps

1. Disable privileged precontact alignment/centering for the target run.
2. Make the grasp rigid using only approved public inputs and a predeclared
   open-loop finger preload; do not tune it from true contact or object pose.
3. Verify that a small wrist motion is transferred to the peg using deployable
   observations, not MuJoCo `xpos`, `xmat`, or contact truth.
4. Run the continuous energy-layer spiral. Detect entry only from sustained
   public axial displacement plus force reduction, then freeze XY and insert.
5. Run without video first; after native `info["succeed"]`, reproduce once with
   ego video.

## Key Files

- `controller.py`: spiral generation and action execution.
- `optimizer.py`: energy model, candidate safety, and spiral parameters.
- `run_demo_handoff.py`: diagnostic runner; currently contains privileged paths.
- `DISCUSSION.md`: experiment history through v55.

## Verification

```bash
/home/wangrenpeng/miniconda3/bin/conda run -n dexjoco \
  python -m pytest -q echo_insert/tests
```

Do not report success unless native `info["succeed"]` is true. Keep privileged
diagnostics separate from eligible results.
