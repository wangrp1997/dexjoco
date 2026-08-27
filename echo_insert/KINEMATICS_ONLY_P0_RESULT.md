合规状态: 不合规
符合约束的完整成功: 0/0 (未完成合规评估)

# Kinematics-Only P0 Result

## Decision

Stop the encoder-only coarse-geometry branch. It did not produce a repeatable
first contact, so ECHO never received the local interaction samples it needs.
Do not tune proxy angles, expand the blind workspace, or select variants using
simulator object truth.

## Protocol

- Diagnostic only: demo replay and privileged approach prepared each handoff.
- Episodes: 11, 12, 13; seed 0; at most 900 controller steps.
- Post-handoff action inputs: public state46, previous action44, named wrist F/T.
- No object/site pose, contact truth, reward, success, or demo action entered the
  controller or kinematic estimator.

## Results

The first grasp-axis proxy admitted episodes 11 and 12, rejected episode 13 for
axis disagreement, and achieved diagnostic native success 0/2. Both admitted
episodes used 50 approach steps (40 mm), never confirmed interaction, performed
zero RLS updates, and then held at the advance workspace limit.

The corrected grasp-center-bearing proxy admitted all three episodes and
achieved diagnostic native success 0/3:

| Episode | Baseline | Approach | Interaction confirm | RLS updates | Terminal |
| --- | ---: | ---: | ---: | ---: | --- |
| 11 | 9 | 50 | 2 | 0 | advance workspace limit |
| 12 | 9 | 50 | 0 | 0 | advance workspace limit |
| 13 | 9 | 50 | 0 | 0 | advance workspace limit |

The two force-threshold frames in episode 11 did not satisfy the three-sample
interaction debounce and therefore correctly did not initialize the contact
model.

After adding the explicit no-contact terminal, episode 11 reproduced the same
trajectory and stopped at controller step 62: 9 baseline, 50 approach, 2
interaction-confirm holds, and 1 terminal safety step. The stop reason was
`no_contact_workspace_exhausted`.

## Interpretation

Analytic Allegro FK is correct as robot kinematics, but it does not observe the
held object. Without deployable fingertip contact correspondence, the same
encoder state is compatible with multiple peg/tray poses. Known CAD does not
resolve the missing surface correspondence. The grasp-center bearing is also not
a hole axis and did not reliably reach the tray surface.

The next admissible coarse-geometry module needs at least one new deployable
observation: calibrated RGB object pose/mask features, or real per-finger
contact location/normal with a no-slip validity signal. Current DexJoCo exposes
RGB only, not depth or masks; FoundationPose is not installed and cannot be
silently substituted without those inputs.
