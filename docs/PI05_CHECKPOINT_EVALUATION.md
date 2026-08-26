# PI0.5 Insert Fine-Tune Checkpoint Evaluation

This protocol preserves intermediate checkpoints and compares them with the original
`bimanual_assembly` policy under identical seeds and rollout counts.

## 1. Archive checkpoints during training

OpenPI currently uses `max_to_keep=1`. The insert fine-tune saves every 2,000 steps,
so a watcher must archive each finalized checkpoint before the next save removes it.

```bash
python scripts/archive_openpi_checkpoints.py \
  --source /mnt/hdd/dexjoco/checkpoints/pi05_insert_ft/bimanual_assembly_insert_ft/insert_ft_mix_v1 \
  --archive /mnt/hdd/dexjoco/checkpoints/pi05_insert_ft_archive/bimanual_assembly_insert_ft/insert_ft_mix_v1 \
  --expected-steps 2000,4000,6000,7999 \
  --training-pid 3779356
```

The archive contains policy-serving files only (`params`, `assets`, and checkpoint
metadata). It first attempts hard links on the same filesystem, then reflinks, then a
normal copy.

## 2. Preview the evaluation commands

```bash
python scripts/eval_openpi_checkpoint_sweep.py --dry-run
```

The default paired protocol evaluates the baseline and all archived numeric checkpoints
with seeds `0,1,2`, 20 episodes per seed, action horizon 30, and replan ratio 0.8.

## 3. Run the comparison

Run this after training releases a GPU:

```bash
python scripts/eval_openpi_checkpoint_sweep.py \
  --gpu 0 \
  --seeds 0,1,2 \
  --episodes 20 \
  --output /mnt/hdd/dexjoco/outputs/pi05_insert_ft_checkpoint_eval
```

For a quick screening pass, use five episodes per seed. The final selection should rerun
all candidates with the same larger budget rather than allocating more episodes only to
promising checkpoints.

## Outputs

- `protocol.json`: immutable comparison settings and checkpoint paths.
- `<checkpoint>/seed_<seed>/`: rollout videos and per-run result.
- `ranking.csv`: aggregate success rate, worst-seed rate, and Wilson 95% interval.
- `summary.json`: run-level outcomes and paired checkpoint comparisons.
- `BEST_CHECKPOINT.txt`: top checkpoint by success rate, then worst-seed rate, then the
  earlier fine-tune step when tied.

Because this fine-tune uses insertion-only segments, include the original full-task
checkpoint as the baseline. Full-task rollout success catches catastrophic forgetting in
grasp and transport even when insertion behavior improves.

## Insertion-only comparison from varied demo handoffs

The full-task comparison must be complemented by an insertion-only protocol. This
protocol does not restore one fixed pre-insert snapshot. Instead, it selects multiple
different demonstrations and reproduces the hybrid data-generation path for each one:

1. restore that demonstration's initial simulator state;
2. replay its actions through `peg_lift_end_frame`;
3. run the same deterministic hybrid `_approach` used before FT segment recording;
4. hand control to the OpenPI checkpoint and score the normal 30-step bottom contact.

Preview the selected demonstrations and commands:

```bash
python scripts/eval_openpi_insert_checkpoint_sweep.py --dry-run
```

Run 30 varied demo handoffs for the baseline and every archived checkpoint:

```bash
python scripts/eval_openpi_insert_checkpoint_sweep.py \
  --gpu 3 \
  --episode-count 30 \
  --output /mnt/hdd/dexjoco/outputs/pi05_insert_ft_handoff_eval
```

Use the insertion-only ranking together with the full-task ranking: insertion-only
success measures the skill that was fine-tuned, while full-task success measures retained
grasp and transport capability.

Both evaluation paths now emit structured failure reasons. Full-task episodes distinguish
tray grasp/lift failure, peg grasp/lift failure, bimanual coordination failure, object loss,
transport failure, hole-entry failure, and unstable insertion. Demo-handoff insertion
episodes distinguish setup failure, tray/peg loss, hole-entry failure, unstable insertion,
and timeout. Infrastructure errors abort the run instead of being counted as policy failures.
