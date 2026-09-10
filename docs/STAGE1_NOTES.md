# Stage-1 supervised fQRS (Challenge-2013)

- **Split:** record-level 60/15 train/val windows 3420/855 (seed 42)
- **Model:** backbone=`transformer`, `use_sqi=false`, QRS-only loss
- **Hardware:** `cpu`
- **Epochs:** 8
- **Best val F1@50ms:** 0.7695 (epoch 8)
- **Best val F1@100ms (that epoch):** 0.7965
- **Sens/PPV @50ms:** 0.685 / 0.877
- **Checkpoint:** `checkpoints/stage1_challenge2013.pt` (local; not in git)
- **Metrics JSON:** `checkpoints/stage1_metrics.json` (local)

These are prototype pipeline numbers, not publication claims.
