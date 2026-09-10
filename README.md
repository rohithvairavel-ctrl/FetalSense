# FetalSense

Quality-aware Transformer for **fetal ECG extraction** and **fetal QRS (fQRS) detection** from multi-channel abdominal ECG, with CLOCS-inspired self-supervised pretraining and explicit signal-quality (SQI) conditioning.

**Repo:** [rohithvairavel-ctrl/FetalSense](https://github.com/rohithvairavel-ctrl/FetalSense)

## Paper goal

Build an ablatable, methods-focused pipeline for non-invasive fetal monitoring that cleanly separates three pillars:

1. **SSL** — temporal + channel contrastive pretraining (no QRS / waveform labels)
2. **SQI conditioning** — soft channel weights, quality tokens, and loss weighting
3. **Transformer backbone** — patch tokenizer + Pre-LN encoder, vs CNN / BiLSTM ablations

Primary reporting: fQRS **Sensitivity / PPV / F1** at **50 ms** and **100 ms** match tolerances (refractory peak-pick ≈ 180 ms). Extraction quality via Pearson + MSE when waveform GT exists. See `docs/ARCHITECTURE.md` for the locked design.

> This repository ships a **working training scaffold** (CPU-safe forward passes, losses, metrics, loaders). It does **not** claim Challenge leaderboard scores or clinical outcomes without completed experiments.

## Install

```bash
cd code   # or clone root
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python ≥ 3.10. Torch installs CPU wheels by default on most platforms; CUDA is optional.

## Data download

PhysioNet corpora (Challenge 2013, ADFECGDB, FECGSYNDB, …) are **not** bundled.

```bash
export FETALSENSE_DATA=/workspace/fetalsense/data/downloads   # or your path
bash scripts/download_physionet.sh
```

Update `configs/default.yaml` → `paths.*` to match. Loaders for Challenge-2013 and ADFECGDB sketch `wfdb` reading when files are present; Matonia / FECGSYNDB / NInFEA are documented stubs. Missing data must not break unit tests.

### PhysioNet citation

Please cite PhysioNet / PhysioBank when using these databases:

> Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. *Circulation* 101(23):e215–e220, 2000.

Also cite each individual database paper (Challenge 2013, ADFECGDB, FECGSYNDB, NInFEA, etc.) as required by PhysioNet terms.

## Locked defaults

| Item | Value |
|------|-------|
| Input | \(X \in \mathbb{R}^{C \times T}\), \(C \le 4\), 250 Hz, \(T=512\) |
| Patch / model | \(P=16\), \(d=128\), \(L=6\), \(H=4\), Pre-LN, dropout 0.1 |
| SSL | NT-Xent, \(\tau=0.1\), temporal + channel views |
| Peak refractory | 180 ms |
| Metrics | Sens / PPV / F1 @ 50 ms & 100 ms |

Config flags: `use_sqi`, `backbone: transformer|cnn|bilstm`, `paths.data_root`, …

## Train stages

```bash
# Stage 3 — SSL dry-run (synthetic)
python -m fetalsense.train.pretrain --config configs/default.yaml --steps 5

# Stage 4 — fine-tune dry-run
python -m fetalsense.train.finetune --config configs/default.yaml --steps 5

# Eval smoke
python -m fetalsense.train.eval --config configs/default.yaml --synthetic
```

Recommended recipe (see architecture doc):

0. Data sanity / classical baselines on public sets  
1. Supervised QRS-only prototype (`use_sqi: false`, no SSL)  
2. Enable SQI conditioning  
3. SSL pretrain on unlabeled pools  
4. Multi-task fine-tune (Pearson+MSE extraction, BCE+Dice QRS, optional SQI BCE)  
5. LOSO / leave-one-dataset-out + ablation matrix  

## Ablations

| Model | SSL | SQI | Backbone |
|-------|-----|-----|----------|
| FetalSense (full) | ✓ | ✓ | Transformer |
| −SSL | ✗ | ✓ | Transformer |
| −SQI | ✓ | ✗ | Transformer |
| −Transformer | ✓ | ✓ | CNN / BiLSTM |
| Supervised CNN-only | ✗ | ✗ | CNN |

Toggle via `configs/default.yaml` (`use_sqi`, `backbone`) or CLI overrides when wiring full trainers.

## Tests

```bash
pip install -e ".[dev]" -q
pytest -q
```

## Layout

```
configs/default.yaml
docs/ARCHITECTURE.md
scripts/download_physionet.sh
src/fetalsense/
  data/          # Challenge-2013, ADFECGDB, stubs
  models/        # SQI, tokenizer, transformer, heads, baselines
  ssl/           # NT-Xent contrastive
  metrics/       # fQRS Sens/PPV/F1
  train/         # pretrain, finetune, eval
  utils/         # preprocess, peaks, seed
tests/
```

## License

MIT — see `LICENSE`.
