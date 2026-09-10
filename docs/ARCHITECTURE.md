# FetalSense — Locked Architecture & Training Recipe

**Status:** design locked for PMEA track (2026-09-10)  
**Principle:** three pillars must each be ablatable — SSL, SQI conditioning, Transformer backbone.

---

## 1. Problem formulation

**Input:** multi-channel abdominal ECG window  
\( X \in \mathbb{R}^{C \times T} \) with \( C \in \{1..4\} \) (zero-pad / mask missing leads), \( T \) samples after resampling to **250 Hz**, window **2.048 s** → \( T = 512 \) (or 4.096 s / 1024 if GPU allows; default **512**).

**Outputs (fine-tune):**
1. **Extracted fetal ECG** \( \hat{y} \in \mathbb{R}^{T} \) (single-channel recovered fetal waveform).
2. **Fetal QRS likelihood** \( \hat{p} \in [0,1]^{T} \) (per-sample; peaks → beat times).
3. **Optional SQI** \( \hat{q} \in [0,1]^{C} \) (per-channel quality) when using learned quality head.

**Inference:** quality-aware channel fusion → \( \hat{y}, \hat{p} \) → refractory peak-pick → FHR/RR.

---

## 2. Preprocessing (shared)

1. Bandpass **3–90 Hz** (FIR/Butterworth), optional 50/60 Hz notch.
2. Resample to **250 Hz**; z-score **per channel per window** (train stats optional for synth).
3. Sliding windows: train stride 0.5–1.0 s; eval stride 0.25 s with overlap-add for \( \hat{y} \) / max for \( \hat{p} \).
4. Channel mask \( m \in \{0,1\}^{C} \) for missing leads.
5. QRS soft targets: Gaussian bumps (σ ≈ 20–30 ms) on reference fetal R (or V-peak on NInFEA, **flagged**).

---

## 3. Module A — Signal Quality (SQI)

**Two modes (both implemented; paper reports learned + classic features):**

### A1. Classic SQI features (fast baseline)
Per channel, per window: kurtosis, spectral power ratio in fetal band (~1.5–3.5 Hz FHR-related / QRS band energy), baseline wander index, residual after maternal template hint (optional light). Concatenate → MLP → \( q_c \in (0,1) \).

### A2. Learned SQI head
Small 1D CNN on each channel independently → global pool → sigmoid \( q_c \). Can be pretrained with weak labels (high agreement across detectors / synth SNR) or jointly.

### How SQI enters the model (required for “quality-aware”)
Not post-hoc channel pick alone. Use **all three** (ablate combinations):
1. **Soft channel weights:** \( X'_c = q_c \cdot X_c \) (broadcast in time).
2. **Quality tokens:** project \( q \) to a length-\( C \) embedding prepended/appended to the Transformer sequence.
3. **Loss weighting:** down-weight low-\( q \) channels / windows in SSL and fine-tune losses.

**Ablation “w/o SQI”:** set \( q_c = 1 \), drop quality tokens and loss weights.

---

## 4. Module B — Tokenizer + Transformer backbone

### Patch embedding
- Multi-channel patches: width \( P = 16 \) samples (64 ms @ 250 Hz), stride \( P \) (non-overlap) or \( P/2 \).
- Linear proj: flatten \( C \times P \) (after SQI weighting) → \( d \)-dim token.  
- Default **\( d = 128 \)** (light student model); scale-up **256** if needed.
- Sequence length \( N \approx T/P \) (+ quality tokens + optional CLS).
- Additive **learned positional** encodings + **channel-type** embedding if using per-lead tokens (alt design: keep fused multi-lead patches as default).

**Default tokenization (locked):** fused multi-lead patches (SQI-weighted channels stacked in the patch vector). Simpler, matches abdominal array as one view.

### Encoder
- **L = 6** Transformer encoder layers  
- **H = 4** heads, FFN dim **4d**, GELU, dropout **0.1**  
- Pre-LN

### Decoder (extraction)
- Lightweight **4-layer** Transformer decoder **or** (preferred for speed) **1D ConvTranspose / U-Net-style decoder** attending to encoder memory via cross-attention **once** (hybrid):  
  **Locked choice:** Encoder (Transformer) + **cross-attention Conv decoder** to map tokens → \( T \)-sample waveform (W-NETR-adjacent but SSL+SQI differentiated).

**Ablation “w/o Transformer”:** replace encoder with **1D CNN / BiLSTM** of matched parameter count (~same \( d \), depth); keep decoder/heads.

---

## 5. Module C — Task heads

### C1. Extraction head
Decoder output → 1×Conv1d → \( \hat{y} \). Supervised when fetal waveform GT exists (FECGSYNDB components; optional direct scalp alignment on ADFECGDB/B2).

### C2. QRS head
On encoder CLS/mean pool **or** upsampled token stream → Conv1d → sigmoid \( \hat{p}_t \). Prefer **temporal upsampling** so peaks stay time-aligned.

### C3. SQI head (if learned)
Parallel branch from raw/light CNN (does not see labels of QRS during pure SSL quality pretrain).

---

## 6. Module D — Self-supervised pretraining

**Locked objective: Temporal + Instance contrastive (CLOCS-inspired, abdomen-adapted)**

For each window, form views:
- **T:** time-adjacent crop / mild time-shift  
- **C:** channel dropout / SQI-stochastic masking  
- **Aug:** noise, amplitude scale, mild band-limit

Encoder → projected embedding \( z \) (MLP proj dim 64–128).  
Loss: NT-Xent / InfoNCE on positive pairs (same recording window under different views); temperature τ = 0.1.

**Optional auxiliary (if stable):** masked patch reconstruction on 15–25% tokens (MAE-lite) — include only if contrastive alone underperforms; report in ablation appendix.

**No QRS / fetal waveform labels in SSL.**

**Ablation “w/o SSL”:** random init → supervised fine-tune only.

---

## 7. Losses (fine-tune)

\[
\mathcal{L} = \lambda_y \mathcal{L}_{\text{ext}} + \lambda_p \mathcal{L}_{\text{qrs}} + \lambda_q \mathcal{L}_{\text{sqi}}
\]

| Term | Definition | When |
|------|------------|------|
| \( \mathcal{L}_{\text{ext}} \) | \( 1 - \mathrm{Pearson}(y,\hat{y}) \) + 0.5·MSE | waveform GT |
| \( \mathcal{L}_{\text{qrs}} \) | BCE with logits on soft Gaussian targets; optional Dice on peaks | fQRS / V-peak labels |
| \( \mathcal{L}_{\text{sqi}} \) | BCE on quality weak labels / consistency | if learned SQI |

**Defaults:** \( \lambda_p = 1.0 \), \( \lambda_y = 0.5 \) (0 if no waveform GT in batch), \( \lambda_q = 0.1 \).  
**SQI loss weight on samples:** multiply \( \mathcal{L}_{\text{qrs}} \) by mean \( q \) of kept channels (quality-aware).

**Peak inference:** local maxima of \( \hat{p} \) above threshold τ (tuned on val), refractory **150–200 ms** (fetal).

**Match tolerance (reporting):** primary **50 ms** on ADFECG-style; also report **100 ms** for Challenge-2013 comparability.

---

## 8. Training recipe (staged)

### Stage 0 — Data sanity
Reproduce classical / open scoring on B1/B2 subset; fix loaders.

### Stage 1 — Supervised QRS-only prototype (1–3 days compute)
- Encoder + QRS head; no SSL; SQI off or classic only.  
- Optimizer: AdamW, lr \( 3\times10^{-4} \), weight decay 0.01, batch 32–64, cosine 50–100 epochs, early stop on LOSO val F1.  
- Amp (fp16) OK.

### Stage 2 — Add SQI conditioning
- Enable soft weights + quality tokens; verify ΔF1 on low-quality strata.

### Stage 3 — SSL pretrain
- 100–200 epochs on unlabeled pools (FECGSYNDB + NIFECGDB + NInFEA channels).  
- lr \( 1\times10^{-3} \) → cosine; large batch if possible (grad accum).  
- Freeze nothing yet; save encoder weights.

### Stage 4 — Multi-task fine-tune
- Load SSL encoder; add extraction + QRS; SQI on.  
- lr \( 1\times10^{-4} \) encoder, \( 3\times10^{-4} \) heads; 50–100 epochs.  
- Curriculum optional: high-q windows first 10 epochs.

### Stage 5 — Full eval matrix
- LOSO / leave-one-dataset-out; ablations; baselines; seeds ×3.

**Hardware assumption:** single consumer GPU (8–24 GB). Keep \( d=128 \), \( L=6 \), \( T=512 \) as default footprint.

---

## 9. Parameterization summary (locked defaults)

| Item | Value |
|------|-------|
| Sample rate | 250 Hz |
| Window | 512 samples (2.048 s) |
| Patch | 16 |
| \( d \) | 128 |
| Encoder layers | 6 |
| Heads | 4 |
| Decoder | Cross-attn conv upsample → waveform |
| SSL | CLOCS-like NT-Xent (temporal + channel views) |
| Fine-tune | Pearson+MSE + BCE/Dice QRS + optional SQI BCE |
| Peak refractory | 180 ms |
| Primary F1 tolerance | 50 ms (also report 100 ms) |

---

## 10. Ablation matrix (must ship)

| Model | SSL | SQI cond. | Backbone |
|-------|-----|-----------|----------|
| FetalSense (full) | ✓ | ✓ | Transformer |
| −SSL | ✗ | ✓ | Transformer |
| −SQI | ✓ | ✗ | Transformer |
| −Transformer | ✓ | ✓ | CNN/BiLSTM |
| Supervised CNN-only | ✗ | ✗ | CNN |

---

## 11. What we will *not* claim without evidence

- Challenge Set B/C scores without official labels.  
- NInFEA “fQRS” without stating V-peak protocol.  
- Subject-independent results using NIFECGDB as a multi-subject set.  
- Clinical outcome improvement (methods paper).

---

## 12. Implementation map (next coding step)

```
fetalsense/
  data/          # loaders: challenge2013, matonia, fecgsyndb, ninfea, adfecgdb
  models/
    sqi.py
    tokenizer.py
    transformer.py
    heads.py
    fetalsense.py
  ssl/contrastive.py
  train/{pretrain,finetune,eval}.py
  metrics/fqrs.py
  configs/default.yaml
```
