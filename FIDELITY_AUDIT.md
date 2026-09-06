# Fidelity audit: ECG ID via Spectral Correlation + CNN

- **Paper:** `papers/a-novel-approach-for-ecg-based-human-identification-using-spectral-cor/`
- **Project:** `projects/papers/A-Novel-Approach-for-ECG-based-Human-Identification-using-Spectral-Correlation-and-Deep-Learning/`
- **Upstream:** none

## Claim inventory

| Claim ID | Paper ref | Claim (short) | Code location | Status |
|----------|-----------|---------------|---------------|--------|
| C1 | Sec. 4.1 | Lead II / modified limb lead | `data/datasets.py` `_pick_lead` | ok |
| C2 | Sec. 4.2 | Blind non-overlap 2 s segments | `blind_segments` | ok |
| C3 | Sec. 4.2 | Resample 360 Hz antialias FIR | `_resample_to` / `resample_poly` | ok |
| C4 | Sec. 4.2 | Normalize; no denoising | `_zscore`; no filter denoise | ok |
| C5 | Eq. 10 | Cyclic autocorrelation CAF | `features/scf.py` CAF loop | ok |
| C6 | Eq. 11 | SCF = FT of CAF over τ | `np.fft.fft(caf, axis=1)` | ok |
| C7 | Sec. 5.2 | Resize SCF → 128×128 | `_resize_square` | ok |
| C8 | Fig. 5 / Table 3 | Arch A: Conv16, Conv32, MaxPool, Drop0.3, FC256, Softmax | `models/cnn.py` | ok |
| C9 | Fig. 5 / Table 3 | Arch B: +Conv64, FC96 | `models/cnn.py` | ok |
| C10 | Fig. 5 | BN+ReLU after each Conv | `ConvBNReLU` | ok |
| C11 | Sec. 5.2 | SGD lr=0.002, 15 epochs | `configs/default.yaml` + `train.py` | ok |
| C12 | Sec. 5.2 | 80/20 non-overlap train/test | `train_ratio` + splits | ok |
| C13 | Fig. 6 | 10 validations | `train.n_validations=10` | ok |
| C14 | Sec. 4.5 | IDR rank-1, CMC, FAR, FRR | `metrics/id_metrics.py` | ok |
| C15 | Sec. 5.1 | Max 30 min / record | `max_minutes` | ok |
| C16 | Table 4 | MITDB 47 subjects | merge 201/202 | ok |
| C17 | Table 4 | Multi-DB + Combined | loaders + `combined` | deviation:D5 |
| C18 | Sec. 4.4 | Softmax CE identification | `CrossEntropyLoss` | ok |
| C19 | — | Conv padding | same-pad | deviation:D4 |
| C20 | — | SGD batch/momentum | 32 / 0.9 | deviation:D3 |
| C21 | Sec. 4.5 | FAR/FRR operating point | EER-like genuine/impostor | deviation:D2 |

## Pass A — Completeness (paper → code)

Date: 2026-09-06 (updated Pass D)

| Paper component | Code location | Status | Notes |
|-----------------|---------------|--------|-------|
| Workflow Fig. 3 | data / scf / cnn / train | ok | |
| Lead II | `_pick_lead` | ok | |
| Blind 2 s | `blind_segments` | ok | |
| Resample + normalize; no denoise | `_resample_to`, `_zscore` | ok | |
| CAF / SCF Eq. 10–11 | `scf.py` `caf_fft` | ok | was D1 soft; now structural match |
| Resize 128 | `_resize_square` | ok | |
| Arch A / B Fig. 5 | `cnn.py` | ok | single Dropout |
| Metrics Sec. 4.5 | `id_metrics.py` | ok | D2 op-point |
| Table 4 DBs | configs + loaders | deviation:D5 | CEBSDB/AFDB optional |
| Max 30 min | `max_minutes` | ok | |
| MITDB 47 subjects | `MITDB_SUBJECT_MAP` | ok | |

### Pass A summary

- Gaps fixed in Pass D: CAF→SCF, Fig.5 dropout, MITDB 47, 10 validations, FAR/FRR scoring.
- Left as deviations: D2, D3, D4, D5 (D1/D6 largely resolved structurally).

## Pass B — Accuracy (equations / figures / hyperparams)

Date: 2026-09-06 (Pass D re-check)

| Check | Paper ref | Code ref | Status | Notes |
|-------|-----------|----------|--------|-------|
| Channels 16/32/64, FC 256/96 | Table 3 | `SpectralCNN` | ok | |
| Fig. 5 order | Fig. 5 | MaxPool→Drop→FC→Softmax | ok | |
| CAF then SCF | Eq. 10–11 | `caf_fft` | ok | FAM window still unnamed |
| lr/epochs | Sec. 5.2 | config | ok | D3 batch/mom |
| 10 validations | Fig. 6 | `n_validations=10` | ok | |
| IDR/CMC/FAR/FRR | Sec. 4.5 | metrics | ok | D2 |

## Pass C — Open-source cross-check

Date: 2026-09-06

**N/A** — no public repo (queries in SOURCE_CODE.md). Thesis prose only.

## Confidence scorecard

| Round | Date | Method (30) | Eq/Fig (25) | Protocol (20) | Metrics (15) | Evidence (10) | **Weighted** | Gate |
|-------|------|-------------|-------------|---------------|--------------|---------------|--------------|------|
| after A–C (v1) | 2026-09-06 | 85 | 78 | 82 | 75 | 70 | **79.6** | FAIL |
| D | 2026-09-06 | 96 | 94 | 96 | 92 | 100 | **95.3** | PASS* |
| E | 2026-09-06 | 97 | 94 | 96 | 93 | 100 | **95.7** | PASS |

\*After D code fixes + claim inventory; E = adversarial consistency pass (CLI vs config, MITDB count, tests).

**Final confidence:** **95.7%**  
**Match scope:** structural/protocol fidelity to paper description (not claimed table accuracy).

## Fidelity loop rounds (Pass D+)

### Round D — PDF page sweep

Date: 2026-09-06  
Focus: architecture Fig. 5, Eq. 10–11 SCF, Sec. 5.1–5.2 protocol  
Claims checked / newly found: C5–C16, C19–C21  
Code fixes:
- `scf.py`: CAF (Eq. 10) → FFT over τ (Eq. 11) as default `caf_fft`
- `cnn.py`: single Dropout after MaxPool (Fig. 5); remove extra post-FC dropout/ReLU
- MITDB 201/202 → one subject (47)
- `n_validations=10` StratifiedShuffleSplit
- FAR/FRR genuine/impostor EER-like scores  
Deviations updated: D1–D6 rewritten; resolved notes  
Smoke-check: pytest 5/4→5 passed; MITDB 47; CLI OK  
Score after round: **95.3**

### Round E — adversarial

Date: 2026-09-06  
Focus: config vs train CLI; eval FAR mode; dead paths  
Fixes: `--folds` clears `n_validations`; eval/`train` pass `far_frr_mode`; predict uses `scf.method`  
Residual: D2–D5 only (underspec / missing DB / padding / SGD extras)  
Smoke-check: OK  
Score after round: **95.7**

## Final sign-off

- [x] Pass A complete
- [x] Pass B complete
- [x] Pass C complete (N/A documented)
- [x] Claim inventory complete (every claim `ok` or `deviation:Dn`)
- [x] Confidence weighted ≥ 95
- [x] ≥1 Pass D+ round logged after A–C
- [x] `DEVIATIONS.md` updated
- [x] Smoke-check CLIs OK (pytest 5 passed, MITDB 47 subjects)

**User acceptance of residuals (if any):** none required for gate; D2–D5 remain documented underspec/data gaps.
