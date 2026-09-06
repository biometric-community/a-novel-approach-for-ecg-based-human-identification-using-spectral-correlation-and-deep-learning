# Implementation plan: ECG ID via Spectral Correlation + CNN

- **Paper:** `papers/a-novel-approach-for-ecg-based-human-identification-using-spectral-cor/`
- **Project:** `projects/papers/A-Novel-Approach-for-ECG-based-Human-Identification-using-Spectral-Correlation-and-Deep-Learning/`
- **Stack:** Python 3.10+ / PyTorch
- **Upstream:** none (see SOURCE_CODE.md)
- **Package:** `ecg_scf`

## Components to build

1. **Preprocess (Sec. 4.2)** — resample to 360 Hz (antialias FIR), z-normalize; **no** denoising / fiducials; blind non-overlapping **2 s** segments (720 samples).
2. **SCF feature (Sec. 4.3, Eq. 10–11)** — cyclic autocorrelation → spectral correlation image; resize to **128×128**.
3. **CNN Arch A / Arch B (Sec. 4.4, Fig. 5, Table 3)** — Conv(5×5@16/32[/64]) + BN + ReLU; MaxPool 2×2; Dropout 0.3; FC 256 (A) or 96 (B); Softmax(#subjects).
4. **Train** — SGD lr=0.002, 15 epochs, CE loss; 5-fold 80/20 non-overlapping.
5. **Metrics (Sec. 4.5)** — IDR (rank-1), CMC (rank-k), FAR, FRR.

## Data

| Paper DB | Local path | Notes |
|----------|------------|-------|
| Fantasia | `projects/datasets/fantasia/data/` | on disk |
| STDB | `projects/datasets/stdb/data/` | on disk |
| VFDB | `projects/datasets/vfdb/data/` | on disk |
| NSRDB | `projects/datasets/nsrdb/` | on disk |
| MITDB | `projects/datasets/mit-bih/` | on disk |
| PTBDB | `projects/datasets/ptb/` | on disk |
| CEBSDB | `projects/datasets/cebsdb/` | placeholder |
| AFDB | `projects/datasets/afdb/` | placeholder |
| Combined | union of available | 488 when all present |

Lead II / first usable channel; max **30 min** per record (paper Sec. 5.1).

## Training protocol

- Loss: cross-entropy (softmax ID)
- Optimizer: SGD, lr=0.002
- Epochs: 15; input: 1×128×128 SCF image
- Primary metric: IDR; also FAR/FRR/CMC

## Fidelity plan

- Pass A: workflow Fig. 3 + Arch A/B + metrics
- Pass B: Eq. 10–11 SCF; Table 3 layers; hyperparams
- Pass C: N/A (no public code)

## Results figure inventory (paper Results → our plots)

| Paper fig | Type | Metrics / axes | Our path |
|-----------|------|----------------|----------|
| Fig. 6 | boxplot | IDR across validations | `outputs/figures/idr_boxplot.svg` |
| Fig. 7 | grouped bars | IDR / FAR / FRR | `outputs/figures/metrics_bars.svg` |
| Fig. 8 | CMC lines | rank → IDR | `outputs/figures/cmc.svg` |

## Reporting

- `ecg_scf/plot_style.py` (dev-plot) + `python -m ecg_scf.report` / `scripts/report.sh`
- Deliverable: `REPORT.md`

## Risks / unknowns

- Exact SCF estimator / frequency smoothing not named (FAM vs direct FFT product) → implement Eq. 10–11 FFT form; document D1.
- SGD momentum / batch size / weight decay not stated → defaults in config + DEVIATIONS.
- FAR/FRR operating point underspecified for closed-set softmax → threshold on max posterior (D2).

## Deviation seeds

- D1 SCF numerical recipe
- D2 FAR/FRR threshold protocol
- D3 batch size / SGD momentum
- D4 missing CEBSDB/AFDB until downloaded
