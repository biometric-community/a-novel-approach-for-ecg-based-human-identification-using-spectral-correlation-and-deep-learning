# Experiment report: ECG SCF + CNN

- **Paper:** `a-novel-approach-for-ecg-based-human-identification-using-spectral-cor`
- **Project:** ECG spectral-correlation identification
- **DOI:** 10.1109/TBIOM.2019.2947434
- **Generated from:** `outputs/logs/train_summary.json`
- **Fidelity:** see [FIDELITY_AUDIT.md](./FIDELITY_AUDIT.md) (method match; not accuracy claims)
- **Plot style:** [dev-plot](../../../.cursor/skills/dev-plot/SKILL.md)

## Summary

| Metric | Ours | Paper Table 5 (Arch A Mean All) | Notes |
|--------|------|----------------------------------|-------|
| IDR | 0.8825 | 0.956 | Fuller run |
| FAR | 0.0290 | 0.022 | FAR/FRR mode may differ (D2) |
| FRR | 0.0290 | 0.001 | |

Database=`fantasia`, arch=`arch_a`, classes=`40`, folds/validations=`10`.

Numbers are from our local protocol; compare carefully to paper subsets.

## Experimental setup (ours)

- Lead preference / resample: Lead II-ish -> 360 Hz, 2 s blind segments, max 30 min
- SCF: `caf_fft` (Eq. 10–11)
- Train: SGD lr=0.002, epochs=15, batch=32
- Validations: `n_validations`=10, `n_folds`=5

## Results tables

### Per-fold metrics

| Fold | IDR | FAR | FRR |
|------|-----|-----|-----|
| 0 | 0.9375 | 0.0175 | 0.0175 |
| 1 | 0.9225 | 0.0197 | 0.0200 |
| 2 | 0.9375 | 0.0225 | 0.0225 |
| 3 | 0.8850 | 0.0300 | 0.0300 |
| 4 | 0.8775 | 0.0325 | 0.0325 |
| 5 | 0.8750 | 0.0300 | 0.0300 |
| 6 | 0.8225 | 0.0500 | 0.0500 |
| 7 | 0.7425 | 0.0500 | 0.0500 |
| 8 | 0.8925 | 0.0225 | 0.0225 |
| 9 | 0.9325 | 0.0150 | 0.0150 |

## Figures

### Figure: `cmc`

![CMC curve (paper Fig. 8 style). Ours from logged folds.](outputs/figures/cmc.svg)

*CMC curve (paper Fig. 8 style). Ours from logged folds.*

### Figure: `metrics_bars`

![IDR / FAR / FRR bars (paper Fig. 7 style). Paper bars = Table 5 Mean All Arch A.](outputs/figures/metrics_bars.svg)

*IDR / FAR / FRR bars (paper Fig. 7 style). Paper bars = Table 5 Mean All Arch A.*

### Figure: `idr_boxplot`

![IDR boxplot across validations (paper Fig. 6 style).](outputs/figures/idr_boxplot.svg)

*IDR boxplot across validations (paper Fig. 6 style).*

## Comparison notes

- Paper Fig. 6–8 are **Results** plots; we regenerate the same *types* from our JSON logs with **dev-plot** styling.
- This Fantasia run used `--max-segments 50` (caf_fft SCF build is costly at full 30 min); protocol kept `n_validations=10`, epochs=15.
- Gap vs paper Table 5 is expected without Combined-488 / CEBSDB and with segment caps.
- See [DEVIATIONS.md](./DEVIATIONS.md) for D2–D7.

## How to regenerate

```bash
# from this project directory
$env:PYTHONPATH = (Get-Location).Path   # PowerShell
python -m ecg_scf.train --database fantasia --max-segments 30   # or full protocol
python -m ecg_scf.report --config configs/default.yaml
# or: bash scripts/report.sh
```

## Artifacts

| Path | Role |
|------|------|
| `outputs/figures/*.svg` | Result figures |
| `outputs/logs/train_summary.json` | Metrics |
| `FIDELITY_AUDIT.md` | Method fidelity |
| `DEVIATIONS.md` | Intentional gaps |
