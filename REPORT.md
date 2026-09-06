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
| IDR | 0.1200 | 0.956 | **Smoke / incomplete protocol** |
| FAR | 0.5000 | 0.022 | FAR/FRR mode may differ (D2) |
| FRR | 0.5000 | 0.001 | |

Database=`__smoke__`, arch=`arch_a`, classes=`8`, folds/validations=`5`.

This report is from a **smoke** run (synthetic or capped protocol). Do not treat metrics as a paper reproduction.

## Experimental setup (ours)

- Lead preference / resample: Lead II-ish -> 360 Hz, 2 s blind segments, max 30 min
- SCF: `bifrequency` (Eq. 10–11)
- Train: SGD lr=0.002, epochs=5, batch=16
- Validations: `n_validations`=5, `n_folds`=5

## Results tables

### Per-fold metrics

| Fold | IDR | FAR | FRR |
|------|-----|-----|-----|
| 0 | 0.1000 | 0.4500 | 0.4500 |
| 1 | 0.1500 | 0.5500 | 0.5500 |
| 2 | 0.1000 | 0.5000 | 0.5000 |
| 3 | 0.1500 | 0.5000 | 0.5000 |
| 4 | 0.1000 | 0.5000 | 0.5000 |

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
- Gap vs paper Table 5 is expected on smoke data and when CEBSDB/AFDB / full 10 validations are not run.
- See [DEVIATIONS.md](./DEVIATIONS.md) for D2–D5.

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
