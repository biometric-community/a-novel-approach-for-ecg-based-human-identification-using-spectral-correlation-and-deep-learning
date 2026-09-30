# Experiment report: ECG SCF + CNN

- **Paper:** `a-novel-approach-for-ecg-based-human-identification-using-spectral-cor`
- **Project:** ECG spectral-correlation identification
- **DOI:** 10.1109/TBIOM.2019.2947434
- **Generated from:** `outputs/logs_cebsdb_full/train_summary.json`
- **Fidelity:** see [FIDELITY_AUDIT.md](./FIDELITY_AUDIT.md) (method match; not accuracy claims)
- **Plot style:** [dev-plot](../../../.cursor/skills/dev-plot/SKILL.md)

## Summary

| Metric | Ours | Paper Table 5 (Arch A Mean All) | Notes |
|--------|------|----------------------------------|-------|
| IDR | 0.9875 | 0.956 | Fuller run |
| FAR | 0.0034 | 0.022 | FAR/FRR mode may differ (D2) |
| FRR | 0.0034 | 0.001 | |

Database=`cebsdb`, arch=`arch_a`, classes=`20`, folds/validations=`10`.

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
| 0 | 0.9889 | 0.0029 | 0.0028 |
| 1 | 0.9894 | 0.0031 | 0.0031 |
| 2 | 0.9856 | 0.0039 | 0.0039 |
| 3 | 0.9872 | 0.0039 | 0.0039 |
| 4 | 0.9869 | 0.0042 | 0.0042 |
| 5 | 0.9861 | 0.0036 | 0.0036 |
| 6 | 0.9914 | 0.0033 | 0.0033 |
| 7 | 0.9858 | 0.0033 | 0.0033 |
| 8 | 0.9881 | 0.0026 | 0.0025 |
| 9 | 0.9858 | 0.0035 | 0.0033 |

## Figures

### Figure: `cmc`

![CMC curve (paper Fig. 8 style). Ours from logged folds.](outputs/figures_cebsdb_full/cmc.svg)

*CMC curve (paper Fig. 8 style). Ours from logged folds.*

### Figure: `metrics_bars`

![IDR / FAR / FRR bars (paper Fig. 7 style). Paper bars = Table 5 Mean All Arch A.](outputs/figures_cebsdb_full/metrics_bars.svg)

*IDR / FAR / FRR bars (paper Fig. 7 style). Paper bars = Table 5 Mean All Arch A.*

### Figure: `idr_boxplot`

![IDR boxplot across validations (paper Fig. 6 style).](outputs/figures_cebsdb_full/idr_boxplot.svg)

*IDR boxplot across validations (paper Fig. 6 style).*

## Comparison notes

- Primary reported run is usually **Combined** (~485 classes vs paper 488; see D5 / AFDB).
- Sibling logs: `outputs/logs_cebsdb/` (CEBSDB), `outputs/logs/train_fantasia_ms50.log` (Fantasia capped), `outputs/logs_combined/` (Combined).
- Paper Fig. 6–8 are **Results** plots; we regenerate the same *types* from our JSON logs with **dev-plot** styling.
- `--max-segments` caps trade coverage for SCF build time; keep `n_validations=10` when matching Fig. 6.
- Gap vs paper Table 5 may remain with subject shortfall, segment caps, and FAR/FRR mode D2.
- See [DEVIATIONS.md](./DEVIATIONS.md) for D2–D7.

## How to regenerate

```bash
# from this project directory
source .venv/bin/activate
export PYTHONPATH=.
python -m ecg_scf.report --config configs/default.yaml \
  --logs outputs/logs_combined --figdir outputs/figures_combined --summary train_summary.json
```

## Artifacts

| Path | Role |
|------|------|
| `outputs/figures_combined/*.svg` | Combined figures (when using logs_combined) |
| `outputs/logs_combined/train_summary.json` | Combined metrics |
| `outputs/logs_cebsdb/` | CEBSDB run |
| `FIDELITY_AUDIT.md` | Method fidelity |
| `DEVIATIONS.md` | Intentional gaps |
