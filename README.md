"""ECG Spectral Identification via Spectral Correlation + Deep Learning

Reproduction of **A Novel Approach for ECG-based Human Identification using Spectral Correlation and Deep Learning** (Abdeldayem & Bourlai, IEEE TBIOM 2019).

PDF: [`papers/.../paper.pdf`](../../../papers/a-novel-approach-for-ecg-based-human-identification-using-spectral-cor/paper.pdf) · Analysis: [`analysis.md`](../../../papers/a-novel-approach-for-ecg-based-human-identification-using-spectral-cor/analysis.md) · DOI: [10.1109/TBIOM.2019.2947434](https://doi.org/10.1109/TBIOM.2019.2947434)

## Method (short)

Blind **2 s** ECG segments (no fiducials / no denoising) → **spectral correlation (SCF)** images via CAF (Eq. 10) then FT over lag (Eq. 11) → resize **128×128** → **2D CNN** (Arch A preferred; Arch B deeper) with softmax over subjects. Default protocol: **10** stratified 80/20 validations (Fig. 6). Metrics: **IDR**, **FAR**, **FRR**, **CMC**.

## Setup

```bash
# Git Bash / WSL / Linux / macOS — from repo root
bash projects/papers/A-Novel-Approach-for-ECG-based-Human-Identification-using-Spectral-Correlation-and-Deep-Learning/scripts/setup_env.sh
```

Windows PowerShell (manual):

```powershell
cd projects/papers/A-Novel-Approach-for-ECG-based-Human-Identification-using-Spectral-Correlation-and-Deep-Learning
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = (Get-Location).Path
```

## Data

| Paper dataset | Local path | Notes |
|---------------|------------|-------|
| Fantasia | `projects/datasets/fantasia/data/` | on disk |
| STDB | `projects/datasets/stdb/data/` | on disk |
| VFDB | `projects/datasets/vfdb/data/` | on disk |
| NSRDB | `projects/datasets/nsrdb/` | on disk |
| MITDB | `projects/datasets/mit-bih/` | on disk |
| PTBDB | `projects/datasets/ptb/` | on disk |
| CEBSDB / AFDB | `projects/datasets/cebsdb/`, `afdb/` | download scripts available |
| Combined | set `data.database: combined` | uses `data.databases` list |

Edit `configs/default.yaml` → `data.database` (default `fantasia`). If a DB is missing and `allow_synthetic: true`, a small synthetic cohort is used so CLIs still run.

Lead II preferred; signals resampled to **360 Hz**, max **30 min**/record (Sec. 5.1).

## Train

```bash
bash projects/papers/A-Novel-Approach-for-ECG-based-Human-Identification-using-Spectral-Correlation-and-Deep-Learning/scripts/train.sh
# Faster SCF estimate for smoke (optional):
# edit configs/default.yaml → scf.method: bifrequency
# Arch B, single fold smoke:
bash .../scripts/train.sh --arch arch_b --folds 1 --max-segments 20 --epochs 2
```

## Evaluate

```bash
bash .../scripts/eval.sh --checkpoint outputs/checkpoints/best.pt
```

## Report (figures + REPORT.md)

Uses **dev-plot** (`ecg_scf/plot_style.py`). Writes SVGs under `outputs/figures/` and refreshes [REPORT.md](./REPORT.md) (CMC ≈ Fig. 8, metric bars ≈ Fig. 7, IDR boxplot ≈ Fig. 6).

```bash
bash .../scripts/report.sh
# PowerShell:
$env:PYTHONPATH = (Get-Location).Path
python -m ecg_scf.report --config configs/default.yaml
```

## Predict

```bash
bash .../scripts/predict.sh --input /path/to/record_stem_or_dir --checkpoint outputs/checkpoints/best.pt
```

## Outputs

- Checkpoints: `outputs/checkpoints/`
- Logs: `outputs/logs/`
- Figures: `outputs/figures/` (SVG, dev-plot)
- Predictions: `outputs/predictions/`
- Experiment write-up: [REPORT.md](./REPORT.md)

## Source code

Upstream inquiry: [SOURCE_CODE.md](./SOURCE_CODE.md) (no public author code found).

## Fidelity

[FIDELITY_AUDIT.md](./FIDELITY_AUDIT.md) · [DEVIATIONS.md](./DEVIATIONS.md) · [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)

## Citation

```bibtex
@article{abdeldayem2019ecg,
  title={A Novel Approach for ECG-based Human Identification using Spectral Correlation and Deep Learning},
  author={Abdeldayem, Sara S. and Bourlai, Thirimachos},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year={2019},
  doi={10.1109/TBIOM.2019.2947434}
}
```
