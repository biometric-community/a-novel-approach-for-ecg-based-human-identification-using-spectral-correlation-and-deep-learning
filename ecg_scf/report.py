"""Generate Results-style figures (dev-plot) and REPORT.md from train/eval JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecg_scf.plot_style import (
    BAR_COLOR,
    BAR_EDGE_COLOR,
    BAR_EDGE_WIDTH,
    FIGSIZE,
    LABEL_SIZE,
    LINEWIDTH_MAIN,
    LINEWIDTH_SECONDARY,
    MULTI_SERIES_COLORS,
    TITLE_SIZE,
    apply_rcparams,
    apply_style,
)
from ecg_scf.utils import load_config, project_root

# Paper Table 5 Arch A Mean All (for comparison columns only)
PAPER_REF = {
    "mean_idr": 0.956,
    "mean_far": 0.022,
    "mean_frr": 0.001,
}


def parse_args():
    p = argparse.ArgumentParser(description="Plot experiment figures and write REPORT.md")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--logs", type=str, default=None, help="Dir with train_summary.json")
    p.add_argument("--figdir", type=str, default=None)
    p.add_argument("--summary", type=str, default="train_summary.json")
    return p.parse_args()


def _load_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run train/eval first so metrics JSON exists."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_cmc(summary: dict, figdir: Path) -> Path:
    """Paper Fig. 8 style — CMC curves."""
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    folds = summary.get("folds") or []
    if folds and any(f.get("cmc") for f in folds):
        cmcs = [np.asarray(f["cmc"], dtype=float) for f in folds if f.get("cmc")]
        m = max(len(c) for c in cmcs)
        stacked = np.full((len(cmcs), m), np.nan)
        for i, c in enumerate(cmcs):
            stacked[i, : len(c)] = c
        mean_cmc = np.nanmean(stacked, axis=0)
        ranks = np.arange(1, len(mean_cmc) + 1)
        db_label = str(summary.get("database", "ours")).strip("_") or "ours"
        ax.plot(
            ranks,
            mean_cmc,
            color=MULTI_SERIES_COLORS[0],
            linewidth=LINEWIDTH_MAIN,
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label=f"{db_label} (ours)",
        )
        ax.legend(fontsize=9, loc="lower right")
    apply_style(ax)
    ax.set_xlabel("Rank", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Identification rate", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_title("CMC curve (ours)", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    out = figdir / "cmc.svg"
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def plot_metric_bars(summary: dict, figdir: Path) -> Path:
    """Paper Fig. 7 style — IDR / FAR / FRR bars."""
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    labels = ["IDR", "FAR", "FRR"]
    ours = [
        float(summary.get("mean_idr", 0.0)),
        float(summary.get("mean_far", 0.0)),
        float(summary.get("mean_frr", 0.0)),
    ]
    paper = [PAPER_REF["mean_idr"], PAPER_REF["mean_far"], PAPER_REF["mean_frr"]]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(
        x - w / 2,
        ours,
        w,
        label="Ours",
        color=BAR_COLOR,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
    )
    ax.bar(
        x + w / 2,
        paper,
        w,
        label="Paper Mean All (Arch A)",
        color=MULTI_SERIES_COLORS[1],
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
    )
    apply_style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_title("IDR / FAR / FRR (ours vs paper Table 5)", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=9, loc="upper right")
    out = figdir / "metrics_bars.svg"
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def plot_idr_boxplot(summary: dict, figdir: Path) -> Path | None:
    """Paper Fig. 6 style — IDR across validations/folds."""
    folds = summary.get("folds") or []
    idrs = [float(f["idr"]) for f in folds if "idr" in f]
    if len(idrs) < 2:
        return None
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bp = ax.boxplot(
        [idrs],
        tick_labels=[str(summary.get("database", "run"))],
        patch_artist=True,
        widths=0.5,
    )
    for box in bp["boxes"]:
        box.set_facecolor(BAR_COLOR)
        box.set_edgecolor(BAR_EDGE_COLOR)
        box.set_linewidth(BAR_EDGE_WIDTH)
    for med in bp["medians"]:
        med.set_color(MULTI_SERIES_COLORS[1])
        med.set_linewidth(LINEWIDTH_SECONDARY)
    apply_style(ax)
    ax.set_ylabel("Identification rate", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_title("IDR across validations (ours)", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    out = figdir / "idr_boxplot.svg"
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def write_report(
    summary: dict,
    cfg: dict,
    fig_paths: list[Path],
    report_path: Path,
    summary_path: Path,
) -> None:
    title = cfg.get("paper", {}).get("title") or "ECG SCF Identification"
    doi = cfg.get("paper", {}).get("doi", "")
    db = summary.get("database", cfg.get("data", {}).get("database"))
    arch = summary.get("arch", cfg.get("model", {}).get("name"))
    smoke = str(db).startswith("__") or db == "synthetic"

    fig_md = []
    for p in fig_paths:
        if p is None:
            continue
        rel = p.relative_to(project_root()).as_posix()
        name = p.stem
        captions = {
            "cmc": "CMC curve (paper Fig. 8 style). Ours from logged folds.",
            "metrics_bars": "IDR / FAR / FRR bars (paper Fig. 7 style). Paper bars = Table 5 Mean All Arch A.",
            "idr_boxplot": "IDR boxplot across validations (paper Fig. 6 style).",
        }
        cap = captions.get(name, name)
        fig_md.append(f"### Figure: `{name}`\n\n![{cap}]({rel})\n\n*{cap}*\n")

    body = f"""# Experiment report: ECG SCF + CNN

- **Paper:** `{cfg.get("paper", {}).get("slug", "")}`
- **Project:** ECG spectral-correlation identification
- **DOI:** {doi}
- **Generated from:** `{summary_path.relative_to(project_root()).as_posix()}`
- **Fidelity:** see [FIDELITY_AUDIT.md](./FIDELITY_AUDIT.md) (method match; not accuracy claims)
- **Plot style:** [dev-plot](../../../.cursor/skills/dev-plot/SKILL.md)

## Summary

| Metric | Ours | Paper Table 5 (Arch A Mean All) | Notes |
|--------|------|----------------------------------|-------|
| IDR | {summary.get("mean_idr", float("nan")):.4f} | {PAPER_REF["mean_idr"]:.3f} | {"**Smoke / incomplete protocol**" if smoke else "Fuller run"} |
| FAR | {summary.get("mean_far", float("nan")):.4f} | {PAPER_REF["mean_far"]:.3f} | FAR/FRR mode may differ (D2) |
| FRR | {summary.get("mean_frr", float("nan")):.4f} | {PAPER_REF["mean_frr"]:.3f} | |

Database=`{db}`, arch=`{arch}`, classes=`{summary.get("n_classes", "?")}`, folds/validations=`{len(summary.get("folds") or [])}`.

{"This report is from a **smoke** run (synthetic or capped protocol). Do not treat metrics as a paper reproduction." if smoke else "Numbers are from our local protocol; compare carefully to paper subsets."}

## Experimental setup (ours)

- Lead preference / resample: Lead II-ish -> 360 Hz, 2 s blind segments, max 30 min
- SCF: `{cfg.get("scf", {}).get("method", "caf_fft")}` (Eq. 10–11)
- Train: SGD lr={cfg.get("train", {}).get("lr")}, epochs={cfg.get("train", {}).get("epochs")}, batch={cfg.get("train", {}).get("batch_size")}
- Validations: `n_validations`={cfg.get("train", {}).get("n_validations")}, `n_folds`={cfg.get("train", {}).get("n_folds")}

## Results tables

### Per-fold metrics

| Fold | IDR | FAR | FRR |
|------|-----|-----|-----|
"""
    for i, f in enumerate(summary.get("folds") or []):
        body += f"| {i} | {f.get('idr', float('nan')):.4f} | {f.get('far', float('nan')):.4f} | {f.get('frr', float('nan')):.4f} |\n"

    body += "\n## Figures\n\n"
    body += "\n".join(fig_md) if fig_md else "_No figures generated._\n"
    body += """
## Comparison notes

- Paper Fig. 6–8 are **Results** plots; we regenerate the same *types* from our JSON logs with **dev-plot** styling.
- Fantasia runs may use `--max-segments` (caf_fft SCF build is costly at full 30 min); keep `n_validations=10` when matching Fig. 6.
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
"""
    report_path.write_text(body, encoding="utf-8")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    root = project_root()
    log_dir = Path(args.logs) if args.logs else root / cfg.get("paths", {}).get("log_dir", "outputs/logs")
    if not log_dir.is_absolute():
        log_dir = root / log_dir
    figdir = Path(args.figdir) if args.figdir else root / cfg.get("paths", {}).get("figure_dir", "outputs/figures")
    if not figdir.is_absolute():
        figdir = root / figdir
    figdir.mkdir(parents=True, exist_ok=True)

    summary_path = log_dir / args.summary
    summary = _load_summary(summary_path)

    paths = [
        plot_cmc(summary, figdir),
        plot_metric_bars(summary, figdir),
        plot_idr_boxplot(summary, figdir),
    ]
    paths = [p for p in paths if p is not None]
    report_path = root / "REPORT.md"
    write_report(summary, cfg, paths, report_path, summary_path)
    print(f"Wrote {len(paths)} figures under {figdir}")
    for p in paths:
        print(f"  {p}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
