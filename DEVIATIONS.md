# Deviations from the paper

Reproduction target: faithful **runnable** reimplementation, not bit-exact author code.
Every row should also appear in `FIDELITY_AUDIT.md` (Pass A/B) or be marked resolved there.

| ID | Paper detail | Our choice | Reason |
|----|--------------|------------|--------|
| D1 | SCF via CAF (Eq. 10) then Fourier in τ (Eq. 11); FAM/smoothing window not named | Default `scf.method=caf_fft`: discrete CAF via FFT per lag τ, then FFT over τ; magnitude + resize 128×128. Optional `bifrequency` shortcut | Matches Eq. 10→11 structure; no author FAM params |
| D2 | FAR / FRR (Sec. 4.5) without operating point | Default `eval.far_frr_mode=eer` on genuine=P(true) vs impostor=P(other) softmax scores; optional fixed `accept_threshold` | Closed-set softmax; threshold underspecified |
| D3 | SGD lr=0.002, 15 epochs; batch / momentum not stated | `batch_size=32`, `momentum=0.9`, `weight_decay=0` | Common SGD defaults in config |
| D4 | Conv padding not stated (Fig. 5) | Same padding so spatial size stays 128 until MaxPool | Stable shapes; paper silent on pad |
| D5 | Nine DBs + Combined 488 | Local trees: CEBSDB+AFDB+… → **~485** subjects (paper 488). AFDB on disk has **20** signal records (paper 23; 2 annotation-only + possible gaps) | CEBSDB music-phase `mNNN` merged by volunteer id; Combined subject count short by AFDB completeness |
| D6 | Sec. 5.2 five-fold vs Fig. 6 ten validations | Default `train.n_validations=10` (StratifiedShuffleSplit 80/20); `n_folds=5` still available if `n_validations: null` | Aligns with Fig. 6; both modes configurable |
| D7 | Clean ECG samples assumed | Linear-interpolate sparse WFDB NaN/±inf before z-score; skip non-finite segments/SCF images | Fantasia ECG leads have rare missing samples that otherwise NaN-poison training |

## Upstream-related

| ID | Upstream behavior | Paper | Our choice | Reason |
|----|-------------------|-------|------------|--------|
| — | n/a | — | — | No public code |

## Missing artifacts

- Author code: not released — see SOURCE_CODE.md
- Pretrained weights: not released
- Private data: n/a (public PhysioNet)
- CEBSDB: on disk under `projects/datasets/cebsdb/data/` (enabled; music-phase `mNNN` preferred; submodule `biometric-community/CEBSDB-Combined-ECG-Breathing-Seismocardiograms`)
- AFDB: on disk under `projects/datasets/afdb/` (enabled; ~20 signal subjects vs paper 23 → Combined **485** not 488)

## Segment caps (runtime)

- Combined / CEBSDB train used `--max-segments 30` (SCF `caf_fft` cost); Fantasia smoke used ms50. Full 30 min/subject without cap is supported via config `max_segments_per_subject: null`.

## Hyperparameters guessed

- SGD momentum = 0.9
- Batch size = 32
- Conv padding = same

## Resolved in Pass D (no longer soft guesses)

- SCF path uses CAF→FFT (Eq. 10–11), not only outer-product
- CNN: single Dropout after MaxPool (Fig. 5); FC then Softmax head
- MITDB maps records 201+202 → one subject (47 subjects)
- FAR/FRR use genuine/impostor score framing (EER-like default)
- Ten validations via `n_validations=10`

## Not implemented (out of scope unless requested)

- Exact timing microbenchmark (~54 ms)
- Exact Combined-488 headcount until AFDB matches paper’s 23 signal subjects
- Literature baseline re-runs (Zhang HeartID, etc.)
- CEBSDB basal/post phases (we prefer music-phase `mNNN` per Table 4 duration)
