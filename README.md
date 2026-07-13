# INESCO SER — Classical Baseline (eGeMAPS + SVM)

Speaker-independent Speech Emotion Recognition on the **INESCO** corpus
(Indonesian Expressive Speech Corpus) — 3 emotions: happiness, anger, sadness.

This is the **classical baseline** of a larger comparison whose top method is
`emotion2vec`. Method and evaluation protocol follow **Eyben et al. (2016)**, the
eGeMAPS paper.

> **Compute:** CPU-only. openSMILE (C++) extracts features and SVM trains on CPU —
> no GPU is used at this stage. The RTX 5070 only matters for the later
> self-supervised models (emotion2vec / wav2vec2 / CNN).

## Pipeline

| Stage | Script | What it does | Output |
|-------|--------|--------------|--------|
| 0 | `build_manifest.py`  | Parse filenames → speaker / emotion / sentence labels | `outputs/manifest.csv` |
| 1–2 | `extract_features.py` | 16 kHz mono + silence-trim, then eGeMAPSv02 (88 feats) | `outputs/egemaps_features.csv` |
| 3–5 | `train_svm.py` | LOSO CV, z-norm, SVM (C tuned by nested CV), metrics | `outputs/results/` |

## Method summary

- **Features:** eGeMAPSv02 functionals → one **88-dim** vector per utterance
  (F0, jitter, shimmer, loudness, HNR, formants, spectral balance, MFCC 1–4, …).
  Frame-level LLDs are aggregated by functionals, so no temporal model is needed.
- **Preprocessing:** resample to **16 kHz mono** (consistent with later models),
  trim leading/trailing silence only (`top_db=30`); no amplitude normalisation
  (loudness is auditory-relative; z-norm handles scale).
- **Evaluation:** **Leave-One-Speaker-Out** (4 speakers → 4 folds), so no speaker
  leakage. Features **z-normalised** per fold (fit on train only). SVM complexity
  **C tuned by nested speaker-independent CV** (test fold never seen during
  tuning). Class imbalance → `class_weight='balanced'`.
- **Metrics:** primary **UAR** (unweighted average recall) — the paralinguistics
  standard used in the eGeMAPS paper — plus WA, macro-F1, confusion matrix, and
  linear-SVM feature importances tied back to GeMAPS theory.

## Setup & run

```bash
# from the ser_egemaps/ directory, on the target machine
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python build_manifest.py      # Stage 0
python extract_features.py    # Stage 1-2  (a few minutes, CPU)
python train_svm.py           # Stage 3-5
```

Edit knobs in `config.py` (paths, `TARGET_SR`, `TRIM_TOP_DB`) and in
`train_svm.py` (`KERNEL` = `linear`/`rbf`, `C_GRID`).

## Outputs (`outputs/results/`)

- `summary.json` — per-speaker (mean±std) and pooled UAR / WA / macro-F1, best C per fold
- `per_fold_metrics.csv` — one row per LOSO fold
- `predictions.csv` — per-clip true vs predicted label
- `confusion_matrix.png` — pooled, row-normalised (recall view)
- `feature_importance.csv` — eGeMAPS features ranked by |SVM weight| (linear kernel)

## Notes / caveats (for the write-up)

- INESCO is **non-parallel** (emotion correlates with sentence content) — plan a
  separate **text-only baseline** to quantify this content confound.
- Only **4 speakers** → LOSO is mandatory; generalisation claims stay scoped.
- Emotions are **acted** (theatre artists) — scope claims to expressive/acted emotion.
