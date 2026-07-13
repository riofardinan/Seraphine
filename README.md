# INESCO SER — 4 Baselines (apple-to-apple)

Speaker-independent Speech Emotion Recognition on **INESCO** (Indonesian
Expressive Speech Corpus) — 3 emotions: happiness, anger, sadness.

Four representations, **one identical downstream evaluator**, so differences in
score come only from the representation:

| # | Baseline | Representation | Dim | Rate | Compute |
|---|----------|----------------|-----|------|---------|
| 1 | eGeMAPS + SVM | openSMILE eGeMAPSv02 functionals | 88 | native 44.1 kHz | CPU |
| 2 | wav2vec2-base + SVM | `facebook/wav2vec2-base`, mean-pooled | 768 | 16 kHz | GPU |
| 3 | HuBERT-base + SVM | `facebook/hubert-base-ls960`, mean-pooled | 768 | 16 kHz | GPU |
| 4 | emotion2vec + SVM | `iic/emotion2vec_base`, utterance emb | 768 | 16 kHz | GPU |

**Shared, identical across all four** (`common.py`):
- Silence-trim edges (`top_db=40`), mono downmix; **no** manual amplitude norm.
- **Leave-One-Speaker-Out** (4 speakers → 4 folds); `StandardScaler` fit on the
  training fold only (no leakage).
- **LinearSVC, C=1.0** downstream classifier (grid-search over C is a separate
  ablation, kept out to preserve apple-to-apple comparability).
- Metrics: accuracy + macro-F1 (headline), UAR, per-class P/R/F1, confusion matrix.

> Preprocessing can't be *fully* identical: eGeMAPS keeps native 44.1 kHz and its
> loudness feature, while the SSL models require 16 kHz and normalise the waveform
> (`do_normalize=True`). That is inherent to the methods — report it as a caveat.

## Files

| File | Role |
|------|------|
| `config.py` | paths, labels, shared hyperparameters, model checkpoints |
| `build_manifest.py` | filenames → `outputs/manifest.csv` (speaker/emotion/sentence) |
| `common.py` | shared audio preprocessing + LOSO/SVM evaluator |
| `extract_egemaps.py` | Baseline 1 features → `outputs/features_egemaps.csv` |
| `extract_ssl.py --model {wav2vec2,hubert}` | Baseline 2/3 features |
| `extract_emotion2vec.py` | Baseline 4 features |
| `run_baseline.py --baseline {egemaps,wav2vec2,hubert,emotion2vec}` | shared eval → `outputs/results/` |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # baseline 1 (CPU)
pip install -r requirements-deep.txt       # baselines 2-4 (GPU box); match torch to your CUDA
```

## Run

```bash
python build_manifest.py

# Baseline 1 (CPU-friendly)
python extract_egemaps.py
python run_baseline.py --baseline egemaps

# Baselines 2-4 (GPU)
python extract_ssl.py --model wav2vec2   && python run_baseline.py --baseline wav2vec2
python extract_ssl.py --model hubert     && python run_baseline.py --baseline hubert
python extract_emotion2vec.py            && python run_baseline.py --baseline emotion2vec
```

## Outputs (`outputs/results/`, one set per baseline)

- `baselineN_*_per_fold_results.csv` — accuracy / macro-F1 / UAR / per-class P·R·F1 per speaker fold
- `baselineN_*_summary.csv` — mean ± std across folds (+ pooled)
- `baselineN_*_confusion_matrices.csv` — per-fold + pooled confusion counts
- `baselineN_*_confusion_matrix.png` — pooled, row-normalised
- `baseline1_egemaps_feature_importance.csv` — |SVM weight| per eGeMAPS feature

## Caveats (for the write-up)

- INESCO is **non-parallel** (emotion correlates with sentence content) → plan a
  separate **text-only baseline** to quantify the content confound.
- Only **4 speakers** → LOSO mandatory; report per-fold, not just the mean.
- Emotions are **acted** (theatre artists) → scope claims to expressive/acted emotion.
- Dataset is **2,399** usable clips (fcim missing 1 anger; `mbaz_h138.wav` had a
  stripped WAV header, auto-repaired in `common.load_audio`).
