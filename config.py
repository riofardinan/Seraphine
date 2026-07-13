"""Shared configuration for the INESCO SER baselines (apple-to-apple).

Four baselines share ONE downstream evaluator (common.run_loso): LOSO CV +
train-only z-normalisation + linear SVM (C=1.0). The ONLY thing that differs
between baselines is the feature/embedding table:

  baseline1  eGeMAPS (openSMILE, 88-d, native 44.1 kHz)
  baseline2  wav2vec2-base embedding (768-d, 16 kHz)
  baseline3  HuBERT-base   embedding (768-d, 16 kHz)
  baseline4  emotion2vec   embedding (768-d, 16 kHz)
"""
from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = (
    PROJECT_ROOT
    / "INESCO Dataset Indonesian Expressive Speech Corpus"
    / "INESCO Dataset"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"
MANIFEST_CSV = OUTPUT_DIR / "manifest.csv"


def features_csv(name: str) -> Path:
    """Path of the cached feature table for a given extractor name."""
    return OUTPUT_DIR / f"features_{name}.csv"


# --- Labels / speakers (from the INESCO file-naming guideline) -------------
EMOTION_MAP = {"h": "happiness", "a": "anger", "s": "sadness"}
SPEAKER_GENDER = {"mdpa": "male", "mbaz": "male", "fyat": "female", "fcim": "female"}
EMOTIONS = ["happiness", "anger", "sadness"]           # fixed order for reports
META_COLS = ["filename", "relpath", "speaker", "gender", "emotion", "sentence_id"]

# --- Preprocessing (shared where possible) ---------------------------------
# Common to all: trim leading/trailing silence (edges only); NO amplitude
# normalisation (loudness is an informative eGeMAPS feature; SSL extractors do
# their own waveform normalisation internally).
TRIM_TOP_DB = 40            # light silence trim, applied to every baseline
EGEMAPS_SR = None           # eGeMAPS: keep native 44.1 kHz (openSMILE handles it)
SSL_SR = 16000              # wav2vec2 / HuBERT / emotion2vec pretrained at 16 kHz

# --- Downstream classifier (IDENTICAL across all four baselines) -----------
SVM_C = 1.0                 # fixed default; grid-search over C is a separate ablation

# --- Model checkpoints ------------------------------------------------------
WAV2VEC2_MODEL = "facebook/wav2vec2-base"        # self-supervised, non-ASR-finetuned
HUBERT_MODEL = "facebook/hubert-base-ls960"      # self-supervised, non-ASR-finetuned
EMOTION2VEC_MODEL = "iic/emotion2vec_base"       # base, apple-to-apple with base SSL
XLSR_ID_MODEL = "indonesian-nlp/wav2vec2-large-xlsr-indonesian"  # xlsr-53 large, Indonesian ASR-finetuned, 1024-d

# --- Baseline 5: MFCC + CNN -------------------------------------------------
# SEPARATE deep-learning baseline (trained end-to-end), NOT part of the frozen
# apple-to-apple set. Same LOSO split + output format, different classifier.
MFCC_SR = 16000            # standard rate for MFCC-SER; compact CNN input
N_MFCC = 40                # + delta + delta-delta -> 3 input channels
MFCC_N_FFT = 1024
MFCC_HOP = 512
MFCC_MAX_SECONDS = 4.0     # crop/pad MFCC to this many seconds for batching
CNN_EPOCHS = 60
CNN_BATCH = 32
CNN_LR = 1e-3
CNN_WEIGHT_DECAY = 1e-4
CNN_DROPOUT = 0.5
CNN_PATIENCE = 12          # early-stopping patience (on inner validation)
CNN_VAL_FRAC = 0.15        # inner val split for early stopping (test speaker stays held out)

RANDOM_STATE = 42
