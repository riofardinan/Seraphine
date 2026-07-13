"""Shared configuration for the INESCO eGeMAPS + SVM classical SER baseline.

All paths are anchored to this file's location so the pipeline is portable.
"""
from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = (
    PROJECT_ROOT.parent
    / "INESCO Dataset Indonesian Expressive Speech Corpus"
    / "INESCO Dataset"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MANIFEST_CSV = OUTPUT_DIR / "manifest.csv"
FEATURES_CSV = OUTPUT_DIR / "egemaps_features.csv"
RESULTS_DIR = OUTPUT_DIR / "results"

# --- Label / speaker maps (from the INESCO file-naming guideline) ----------
EMOTION_MAP = {"h": "happiness", "a": "anger", "s": "sadness"}
SPEAKER_GENDER = {"mdpa": "male", "mbaz": "male", "fyat": "female", "fcim": "female"}
EMOTIONS = ["happiness", "anger", "sadness"]  # fixed order for reports

# --- Audio preprocessing ---------------------------------------------------
TARGET_SR = 16000       # resample every clip to 16 kHz mono (consistent across all models)
TRIM_TOP_DB = 30        # trim leading/trailing silence this many dB below peak (edges only)

# --- Reproducibility -------------------------------------------------------
RANDOM_STATE = 42
