"""Run the shared LOSO + linear-SVM evaluation on any baseline's feature table.

The evaluator (common.run_loso) is identical for every baseline, so results are
directly comparable apple-to-apple. Only the input feature CSV changes.

Usage:
    python run_baseline.py --baseline egemaps       # baseline 1
    python run_baseline.py --baseline wav2vec2      # baseline 2
    python run_baseline.py --baseline hubert        # baseline 3
    python run_baseline.py --baseline emotion2vec   # baseline 4
    python run_baseline.py --features outputs/features_x.csv --name my_run
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

import common
import config

# baseline key -> (features name, result prefix, dump feature importances?)
BASELINES = {
    "egemaps": ("egemaps", "baseline1_egemaps", True),
    "wav2vec2": ("wav2vec2", "baseline2_wav2vec2", False),
    "hubert": ("hubert", "baseline3_hubert", False),
    "emotion2vec": ("emotion2vec", "baseline4_emotion2vec", False),
    "xlsr": ("xlsr", "baseline6_xlsr", False),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=BASELINES)
    ap.add_argument("--features", help="explicit path to a feature CSV")
    ap.add_argument("--name", help="result prefix (with --features)")
    args = ap.parse_args()

    if args.baseline:
        feat_name, name, importance = BASELINES[args.baseline]
        path = config.features_csv(feat_name)
    elif args.features and args.name:
        path, name, importance = args.features, args.name, False
    else:
        ap.error("provide --baseline, or both --features and --name")

    if not Path(path).exists():
        sys.exit(f"[error] features not found: {path}\n        run the matching extractor first")

    df = pd.read_csv(path)
    common.run_loso(df, name, feature_importance=importance)


if __name__ == "__main__":
    main()
