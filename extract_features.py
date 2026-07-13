"""Stage 1-2 - Preprocess audio and extract eGeMAPSv02 functionals (88 features).

Per clip:
  1. load -> resample to 16 kHz mono            (config.TARGET_SR)
  2. trim leading/trailing silence (edges only) (config.TRIM_TOP_DB)
  3. eGeMAPSv02 Functionals -> 1 x 88 vector

Output: outputs/egemaps_features.csv  (metadata columns + 88 feature columns).
Run build_manifest.py first.
"""
import sys
import time
import warnings

import librosa
import numpy as np
import opensmile
import pandas as pd

import config

warnings.filterwarnings("ignore", category=UserWarning)


def make_extractor() -> opensmile.Smile:
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def preprocess(path) -> tuple[np.ndarray, int]:
    """Load -> 16 kHz mono -> trim silence at the edges."""
    y, sr = librosa.load(path, sr=config.TARGET_SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=config.TRIM_TOP_DB)
    return y, sr


def main() -> None:
    if not config.MANIFEST_CSV.is_file():
        sys.exit("[error] manifest not found - run build_manifest.py first")

    manifest = pd.read_csv(config.MANIFEST_CSV)
    smile = make_extractor()
    feature_names = smile.feature_names
    print(f"[info] eGeMAPSv02 -> {len(feature_names)} features per clip")
    print(f"[info] extracting from {len(manifest)} clips ...")

    feats, failures = [], []
    t0 = time.time()
    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        wav_path = config.DATASET_ROOT / row.relpath
        try:
            y, sr = preprocess(wav_path)
            if y.size == 0:
                raise ValueError("empty signal after trim")
            vec = smile.process_signal(y, sr).iloc[0].to_numpy(dtype=float)
            feats.append(vec)
        except Exception as e:  # noqa: BLE001 - log and continue
            failures.append((row.filename, str(e)))
            feats.append(np.full(len(feature_names), np.nan))

        if i % 250 == 0 or i == len(manifest):
            rate = i / (time.time() - t0)
            print(f"  {i:>4}/{len(manifest)}  ({rate:4.1f} clips/s)")

    feat_df = pd.DataFrame(feats, columns=feature_names)
    out = pd.concat([manifest.reset_index(drop=True), feat_df], axis=1)

    # Drop clips that failed (all-NaN feature rows) so downstream stays clean.
    bad = out[feature_names].isna().all(axis=1)
    if bad.any():
        print(f"[warn] dropping {int(bad.sum())} clip(s) that failed extraction")
        out = out[~bad].reset_index(drop=True)

    out.to_csv(config.FEATURES_CSV, index=False)
    print(f"\n[ok] wrote {config.FEATURES_CSV}")
    print(f"     shape: {out.shape}  ({len(feature_names)} feature cols)")
    if failures:
        print(f"[warn] {len(failures)} failure(s):")
        for name, err in failures[:10]:
            print(f"       - {name}: {err}")


if __name__ == "__main__":
    main()
