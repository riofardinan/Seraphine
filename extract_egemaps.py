"""Baseline 1 features - eGeMAPSv02 functionals (88-d) from openSMILE.

Preprocessing: native 44.1 kHz (openSMILE handles it) + mono + silence-trim.
NO amplitude normalisation (loudness is an informative eGeMAPS feature).

Output: outputs/features_egemaps.csv  (META_COLS + 88 feature columns).
Run build_manifest.py first. CPU-only.
"""
import sys
import time
import warnings

import numpy as np
import opensmile
import pandas as pd

import common
import config

warnings.filterwarnings("ignore", category=UserWarning)


def main() -> None:
    if not config.MANIFEST_CSV.is_file():
        sys.exit("[error] manifest not found - run build_manifest.py first")

    manifest = pd.read_csv(config.MANIFEST_CSV)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    names = smile.feature_names
    print(f"[egemaps] eGeMAPSv02 -> {len(names)} features | {len(manifest)} clips "
          f"| sr={'native' if config.EGEMAPS_SR is None else config.EGEMAPS_SR}")

    feats, failures = [], []
    t0 = time.time()
    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        try:
            y, sr = common.preprocess(config.DATASET_ROOT / row.relpath, config.EGEMAPS_SR)
            if y.size == 0:
                raise ValueError("empty signal after trim")
            feats.append(smile.process_signal(y, sr).iloc[0].to_numpy(dtype=float))
        except Exception as e:  # noqa: BLE001
            failures.append((row.filename, repr(e)))
            feats.append(np.full(len(names), np.nan))
        if i % 250 == 0 or i == len(manifest):
            print(f"  {i:>4}/{len(manifest)}  ({i / (time.time() - t0):4.1f} clips/s)")

    out = pd.concat([manifest.reset_index(drop=True),
                     pd.DataFrame(feats, columns=names)], axis=1)
    bad = out[names].isna().all(axis=1)
    if bad.any():
        print(f"[warn] dropping {int(bad.sum())} failed clip(s)")
        out = out[~bad].reset_index(drop=True)

    out.to_csv(config.features_csv("egemaps"), index=False)
    print(f"[ok] wrote {config.features_csv('egemaps')}  shape={out.shape}")
    for fn, err in failures[:10]:
        print(f"     [fail] {fn}: {err}")


if __name__ == "__main__":
    main()
