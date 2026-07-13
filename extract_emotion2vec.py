"""Baseline 4 features - frozen emotion2vec utterance embeddings (768-d).

emotion2vec (Ma et al., 2024) is a self-supervised model specialised for speech
emotion. We use it purely as a frozen feature extractor via FunASR: per clip,
utterance-granularity embedding -> 768-d vector (emotion2vec_base).

Preprocessing: resample to 16 kHz + mono + silence-trim (same as baselines 2/3),
so the ONLY difference from the SSL baselines is the representation.

Needs funasr + modelscope (see requirements-deep.txt). GPU recommended; the model
is downloaded from ModelScope on first run.
Output: outputs/features_emotion2vec.csv
"""
import sys
import time

import numpy as np
import pandas as pd
from funasr import AutoModel

import common
import config


def _embedding(rec) -> np.ndarray:
    """Pull the utterance embedding out of a FunASR emotion2vec result."""
    item = rec[0] if isinstance(rec, (list, tuple)) else rec
    feats = item["feats"] if isinstance(item, dict) else item
    return np.asarray(feats, dtype=float).reshape(-1)


def main() -> None:
    if not config.MANIFEST_CSV.is_file():
        sys.exit("[error] manifest not found - run build_manifest.py first")

    model = AutoModel(model=config.EMOTION2VEC_MODEL, disable_update=True)
    print(f"[emotion2vec] {config.EMOTION2VEC_MODEL} loaded")

    manifest = pd.read_csv(config.MANIFEST_CSV)
    rows, failures = [], []
    t0 = time.time()
    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        try:
            y, _ = common.preprocess(config.DATASET_ROOT / row.relpath, config.SSL_SR)
            if y.size == 0:
                raise ValueError("empty signal after trim")
            rec = model.generate(y, granularity="utterance", extract_embedding=True)
            rows.append(_embedding(rec))
        except Exception as e:  # noqa: BLE001
            failures.append((row.filename, repr(e)))
            rows.append(None)
        if i % 200 == 0 or i == len(manifest):
            print(f"  {i:>4}/{len(manifest)}  ({i / (time.time() - t0):4.1f} clips/s)")

    dim = len(next(r for r in rows if r is not None))
    cols = [f"feat_{k}" for k in range(1, dim + 1)]
    mat = np.array([r if r is not None else np.full(dim, np.nan) for r in rows])
    out = pd.concat([manifest.reset_index(drop=True),
                     pd.DataFrame(mat, columns=cols)], axis=1)
    bad = out[cols].isna().all(axis=1)
    if bad.any():
        print(f"[warn] dropping {int(bad.sum())} failed clip(s)")
        out = out[~bad].reset_index(drop=True)

    dest = config.features_csv("emotion2vec")
    out.to_csv(dest, index=False)
    print(f"[ok] wrote {dest}  shape={out.shape}  (embedding dim={dim})")
    for fn, err in failures[:10]:
        print(f"     [fail] {fn}: {err}")


if __name__ == "__main__":
    main()
