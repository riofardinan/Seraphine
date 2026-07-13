"""Baseline 2 & 3 features - frozen wav2vec2 / HuBERT embeddings (768-d).

Preprocessing: resample to 16 kHz + mono + silence-trim, then the HuggingFace
feature extractor applies its own zero-mean/unit-variance waveform normalisation
(do_normalize=True). Per clip: forward pass -> last_hidden_state [1, T, 768]
-> mean-pool over time -> 768-d vector. Model is frozen (eval / no_grad).

Usage:
    python extract_ssl.py --model wav2vec2     # facebook/wav2vec2-base       (768-d)
    python extract_ssl.py --model hubert       # facebook/hubert-base-ls960   (768-d)
    python extract_ssl.py --model xlsr         # facebook/wav2vec2-xls-r-300m (1024-d, multilingual)

Needs torch + transformers (see requirements-deep.txt). GPU strongly recommended.
Output: outputs/features_{wav2vec2,hubert}.csv
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, HubertModel, Wav2Vec2Model

import common
import config

MODELS = {
    "wav2vec2": (Wav2Vec2Model, config.WAV2VEC2_MODEL),
    "hubert": (HubertModel, config.HUBERT_MODEL),
    "xlsr": (Wav2Vec2Model, config.XLSR_MODEL),      # XLS-R uses the Wav2Vec2 architecture
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    args = ap.parse_args()

    if not config.MANIFEST_CSV.is_file():
        sys.exit("[error] manifest not found - run build_manifest.py first")

    model_cls, checkpoint = MODELS[args.model]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = AutoFeatureExtractor.from_pretrained(checkpoint)
    model = model_cls.from_pretrained(checkpoint).to(device).eval()
    print(f"[{args.model}] {checkpoint} on {device} | do_normalize="
          f"{getattr(extractor, 'do_normalize', '?')}")

    manifest = pd.read_csv(config.MANIFEST_CSV)
    rows, failures = [], []
    t0 = time.time()
    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        try:
            y, _ = common.preprocess(config.DATASET_ROOT / row.relpath, config.SSL_SR)
            if y.size == 0:
                raise ValueError("empty signal after trim")
            inputs = extractor(y, sampling_rate=config.SSL_SR, return_tensors="pt")
            with torch.no_grad():
                out = model(inputs.input_values.to(device))
            emb = out.last_hidden_state.squeeze(0).mean(dim=0).cpu().numpy()
            rows.append(emb)
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

    dest = config.features_csv(args.model)
    out.to_csv(dest, index=False)
    print(f"[ok] wrote {dest}  shape={out.shape}  (embedding dim={dim})")
    for fn, err in failures[:10]:
        print(f"     [fail] {fn}: {err}")


if __name__ == "__main__":
    main()
