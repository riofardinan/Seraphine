"""Baseline 5 - MFCC + CNN (trained end-to-end).

A SEPARATE deep-learning baseline, NOT part of the frozen apple-to-apple set:
here the CNN *is* the classifier (trained), unlike baselines 1-4 which share a
frozen-feature + linear-SVM downstream. It answers a different question - can a
CNN trained from scratch on basic spectral features beat frozen representations
on this small corpus? Kept comparable at the reporting level: same LOSO 4-fold
split and same output files via common.report_loso.

Input: MFCC(40) + delta + delta-delta -> 3-channel [3, 40, T], per-utterance
CMVN, crop/pad to MFCC_MAX_SECONDS. Small regularised CNN (dropout + SpecAugment
+ early stopping on an inner validation split; the test speaker stays held out).

Needs torch (see requirements-deep.txt). GPU recommended.
Output: outputs/results/baseline5_mfcc_cnn_*
"""
import sys
import time

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import common
import config

NAME = "baseline5_mfcc_cnn"
TARGET_FRAMES = int(round(config.MFCC_MAX_SECONDS * config.MFCC_SR / config.MFCC_HOP))
EMO2IDX = {e: i for i, e in enumerate(config.EMOTIONS)}


# ---------------------------------------------------------------------------
# MFCC precompute (once, reused across folds)
# ---------------------------------------------------------------------------
def mfcc_3ch(path) -> np.ndarray:
    """Load -> MFCC + delta + delta2 -> per-utterance CMVN -> [3, N_MFCC, T]."""
    y, sr = common.preprocess(path, config.MFCC_SR)
    if y.size < config.MFCC_N_FFT:
        y = np.pad(y, (0, config.MFCC_N_FFT - y.size))
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.N_MFCC,
                             n_fft=config.MFCC_N_FFT, hop_length=config.MFCC_HOP)
    feats = np.stack([m, librosa.feature.delta(m), librosa.feature.delta(m, order=2)])
    mu = feats.mean(axis=2, keepdims=True)
    sd = feats.std(axis=2, keepdims=True) + 1e-8
    return ((feats - mu) / sd).astype(np.float32)       # [3, N_MFCC, T]


def fit_length(x: np.ndarray, train: bool, rng: np.random.Generator) -> np.ndarray:
    """Crop (random in train, centre in eval) or zero-pad time axis to TARGET_FRAMES."""
    t = x.shape[2]
    if t == TARGET_FRAMES:
        return x
    if t > TARGET_FRAMES:
        start = int(rng.integers(0, t - TARGET_FRAMES + 1)) if train else (t - TARGET_FRAMES) // 2
        return x[:, :, start:start + TARGET_FRAMES]
    return np.pad(x, ((0, 0), (0, 0), (0, TARGET_FRAMES - t)))


def spec_augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light SpecAugment: one frequency mask + one time mask."""
    x = x.copy()
    f = int(rng.integers(0, config.N_MFCC // 5 + 1))
    if f:
        f0 = int(rng.integers(0, config.N_MFCC - f + 1))
        x[:, f0:f0 + f, :] = 0.0
    t = int(rng.integers(0, TARGET_FRAMES // 5 + 1))
    if t:
        t0 = int(rng.integers(0, TARGET_FRAMES - t + 1))
        x[:, :, t0:t0 + t] = 0.0
    return x


class MFCCDataset(Dataset):
    def __init__(self, feats, labels, train, seed):
        self.feats, self.labels, self.train = feats, labels, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = fit_length(self.feats[i], self.train, self.rng)
        if self.train:
            x = spec_augment(x, self.rng)
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.labels[i])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MFCCNet(nn.Module):
    def __init__(self, n_classes=len(config.EMOTIONS), in_ch=3, p=config.CNN_DROPOUT):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(p),
            nn.Linear(64 * 4 * 4, 64), nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_fold(feats, labels, tr_idx, device, seed):
    """Train with early stopping on an inner (stratified) validation split."""
    tr, va = train_test_split(tr_idx, test_size=config.CNN_VAL_FRAC,
                              stratify=labels[tr_idx], random_state=seed)
    dl_tr = DataLoader(MFCCDataset([feats[i] for i in tr], labels[tr], True, seed),
                       batch_size=config.CNN_BATCH, shuffle=True, drop_last=False)
    dl_va = DataLoader(MFCCDataset([feats[i] for i in va], labels[va], False, seed),
                       batch_size=config.CNN_BATCH, shuffle=False)

    model = MFCCNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.CNN_LR,
                           weight_decay=config.CNN_WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    best_acc, best_state, patience = -1.0, None, 0
    for epoch in range(config.CNN_EPOCHS):
        model.train()
        for xb, yb in dl_tr:
            opt.zero_grad()
            loss_fn(model(xb.to(device)), yb.to(device)).backward()
            opt.step()
        # validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                pred = model(xb.to(device)).argmax(1).cpu()
                correct += (pred == yb).sum().item()
                total += yb.numel()
        val_acc = correct / max(total, 1)
        if val_acc > best_acc:
            best_acc, best_state, patience = val_acc, {k: v.cpu().clone()
                                                       for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= config.CNN_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_acc


@torch.no_grad()
def predict(model, feats, idx, device, seed):
    model.eval()
    dl = DataLoader(MFCCDataset([feats[i] for i in idx], np.zeros(len(idx)), False, seed),
                    batch_size=config.CNN_BATCH, shuffle=False)
    out = [model(xb.to(device)).argmax(1).cpu().numpy() for xb, _ in dl]
    return np.concatenate(out)


def main() -> None:
    if not config.MANIFEST_CSV.is_file():
        sys.exit("[error] manifest not found - run build_manifest.py first")

    torch.manual_seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest = pd.read_csv(config.MANIFEST_CSV)
    print(f"[{NAME}] MFCC precompute: {len(manifest)} clips "
          f"(3x{config.N_MFCC}xT, target T={TARGET_FRAMES}) on {device}")

    feats, keep = [], []
    t0 = time.time()
    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        try:
            feats.append(mfcc_3ch(config.DATASET_ROOT / row.relpath))
            keep.append(True)
        except Exception as e:  # noqa: BLE001
            print(f"     [fail] {row.filename}: {e!r}")
            keep.append(False)
        if i % 400 == 0 or i == len(manifest):
            print(f"  MFCC {i:>4}/{len(manifest)}  ({i / (time.time() - t0):4.1f} clips/s)")

    manifest = manifest[keep].reset_index(drop=True)
    labels = manifest["emotion"].map(EMO2IDX).to_numpy()
    groups = manifest["speaker"].to_numpy()
    idx_all = np.arange(len(manifest))

    preds = np.empty(len(manifest), dtype=object)
    for sp in pd.unique(groups):
        te_idx = idx_all[groups == sp]
        tr_idx = idx_all[groups != sp]
        model, val_acc = train_fold(feats, labels, tr_idx, device, config.RANDOM_STATE)
        pred_idx = predict(model, feats, te_idx, device, config.RANDOM_STATE)
        preds[te_idx] = [config.EMOTIONS[p] for p in pred_idx]
        print(f"[{NAME}] fold test={sp:5} trained (inner val acc={val_acc:.3f})")

    common.report_loso(NAME, manifest["emotion"].to_numpy(), preds, groups)


if __name__ == "__main__":
    main()
