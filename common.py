"""Shared building blocks for all INESCO SER baselines.

Two things are shared so the four baselines stay apple-to-apple:
  * load_audio / preprocess  - identical silence-trim + mono handling
  * run_loso                 - identical LOSO CV + z-norm + linear SVM + metrics

Only the feature/embedding table fed into run_loso differs between baselines.
"""
from __future__ import annotations

import io
import struct
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import config


# ---------------------------------------------------------------------------
# Audio loading / preprocessing (shared by every extractor)
# ---------------------------------------------------------------------------
def load_audio(path, target_sr: int | None):
    """Load -> mono; resample to target_sr (None keeps native rate).

    Repairs a stripped RIFF header in memory (INESCO mbaz_h138.wav begins at
    'WAVE', missing the leading 8-byte 'RIFF'+size). The raw file is untouched.
    """
    try:
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        return y.astype("float32"), sr
    except Exception:
        data = Path(path).read_bytes()
        if data[:4] != b"WAVE":
            raise                                     # not the header-strip case
        data = b"RIFF" + struct.pack("<I", len(data)) + data
        y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if y.ndim > 1:                                # stereo -> mono
            y = y.mean(axis=1)
        if target_sr is not None and sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return y.astype("float32"), sr


def preprocess(path, target_sr: int | None, top_db: int = config.TRIM_TOP_DB):
    """Load -> mono/(resample) -> trim leading/trailing silence."""
    y, sr = load_audio(path, target_sr)
    y, _ = librosa.effects.trim(y, top_db=top_db)
    return y, sr


# ---------------------------------------------------------------------------
# Shared evaluator: LOSO + z-norm + linear SVM (C=1.0)
# ---------------------------------------------------------------------------
def _make_pipeline() -> Pipeline:
    """The identical downstream classifier used by ALL baselines."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # safety net (usually a no-op)
        ("scale", StandardScaler()),                    # fit on train fold only
        ("svm", LinearSVC(C=config.SVM_C, dual="auto",
                          max_iter=20000, random_state=config.RANDOM_STATE)),
    ])


def run_loso(df: pd.DataFrame, name: str, feature_importance: bool = False) -> dict:
    """LOSO evaluation of the shared frozen-feature + linear-SVM pipeline.

    df must contain config.META_COLS + one column per feature. Used by baselines
    1-4 (eGeMAPS / wav2vec2 / HuBERT / emotion2vec).
    """
    feat_cols = [c for c in df.columns if c not in config.META_COLS]
    X = df[feat_cols].to_numpy(dtype=float)
    y = df["emotion"].to_numpy()
    groups = df["speaker"].to_numpy()
    print(f"[{name}] X={X.shape}  speakers={sorted(set(groups))}  "
          f"classifier=LinearSVC(C={config.SVM_C})")

    preds = np.empty(len(y), dtype=object)
    coefs = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        pipe = _make_pipeline()
        pipe.fit(X[tr], y[tr])                          # scaler fit on train fold only
        preds[te] = pipe.predict(X[te])
        if feature_importance:
            coefs.append(np.abs(pipe.named_steps["svm"].coef_))

    return report_loso(name, y, preds, groups,
                       coefs=coefs or None, feat_cols=feat_cols)


def report_loso(name, y, preds, groups, coefs=None, feat_cols=None) -> dict:
    """Compute per-fold + pooled metrics from LOSO predictions and write outputs.

    Shared by the SVM baselines and the MFCC+CNN baseline so every baseline emits
    the same files: <name>_per_fold_results.csv, <name>_summary.csv,
    <name>_confusion_matrices.csv, <name>_confusion_matrix.png
    (+ <name>_feature_importance.csv when coefs/feat_cols are given).

    Each sample is a test sample exactly once under LOSO, so folds are recovered
    by grouping predictions per test speaker.
    """
    y = np.asarray(y)
    preds = np.asarray(preds)
    groups = np.asarray(groups)
    emos = config.EMOTIONS

    fold_rows, cm_frames = [], []
    for sp in pd.unique(groups):
        m = groups == sp
        yt, yp = y[m], preds[m]
        row = {
            "test_speaker": sp,
            "n_test": int(m.sum()),
            "accuracy": accuracy_score(yt, yp),
            "macro_f1": f1_score(yt, yp, average="macro", labels=emos),
            "uar": recall_score(yt, yp, average="macro", labels=emos),
        }
        prec = precision_score(yt, yp, average=None, labels=emos, zero_division=0)
        rec = recall_score(yt, yp, average=None, labels=emos, zero_division=0)
        f1c = f1_score(yt, yp, average=None, labels=emos, zero_division=0)
        for i, e in enumerate(emos):
            row[f"precision_{e}"] = prec[i]
            row[f"recall_{e}"] = rec[i]
            row[f"f1_{e}"] = f1c[i]
        fold_rows.append(row)
        print(f"  fold test={sp:5}  acc={row['accuracy']:.3f}  "
              f"macroF1={row['macro_f1']:.3f}  UAR={row['uar']:.3f}")

        cm = confusion_matrix(yt, yp, labels=emos)
        cmf = pd.DataFrame(cm, index=emos, columns=emos).reset_index(names="true")
        cmf.insert(0, "fold", sp)
        cm_frames.append(cmf)

    fold_df = pd.DataFrame(fold_rows)
    metric_cols = [c for c in fold_df.columns if c not in ("test_speaker", "n_test")]

    pooled_acc = accuracy_score(y, preds)
    pooled_f1 = f1_score(y, preds, average="macro", labels=emos)
    pooled_uar = recall_score(y, preds, average="macro", labels=emos)

    print("-" * 58)
    print(f"[{name}] per-speaker mean +/- std over {len(fold_df)} folds:")
    for mname in ("accuracy", "macro_f1", "uar"):
        print(f"    {mname:9} {fold_df[mname].mean():.3f} +/- {fold_df[mname].std():.3f}")
    print(f"[{name}] pooled: acc={pooled_acc:.3f}  macroF1={pooled_f1:.3f}  UAR={pooled_uar:.3f}")
    print("\n" + classification_report(y, preds, labels=emos, digits=3))

    # --- Write outputs -----------------------------------------------------
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(config.RESULTS_DIR / f"{name}_per_fold_results.csv", index=False)

    summary = fold_df[metric_cols].agg(["mean", "std"]).T
    summary.index.name = "metric"
    summary.loc["pooled_accuracy"] = [pooled_acc, np.nan]
    summary.loc["pooled_macro_f1"] = [pooled_f1, np.nan]
    summary.loc["pooled_uar"] = [pooled_uar, np.nan]
    summary.to_csv(config.RESULTS_DIR / f"{name}_summary.csv")

    pooled_cm = confusion_matrix(y, preds, labels=emos)
    pooled_frame = pd.DataFrame(pooled_cm, index=emos, columns=emos).reset_index(names="true")
    pooled_frame.insert(0, "fold", "POOLED")
    pd.concat(cm_frames + [pooled_frame], ignore_index=True).to_csv(
        config.RESULTS_DIR / f"{name}_confusion_matrices.csv", index=False)

    _plot_confusion(pooled_cm, name, pooled_acc)

    if coefs and feat_cols:
        mean_abs = np.mean(coefs, axis=0)               # (n_classes, n_features)
        imp = pd.DataFrame({"feature": feat_cols, "importance": mean_abs.mean(axis=0)})
        for i, e in enumerate(emos):
            imp[f"w_{e}"] = mean_abs[i]
        imp.sort_values("importance", ascending=False).to_csv(
            config.RESULTS_DIR / f"{name}_feature_importance.csv", index=False)

    print(f"[{name}] results written to {config.RESULTS_DIR}")
    return {"accuracy": pooled_acc, "macro_f1": pooled_f1, "uar": pooled_uar}


def _plot_confusion(cm: np.ndarray, name: str, acc: float) -> None:
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(5.2, 4.4))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=config.EMOTIONS, yticklabels=config.EMOTIONS,
                cbar_kws={"label": "recall"})
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{name}\nPooled LOSO, accuracy={acc:.3f}")
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / f"{name}_confusion_matrix.png", dpi=150)
    plt.close()
