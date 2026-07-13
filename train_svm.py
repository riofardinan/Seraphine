"""Stage 3-5 - Speaker-independent SER with eGeMAPS + SVM.

Protocol follows Eyben et al. (2016), the eGeMAPS paper (Section 4.3):
  * Leave-One-Speaker-Out cross-validation (4 speakers -> 4 outer folds)
  * per-fold z-normalisation of features (fit on train only)
  * SVM classifier, complexity C tuned by NESTED speaker-independent CV
  * class imbalance handled with class_weight='balanced'
  * primary metric: UAR (Unweighted Average Recall = macro recall)

Reports pooled + per-speaker metrics, a confusion matrix, and (linear kernel)
eGeMAPS feature importances. CPU-only; no GPU required.

Run extract_features.py first.
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

import config

# --- Experiment settings ---------------------------------------------------
KERNEL = "linear"                       # "linear" (interpretable) or "rbf"
C_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
GAMMA_GRID = ["scale", 1e-3, 1e-2, 1e-1]  # only used for rbf
SCORING = "recall_macro"                # == UAR, used to select C in inner CV
TOP_K_FEATURES = 20


def build_search(inner_cv) -> GridSearchCV:
    """Pipeline (impute -> z-norm -> SVM) with C tuned by inner speaker-independent CV."""
    if KERNEL == "linear":
        clf = LinearSVC(class_weight="balanced", dual="auto",
                        max_iter=20000, random_state=config.RANDOM_STATE)
        grid = {"clf__C": C_GRID}
    else:
        clf = SVC(kernel="rbf", class_weight="balanced",
                  random_state=config.RANDOM_STATE)
        grid = {"clf__C": C_GRID, "clf__gamma": GAMMA_GRID}

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])
    return GridSearchCV(pipe, grid, scoring=SCORING, cv=inner_cv, n_jobs=-1)


def main() -> None:
    if not config.FEATURES_CSV.is_file():
        sys.exit("[error] features not found - run extract_features.py first")

    df = pd.read_csv(config.FEATURES_CSV)
    meta_cols = ["filename", "relpath", "speaker", "gender", "emotion", "sentence_id"]
    feat_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feat_cols].to_numpy(dtype=float)
    y = df["emotion"].to_numpy()
    groups = df["speaker"].to_numpy()
    print(f"[info] X={X.shape}  classes={sorted(set(y))}  speakers={sorted(set(groups))}")
    print(f"[info] kernel={KERNEL}  outer=LOSO({len(set(groups))} folds)  metric=UAR\n")

    outer_cv = LeaveOneGroupOut()
    fold_rows, coef_list = [], []
    preds = np.empty(len(y), dtype=object)

    for train_idx, test_idx in outer_cv.split(X, y, groups):
        test_speaker = groups[test_idx][0]
        inner_cv = LeaveOneGroupOut()  # over the 3 training speakers
        search = build_search(inner_cv)
        search.fit(X[train_idx], y[train_idx], groups=groups[train_idx])

        y_pred = search.predict(X[test_idx])
        preds[test_idx] = y_pred
        y_true = y[test_idx]

        uar = recall_score(y_true, y_pred, average="macro", labels=config.EMOTIONS)
        wa = accuracy_score(y_true, y_pred)
        mf1 = f1_score(y_true, y_pred, average="macro", labels=config.EMOTIONS)
        best_c = search.best_params_["clf__C"]
        fold_rows.append({"test_speaker": test_speaker, "n_test": len(test_idx),
                          "best_C": best_c, "UAR": uar, "WA": wa, "macroF1": mf1})
        print(f"  fold test={test_speaker:5}  C={best_c:<7}  "
              f"UAR={uar:.3f}  WA={wa:.3f}  macroF1={mf1:.3f}")

        if KERNEL == "linear":
            coef_list.append(np.abs(search.best_estimator_.named_steps["clf"].coef_))

    # --- Aggregate ---------------------------------------------------------
    fold_df = pd.DataFrame(fold_rows)
    pooled_uar = recall_score(y, preds, average="macro", labels=config.EMOTIONS)
    pooled_wa = accuracy_score(y, preds)
    pooled_f1 = f1_score(y, preds, average="macro", labels=config.EMOTIONS)

    print("\n" + "=" * 60)
    print("PER-SPEAKER (mean +/- std over 4 LOSO folds):")
    for m in ["UAR", "WA", "macroF1"]:
        print(f"  {m:8} {fold_df[m].mean():.3f} +/- {fold_df[m].std():.3f}")
    print("\nPOOLED (all predictions concatenated, GeMAPS-paper style):")
    print(f"  UAR={pooled_uar:.3f}  WA={pooled_wa:.3f}  macroF1={pooled_f1:.3f}")
    print("=" * 60)
    print("\nPer-class report (pooled):")
    print(classification_report(y, preds, labels=config.EMOTIONS, digits=3))

    # --- Save results ------------------------------------------------------
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(config.RESULTS_DIR / "per_fold_metrics.csv", index=False)
    df[meta_cols].assign(y_true=y, y_pred=preds).to_csv(
        config.RESULTS_DIR / "predictions.csv", index=False)

    summary = {
        "kernel": KERNEL,
        "n_clips": int(len(y)),
        "per_speaker": {m: {"mean": float(fold_df[m].mean()),
                            "std": float(fold_df[m].std())}
                        for m in ["UAR", "WA", "macroF1"]},
        "pooled": {"UAR": float(pooled_uar), "WA": float(pooled_wa),
                   "macroF1": float(pooled_f1)},
        "best_C_per_fold": dict(zip(fold_df["test_speaker"], fold_df["best_C"])),
    }
    (config.RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # Confusion matrix (row-normalised = per-class recall)
    cm = confusion_matrix(y, preds, labels=config.EMOTIONS, normalize="true")
    plt.figure(figsize=(5.2, 4.4))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=config.EMOTIONS, yticklabels=config.EMOTIONS,
                cbar_kws={"label": "recall"})
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"INESCO SER - eGeMAPS+SVM ({KERNEL})\nPooled LOSO, UAR={pooled_uar:.3f}")
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # Feature importance (linear kernel only)
    if coef_list:
        mean_abs = np.mean(coef_list, axis=0)          # (n_classes, n_features)
        overall = mean_abs.mean(axis=0)                # mean over classes+folds
        imp = pd.DataFrame({"feature": feat_cols, "importance": overall})
        for i, emo in enumerate(config.EMOTIONS):
            imp[f"w_{emo}"] = mean_abs[i]
        imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
        imp.to_csv(config.RESULTS_DIR / "feature_importance.csv", index=False)
        print(f"\nTop {TOP_K_FEATURES} eGeMAPS features (|weight|, linear SVM):")
        print(imp.head(TOP_K_FEATURES)[["feature", "importance"]].to_string(index=False))

    print(f"\n[ok] results written to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
