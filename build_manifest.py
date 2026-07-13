"""Stage 0 - Build the dataset manifest from INESCO filenames.

Filename convention (see 'Guideline of File Naming INESCO Dataset'):
    {speaker}_{emotion_letter}{sentence_number}.wav
    e.g. mdpa_a250.wav -> speaker mdpa, anger, sentence 250

Emotion letter: h=happiness, a=anger, s=sadness.
Output: outputs/manifest.csv with one row per audio file.
"""
import re
import sys
import pandas as pd

import config

FILENAME_RE = re.compile(r"^(?P<speaker>[a-z]{4})_(?P<emo>[has])(?P<num>\d+)\.wav$")


def build_manifest() -> pd.DataFrame:
    rows = []
    skipped = []
    for wav in sorted(config.DATASET_ROOT.rglob("*.wav")):
        m = FILENAME_RE.match(wav.name)
        if not m:
            skipped.append(wav.name)
            continue
        speaker = m.group("speaker")
        emo_letter = m.group("emo")
        rows.append(
            {
                "filename": wav.name,
                "relpath": str(wav.relative_to(config.DATASET_ROOT)),
                "speaker": speaker,
                "gender": config.SPEAKER_GENDER.get(speaker, "unknown"),
                "emotion": config.EMOTION_MAP[emo_letter],
                "sentence_id": int(m.group("num")),
            }
        )

    if skipped:
        print(f"[warn] {len(skipped)} file(s) did not match the naming pattern:")
        for name in skipped[:10]:
            print(f"       - {name}")

    df = pd.DataFrame(rows).sort_values(["speaker", "sentence_id"]).reset_index(drop=True)
    return df


def main() -> None:
    if not config.DATASET_ROOT.is_dir():
        sys.exit(f"[error] dataset root not found: {config.DATASET_ROOT}")

    df = build_manifest()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.MANIFEST_CSV, index=False)

    # --- Summary -----------------------------------------------------------
    print(f"[ok] wrote {config.MANIFEST_CSV}  ({len(df)} clips)\n")
    print("Clips per speaker x emotion:")
    pivot = (
        df.pivot_table(index="speaker", columns="emotion", values="filename",
                       aggfunc="count", fill_value=0)
        .reindex(columns=config.EMOTIONS)
    )
    pivot["TOTAL"] = pivot.sum(axis=1)
    print(pivot.to_string())
    print("\nTotal per emotion:")
    print(df["emotion"].value_counts().reindex(config.EMOTIONS).to_string())


if __name__ == "__main__":
    main()
