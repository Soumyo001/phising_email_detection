#!/usr/bin/env python

import pandas as pd

def build_transformer_sample(INPUT_PATH, OUTPUT_PATH, TARGET_COUNT, SEED):
    print("=== Stage-3 Transformer Sampling ===")

    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=["url", "label"])
    df["label"] = df["label"].astype(int)

    total_rows = len(df)
    print(f"[INFO] Loaded dataset: {total_rows:,} rows")

    # Shuffle full dataset
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    if TARGET_COUNT >= total_rows:
        print("[WARN] Target count >= dataset size. Using all rows.")
        sampled = df
    else:
        # Balanced fractional sampling
        frac = TARGET_COUNT / total_rows

        legit_df   = df[df["label"] == 0]
        phish_df   = df[df["label"] == 1]

        n_legit = int(len(legit_df) * frac)
        n_phish = TARGET_COUNT - n_legit

        sampled_legit = legit_df.sample(n=n_legit, random_state=SEED)
        sampled_phish = phish_df.sample(n=n_phish, random_state=SEED)

        sampled = pd.concat([sampled_legit, sampled_phish], axis=0)
        sampled = sampled.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"[INFO] Sampled dataset rows: {len(sampled):,}")
    print(sampled["label"].value_counts())

    sampled.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"[OK] Saved balanced sample → {OUTPUT_PATH}")
