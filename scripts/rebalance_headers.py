import pandas as pd
import numpy as np
from data.constants import MAX_RAW_LENGTH, MAX_CANON_LENGTH

def normalize_lengths(df):
    # HARD TRUNCATION
    df["headers_raw"] = df["headers_raw"].astype(str).str.slice(0, MAX_RAW_LENGTH)
    df["canonical_raw"] = df["canonical_raw"].astype(str).str.slice(0, MAX_CANON_LENGTH)

    # OPTIONAL NORMALIZATION: pad short headers so transformer cannot guess length
    df["headers_raw"] = df["headers_raw"].apply(lambda x: x.ljust(MAX_RAW_LENGTH, " "))
    df["canonical_raw"] = df["canonical_raw"].apply(lambda x: x.ljust(MAX_CANON_LENGTH, " "))

    return df

def rebalance(df):
    # Prevent dataset imbalance leakage
    ham = df[df.label == 0]
    phish = df[df.label == 1]

    min_size = min(len(ham), len(phish))

    ham_bal = ham.sample(min_size, random_state=42)
    phish_bal = phish.sample(min_size, random_state=42)

    df_bal = pd.concat([ham_bal, phish_bal]).sample(frac=1, random_state=42).reset_index(drop=True)

    print("Balanced dataset:", df_bal.label.value_counts())
    return df_bal

if __name__ == "__main__":
    df = pd.read_csv("datasets/processed/final_combined.csv")
    df = normalize_lengths(df)
    # df_bal = rebalance(df)
    df.to_csv("datasets/processed/final_normalized.csv", index=False) # use this csv only for header trainning
