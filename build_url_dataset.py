#!/usr/bin/env python

import os
import re
import random
import pandas as pd
from utils.helpers.url_filter import normalize_url_min, defang_cleanup, keep_domain_like
from data.constants import MALICIOUS_URL_LINKS

# --------------------- CONFIG ---------------------

SEED = 42
random.seed(SEED)
DATA_ROOT = "datasets/url_feeds"
OUT_PATH = "datasets/url_feeds/processed"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# Per-source caps (set to None for no limit)
MAX_PER_PHISH_SOURCE = None
MAX_PER_LEGIT_SOURCE = None

def load_urlhaus_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[WARN] URLHaus CSV not found: {path}")
        return pd.DataFrame(columns=["url", "label"])

    print(f"[INFO] Loading URLHaus data from: {path}")

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            df = pd.read_csv(
                f,
                comment="#",      # skip lines starting with '#'
                header=None,
                names=[
                    "id", "dateadded", "url", "url_status", "last_online",
                    "threat", "tags", "urlhaus_link", "reporter"
                ],
                usecols=["url"],
            )
    except Exception as e:
        print(f"[ERROR] Cannot load URLHaus CSV: {e}")
        return pd.DataFrame(columns=["url", "label"])

    df["url"] = df["url"].astype(str).apply(defang_cleanup)
    df = df[df["url"].apply(keep_domain_like)]
    df["label"] = 1

    print(f"[INFO] URLHaus {path} rows kept: {len(df):,}")
    return df[["url", "label"]]

# use this function only when labels are string
def load_csv_generic(
    path: str,
    url_columns=["url", "urls", "link", "phish_url"],
    label_column_names=["label", "labels", "category", "type", "class"],
    phish_labels=["malicious", "phishing", "phish", "bad", "unsafe"],
    legit_labels=["legit","benign"]
) -> pd.DataFrame:

    if not os.path.exists(path):
        print(f"[WARN] CSV not found: {path}")
        return pd.DataFrame(columns=["url", "label"])

    print(f"[INFO] Loading from generic CSV loader, CSV: {path}")

    try:
        df = pd.read_csv(path, dtype=str)
    except Exception as e:
        print(f"[ERROR] Cannot load CSV: {e}")
        return pd.DataFrame(columns=["url", "label"])

    df = df.fillna("")

    url_col = None
    df_columns_lower = {c.lower(): c for c in df.columns}

    for candidate in url_columns:
        if candidate.lower() in df_columns_lower:
            url_col = df_columns_lower[candidate.lower()]
            break

    if url_col is None:
        print("[ERROR] No URL column found in CSV. Provide url_columns candidates.")
        print("Available columns:", list(df.columns))
        return pd.DataFrame(columns=["url", "label"])

    df["url"] = df[url_col].astype(str).apply(defang_cleanup)
    df = df[df["url"].apply(keep_domain_like)]

    label_col = None
    for candidate in label_column_names:
        if candidate.lower() in df_columns_lower:
            label_col = df_columns_lower[candidate.lower()]
            break

    if label_col:
        raw = df[label_col].astype(str).str.lower().str.strip()
        df["label"] = raw.apply(lambda x:
            1 if x in phish_labels
            else 0 if x in legit_labels
            else 1 
        )

        print(f"[INFO] Label column detected: {label_col}")
        print(df["label"].value_counts())
    else:
        print("[INFO] No label column found → assuming all entries are malicious.")
        df["label"] = 1

    out = df[["url", "label"]].copy()
    out = out[out["url"].astype(str).str.len() > 3]

    print(f"[INFO] Rows kept from {path}: {len(out):,}")
    print(out.head())

    return out

def _infer_domain_column_from_df(df: pd.DataFrame) -> pd.Series | None:
    """
    Try to detect which column in a headerless CSV looks like domains.
    Returns a Series or None.
    """
    best_col = None
    best_ratio = 0.0

    for col_name in df.columns:
        col = df[col_name].astype(str)
        # rows that "look like" domain: contain dot and no spaces
        mask = col.str.contains(r"[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}", regex=True)
        ratio = mask.mean()
        if ratio > best_ratio:
            best_ratio = ratio
            best_col = col_name

    if best_col is not None and best_ratio > 0.3:
        return df[best_col].astype(str)
    return None

# 3. Loaders for legitimate sources
def load_legit_from_csv(
    path: str, 
    is_all_legit: bool = True,
    label_column_names: list[str] = [],
) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[WARN] Legit CSV not found: {path}")
        return pd.DataFrame(columns=["url", "label"])
    
    if not is_all_legit and not label_column_names:
        raise Exception("is_all_legit set to False but label column is empty.\nplease specify at least one label column")

    print(f"[INFO] Loading legit CSV from: {path}")

    # First attempt: assume CSV has header
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            df = pd.read_csv(f, dtype=str)
    except Exception as e:
        print(f"[ERROR] Cannot load CSV {path}: {e}")
        return pd.DataFrame(columns=["url", "label"])

    df = df.fillna("")
    df_columns_lower = {c.lower(): c for c in df.columns}

    if is_all_legit:
        df["label"] = 0

    else:
        # Try to find existing label column
        label_col = None
        for cand in label_column_names:
            if cand in df_columns_lower:
                label_col = df_columns_lower[cand]
                break

        if label_col is None:
            print("[WARN] is_all_legit=False but no label column found. Returning empty.")
            return pd.DataFrame(columns=["url", "label"])

        df["label"] = df[label_col].astype(int) # assuming all labels are 0/1 (int)

    # Extract URL or domain column
    url_series = None

    # A) Explicit "url" column
    if "url" in df_columns_lower:
        url_series = df[df_columns_lower["url"]].astype(str).str.strip()

    else:
        # B) Domain-like columns
        dom_col = None
        for cand in ["domain", "host", "fqdn"]:
            if cand in df_columns_lower:
                dom_col = df_columns_lower[cand]
                break

        if dom_col:
            url_series = "http://" + df[dom_col].astype(str).str.strip()

    # C) Fallback → load without header
    if url_series is None:
        print("[INFO] Trying header=None heuristic for legit CSV...")

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            df2 = pd.read_csv(f, header=None, dtype=str)

        df2 = df2.fillna("")
        col = _infer_domain_column_from_df(df2)
        if col is None:
            print("[WARN] Cannot infer URL column from legit CSV; skipping.")
            return pd.DataFrame(columns=["url", "label"])

        url_series = "http://" + col.str.strip()
        df = pd.DataFrame({"url": url_series})
        if is_all_legit:
            df["label"] = 0
        else:
            print("[ERROR] is_all_legit=False but header=None CSV has no labels — cannot use fallback.")
            return pd.DataFrame(columns=["url", "label"])

    else:
        df["url"] = url_series

    # Normalize & Filter *AFTER* labels are attached
    df["url"] = df["url"].astype(str).apply(defang_cleanup)
    df = df[df["url"].apply(keep_domain_like)]
    df = df[df["url"].str.len() > 3]

    print(f"[INFO] Legit CSV rows kept for {path}: {len(df):,}")
    return df[["url", "label"]]


def load_from_text(path: str, label: int) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[WARN] TXT file not found: {path}")
        return pd.DataFrame(columns=["url", "label"])

    print(f"[INFO] Loading TXT database from: {path}")

    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = defang_cleanup(line)
            if keep_domain_like(line):
                # If it's a bare domain, prefix with http:// for consistency
                if not re.match(r"^https?://", line, flags=re.IGNORECASE):
                    line = "http://" + line
                rows.append(line)

    if not rows:
        print("[WARN] No usable lines found in TXT.")
        return pd.DataFrame(columns=["url", "label"])

    df = pd.DataFrame({"url": rows})
    df["label"] = int(label)
    print(f"[INFO] TXT rows kept: {len(df):,}")
    return df[["url", "label"]]

def cap_rows(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=SEED)

def build_stage3_dataset() -> pd.DataFrame:

    # ---- Define local file paths here ----
    urlhaus_full_csv        = os.path.join(DATA_ROOT, "urlhaus_full.csv")
    urlhaus_recent_csv      = os.path.join(DATA_ROOT, "urlhaus_recent.csv")
    urlhaus_online_csv      = os.path.join(DATA_ROOT, "urlhaus_online.csv")
    malicious_url_txt_list = []
    for filename in MALICIOUS_URL_LINKS.keys():
        url_txt_full_path = os.path.join(DATA_ROOT, filename)
        malicious_url_txt_list.append(url_txt_full_path)
    
    malicious_phish       = os.path.join(DATA_ROOT, "malicious_phish.csv")
    phishing_url_dataset  = os.path.join(DATA_ROOT, "phishing_url_dataset.csv")

    tranco_daily_csv      = os.path.join(DATA_ROOT, "tranco_1m_daily.csv")
    tranco_full_csv       = os.path.join(DATA_ROOT, "tranco_1m_full.csv")
    majestic_1m_csv       = os.path.join(DATA_ROOT, "majestic_1m.csv")
    umbrella_csv          = os.path.join(DATA_ROOT, "top-1m.csv")

    # ---- Load phishing sources ----
    phish_frames = []

    urlhaus_full_df   = load_urlhaus_csv(
        path=urlhaus_full_csv,
    )
    urlhaus_recent_df = load_urlhaus_csv(
        path=urlhaus_recent_csv,
    )
    urlhaus_online_df = load_urlhaus_csv(
        path=urlhaus_online_csv,
    )
    malicious_url_txt_df = []
    for full_path in malicious_url_txt_list:
        if os.path.exists(full_path):
            m_df = load_from_text(full_path, label=1)
            malicious_url_txt_df.append(m_df)
        else:
            print(f"{full_path} TXT file not found.\nmaybe you didn't download it from scripts/url_feeds_download.py?")

    malicious_phish_df = load_csv_generic(
        path=malicious_phish,
        url_columns=["url"],
        label_column_names=["type"],
        phish_labels=["phishing", "defacement"],
        legit_labels=["benign"]
    )

    phishing_url_dataset_df = load_legit_from_csv(
        path=phishing_url_dataset,
        is_all_legit=False,
        label_column_names=["label"]
    )

    for m_df in [urlhaus_full_df, urlhaus_recent_df, urlhaus_online_df, malicious_phish_df, phishing_url_dataset_df] + malicious_url_txt_df:
        m_df = cap_rows(m_df, MAX_PER_PHISH_SOURCE)
        if not m_df.empty:
            phish_frames.append(m_df)

    if phish_frames:
        phish_all = pd.concat(phish_frames, ignore_index=True)
    else:
        phish_all = pd.DataFrame(columns=["url", "label"])
    print(f"\n[INFO] Total phishing rows combined: {len(phish_all):,}")

    # ---- Load legit sources ----
    legit_frames = []

    tranco_daily_df     = load_legit_from_csv(tranco_daily_csv)
    tranco_full_df      = load_legit_from_csv(tranco_full_csv)
    majestic_df         = load_legit_from_csv(majestic_1m_csv)
    umbrella_df         = load_legit_from_csv(umbrella_csv)

    for l_df in [tranco_daily_df, tranco_full_df, umbrella_df, majestic_df]:
        l_df = cap_rows(l_df, MAX_PER_LEGIT_SOURCE)
        if not l_df.empty:
            legit_frames.append(l_df)

    if legit_frames:
        legit_all = pd.concat(legit_frames, ignore_index=True)
    else:
        legit_all = pd.DataFrame(columns=["url", "label"])
    print(f"[INFO] Total legit rows combined:    {len(legit_all):,}")

    # ---- Combine phishing + legit ----
    combined = pd.concat([phish_all, legit_all], ignore_index=True)
    if combined.empty:
        print("[ERROR] No data loaded at all. Check your source files.")
        return combined

    # Normalize for dedupe
    combined["url_norm"] = combined["url"].astype(str).apply(normalize_url_min)

    before = len(combined)
    combined = combined[combined["url_norm"].str.len() > 0]
    print(f"[INFO] Dropped {before - len(combined):,} rows with empty normalized URL")

    before = len(combined)
    combined = combined.drop_duplicates(subset=["url_norm", "label"], keep="first")
    print(f"[INFO] Dropped {before - len(combined):,} duplicate (url_norm, label) rows")

    combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Basic stats
    print("\n[INFO] Final dataset stats:")
    print(combined["label"].value_counts())
    print(f"Total rows: {len(combined):,}")

    # For Stage-3 CharCNN, we only need "url" and "label"
    final_df = combined[["url", "label"]].copy()
    return final_df

def main():
    print("=== Stage-3 URL Dataset Builder ===")
    df = build_stage3_dataset()
    if df.empty:
        print("[ERROR] Final dataset is empty; not writing CSV.")
        return

    os.makedirs(OUT_PATH, exist_ok=True)
    df.to_csv(os.path.join(OUT_PATH, "final_urls.csv"), index=False, encoding="utf-8")
    print(f"\n[OK] Stage-3 URL dataset written to: {OUT_PATH}")
    print(df.head(10))


if __name__ == "__main__":
    main()
