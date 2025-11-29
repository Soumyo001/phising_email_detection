#!/usr/bin/env python3
"""
preprocess.py (print-only version)

Build a unified phishing-email dataset from:
- CSV datasets (subject/body/label/sender/receiver)
- .eml datasets (headers/body/attachments)

Output: data/processed/final_master.csv
"""

import os
import re
import uuid
from email import policy
from email.parser import BytesParser
from data.configs import CSV_DATASETS, EML_DATASETS
from data.constants import UNIFIED_COLUMNS, OUT_DIR
import utils.helpers.preprocess_helper as helper
import pandas as pd

def extract_header_anomalies(parsed: dict) -> dict:
    """
    Compute header-anomaly features based on fields in `parsed`.
    Works for both:
      - EML rows (with headers_raw filled)
      - CSV rows (headers_raw is "" but from_email/date/etc. still useful)
    """

    subject = parsed.get("subject", "") or ""
    from_email = parsed.get("from_email", "") or ""
    to_email = parsed.get("to_email", "") or ""
    reply_to_email = parsed.get("reply_to_email", "") or ""
    date = parsed.get("date", "") or ""
    headers_raw = parsed.get("headers_raw", "") or ""

    headers_empty = not headers_raw.strip()

    # 1) From ≠ Reply-To
    from_reply_mismatch = 0
    if from_email and reply_to_email and from_email.lower() != reply_to_email.lower():
        from_reply_mismatch = 1

    # 2) From domain vs Return-Path domain
    from_domain = from_email.split("@")[-1].lower() if "@" in from_email else ""
    return_path_domain = ""
    if not headers_empty:
        m = re.search(r"Return-Path:\s*<?([\w\.-]+@[\w\.-]+)>?", headers_raw, flags=re.IGNORECASE)
        if m:
            return_path_domain = m.group(1).split("@")[-1].lower()

    domain_mismatch = 0
    if return_path_domain and from_domain and return_path_domain != from_domain:
        domain_mismatch = 1

    # 3) Suspicious TLD
    SUSPICIOUS_TLDS = (".xyz", ".top", ".gq", ".ml", ".ga", ".tk")
    suspicious_tld = 1 if any(from_domain.endswith(tld) for tld in SUSPICIOUS_TLDS) else 0

    # 4) Number of Received: headers
    if headers_empty:
        received_count = 0
    else:
        received_count = len(re.findall(r"^Received:", headers_raw, flags=re.IGNORECASE | re.MULTILINE))

    # 5) To header anomaly (missing or multiple) – only if we actually have headers_raw
    if headers_empty:
        header_to_anomaly = 0
    else:
        to_headers = re.findall(r"^To:", headers_raw, flags=re.IGNORECASE | re.MULTILINE)
        to_count = len(to_headers)
        header_to_anomaly = 1 if to_count != 1 else 0

    # 6) X-Mailer anomaly
    if headers_empty:
        x_mailer_anomaly = 0
    else:
        x_mailer = ""
        m = re.search(r"^X-Mailer:\s*(.*)$", headers_raw, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            x_mailer = m.group(1).strip()
        x_mailer_anomaly = 1 if (not x_mailer or "php" in x_mailer.lower() or "unknown" in x_mailer.lower()) else 0

    # 7) Malformed / missing date
    malformed_date = 0
    if not date or len(date) < 10:
        malformed_date = 1

    return {
        "header_from_reply_mismatch": from_reply_mismatch,
        "header_domain_mismatch": domain_mismatch,
        "header_suspicious_tld": suspicious_tld,
        "header_received_count": received_count,
        "header_to_anomaly": header_to_anomaly,
        "header_x_mailer_anomaly": x_mailer_anomaly,
        "header_date_malformed": malformed_date,
    }

# EML parsing
def parse_eml_file(path: str):
    """High-quality EML parsing with RAW header extraction (safe for malformed phishing headers)."""

    # --------------------- RAW HEADER EXTRACTION -----------------------
    try:
        with open(path, "rb") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[WARN] Cannot read {path}: {e}")
        return None

    raw_header_block = b""
    raw_body_block = b""
    sep_index = 0

    # Split headers and body manually (no Python email parsing!)
    for i, line in enumerate(lines):
        raw_header_block += line
        if line.strip() == b"":     # blank line → end of headers
            sep_index = i + 1
            break

    raw_body_block = b"".join(lines[sep_index:])

    raw_subject    = helper.extract_raw_header_field(raw_header_block, "Subject")
    raw_from       = helper.extract_raw_header_field(raw_header_block, "From")
    raw_to         = helper.extract_raw_header_field(raw_header_block, "To")
    raw_reply_to   = helper.extract_raw_header_field(raw_header_block, "Reply-To")
    raw_date       = helper.extract_raw_header_field(raw_header_block, "Date")

    subject        = raw_subject
    from_email     = helper.extract_email_only(raw_from)
    to_email       = helper.extract_email_only(raw_to)
    reply_to_email = helper.extract_email_only(raw_reply_to)
    date           = raw_date

    # Store raw headers
    try:
        headers_raw = raw_header_block.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARN] header decode failed for {path}: {e}")
        headers_raw = ""

    # Use Python parser ONLY for the body (not the headers)
    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw_body_block)
    except Exception as e:
        print(f"[WARN] Body parse failed for {path}: {e}")
        msg = None

    body_text_parts = []
    attachment_text_parts = []

    if msg and msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition", "") or "").lower()

            # Attachments
            if "attachment" in disp or part.get_filename():
                try:
                    payload = part.get_content()
                    if isinstance(payload, bytes):
                        payload = payload.decode(part.get_content_charset() or "utf-8", errors="ignore")
                    if isinstance(payload, str):
                        attachment_text_parts.append(helper.clean_html_simple(payload))
                except:
                    pass
                continue

            # Inline parts
            try:
                payload = part.get_content()
            except:
                payload = None

            if payload is None:
                continue

            if isinstance(payload, bytes):
                payload = payload.decode(part.get_content_charset() or "utf-8", errors="ignore")

            if isinstance(payload, str):
                if ctype == "text/plain":
                    body_text_parts.append(payload)
                elif ctype == "text/html":
                    body_text_parts.append(helper.clean_html_simple(payload))

    elif msg:
        try:
            payload = msg.get_content()
        except:
            payload = ""

        if isinstance(payload, bytes):
            payload = payload.decode(msg.get_content_charset() or "utf-8", errors="ignore")

        if isinstance(payload, str):
            if msg.get_content_type() == "text/html":
                body_text_parts.append(helper.clean_html_simple(payload))
            else:
                body_text_parts.append(payload)

    body_text = " ".join([p for p in body_text_parts if p]).strip()
    attachment_text = " ".join([p for p in attachment_text_parts if p]).strip()

    # URL extraction
    url_list = []
    url_list.extend(helper.extract_urls(subject))
    url_list.extend(helper.extract_urls(body_text))

    # Deduplicate URLs
    seen = set()
    uniq_urls = []
    for u in url_list:
        if u not in seen:
            uniq_urls.append(u)
            seen.add(u)

    return {
        "id": str(uuid.uuid4()),
        "subject": subject,
        "body_text": body_text,
        "attachment_text": attachment_text,
        "headers_raw": headers_raw,
        "from_email": from_email,
        "to_email": to_email,
        "reply_to_email": reply_to_email,
        "date": date,
        "urls": " ".join(uniq_urls),
    }

def load_eml_dir(config: dict) -> pd.DataFrame:
    root = config["root_dir"]
    name = config["name"]
    label_for_all = config.get("label_for_all", None)

    rows = []

    print(f"[INFO] Loading EML dataset: {name} from {root}")

    for dirpath, _, files in os.walk(root):
        for fname in files:
            if not fname.lower().endswith(".eml"):
                continue

            fpath = os.path.join(dirpath, fname)
            data = parse_eml_file(fpath)
            if data is None:
                continue
            
            # Compute header anomaly features
            anoms = extract_header_anomalies(data)

            row = {
                **data,    
                **anoms,         
                "label": label_for_all,
                "source": name,
            }
            rows.append(row)

    if not rows:
        print(f"[WARN] No EML rows found in {root}")
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    df = pd.DataFrame(rows)
    for col in UNIFIED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[UNIFIED_COLUMNS]
    print(f"[INFO] Loaded {len(df)} EML rows from {root}")
    return df

# CSV processing
def unify_row_from_csv(row, cfg):
    subject = str(row[cfg["subject_col"]]) if cfg["subject_col"] in row and pd.notna(row[cfg["subject_col"]]) else ""
    body_text = str(row[cfg["body_col"]]) if cfg["body_col"] in row and pd.notna(row[cfg["body_col"]]) else ""

    from_email = helper.extract_email_only(row[cfg["sender_col"]]) if cfg.get("sender_col") in row else ""
    to_email = helper.extract_email_only(row[cfg["receiver_col"]]) if cfg.get("receiver_col") in row else ""

    raw_label = row[cfg["label_col"]] if cfg["label_col"] in row else None
    label = helper.normalize_label(raw_label, cfg.get("pos_labels"), cfg.get("neg_labels"))
    if label is None:
        return None

    urls = []
    urls.extend(helper.extract_urls(subject))
    urls.extend(helper.extract_urls(body_text))
    seen = set()
    uniq = [u for u in urls if not (u in seen or seen.add(u))]
    urls_str = " ".join(uniq)

    date = str(row[cfg["date_col"]]).strip() if cfg.get("date_col") in row else ""

    base = {
        "id": str(uuid.uuid4()),
        "subject": subject.strip(),
        "body_text": body_text.strip(),
        "attachment_text": "",
        "headers_raw": "",          # no raw headers in CSV
        "from_email": from_email,
        "to_email": to_email,
        "reply_to_email": "",       # CSV usually won't have this
        "date": date,
        "urls": urls_str,
        "label": label,
        "source": cfg["name"],
    }

    anoms = extract_header_anomalies(base)
    base.update(anoms)

    return base

def load_csv_dataset(cfg: dict) -> pd.DataFrame:
    path = cfg["path"]
    if not os.path.isfile(path):
        print(f"[WARN] CSV not found: {path}")
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    print(f"[INFO] Loading CSV dataset: {cfg['name']} from {path}")
  
    rows = []

    try:
        # Try utf-8 with chunks
        reader = pd.read_csv(
            path,
            chunksize=5000,      # process 5000 rows at a time
            dtype=str,           # keep everything as string to avoid dtype guessing overhead
            low_memory=False
        )
    except UnicodeDecodeError:
        # Fallback to latin-1 with chunks
        reader = pd.read_csv(
            path,
            chunksize=5000,
            dtype=str,
            low_memory=False,
            encoding="latin-1"
        )

    # Process chunk by chunk
    total_rows = 0
    for df_raw in reader:
        for _, row in df_raw.iterrows():
            x = unify_row_from_csv(row, cfg)
            if x is not None:
                if x["subject"] or x["body_text"]:
                    rows.append(x)
        total_rows += len(df_raw)

    if not rows:
        print(f"[WARN] All rows dropped for {cfg['name']}")
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    df = pd.DataFrame(rows)
    df = df[UNIFIED_COLUMNS]
    print(f"[INFO] Loaded {len(df)} rows from {cfg['name']} (scanned {total_rows} raw rows)")
    return df


# Merge / Clean
def clean_and_merge(dfs):
    dfs = [df for df in dfs if df is not None and not df.empty]
    if not dfs:
        print("[ERROR] No datasets loaded!")
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    final = pd.concat(dfs, ignore_index=True)

    # Ensure all columns exist
    for c in UNIFIED_COLUMNS:
        if c not in final.columns:
            final[c] = None

    # Drop missing labels
    before = len(final)
    final = final[final["label"].notna()]
    print(f"[INFO] Dropped {before - len(final)} rows with missing label")

    # int label
    final["label"] = final["label"].astype(int)

    # Drop rows with empty subject+body_text
    before = len(final)
    mask = final["subject"].astype(str).str.strip().astype(bool) | final["body_text"].astype(str).str.strip().astype(bool)
    final = final[mask]
    print(f"[INFO] Dropped {before - len(final)} empty content rows")

    # Deduplicate
    before = len(final)
    final = final.drop_duplicates(subset=["from_email", "subject", "body_text"], keep="first")
    print(f"[INFO] Dropped {before - len(final)} duplicate rows")

    final = final.reset_index(drop=True)
    return final

# Main
def main():
    all_dfs = []

    # Load CSV datasets
    for cfg in CSV_DATASETS:
        all_dfs.append(load_csv_dataset(cfg))

    # Load EML datasets
    for cfg in EML_DATASETS:
        all_dfs.append(load_eml_dir(cfg))

    # Merge
    master = clean_and_merge(all_dfs)

    
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "final_combined.csv")
    master.to_csv(out_path, index=False, encoding="utf-8")

    print(f"[DONE] Saved unified dataset with {len(master)} rows → {out_path}")


if __name__ == "__main__":
    main()