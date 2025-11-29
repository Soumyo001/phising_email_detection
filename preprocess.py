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
from data.constants import UNIFIED_COLUMNS, EMAIL_REGEX, URL_REGEX, OUT_DIR
from html import unescape
import pandas as pd


# Helper functions
def extract_email_only(s: str) -> str:
    if not isinstance(s, str):
        return ""
    m = EMAIL_REGEX.search(s)
    return m.group(0) if m else ""


def clean_html_simple(html: str) -> str:
    if not isinstance(html, str):
        return ""
    html = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", " ", html)
    text = re.sub(r"(?s)<.*?>", " ", html)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_urls(text: str) -> list:
    if not isinstance(text, str):
        return []
    return URL_REGEX.findall(text)


def normalize_label(raw, pos_values=None, neg_values=None):
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None

    if pos_values is None:
        pos_values = [1, "1", "spam", "phishing", "malicious", "phish"]
    if neg_values is None:
        neg_values = [0, "0", "ham", "legit", "benign", "normal"]

    s = str(raw).strip().lower()
    if s in {str(v).lower() for v in pos_values}:
        return 1
    if s in {str(v).lower() for v in neg_values}:
        return 0

    try:
        n = int(float(raw))
        if n == 1:
            return 1
        if n == 0:
            return 0
    except:
        pass

    return None

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
    """High-quality EML parsing with support for nested MIME, HTML decoding, attachments, etc."""
    try:
        with open(path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)
    except Exception as e:
        print(f"[WARN] Cannot parse {path}: {e}")
        return None

    # subject = msg.get("subject", "") or ""
    # from_email = extract_email_only(msg.get("from", ""))
    # to_email = extract_email_only(msg.get("to", ""))
    # date = msg.get("date", "") or ""

    subject_headers = msg.get_all("subject", [])
    subject = subject_headers[-1] if subject_headers else ""
    subject = subject.strip() if subject else ""

    from_headers = msg.get_all("from", [])
    raw_from = from_headers[-1] if from_headers else ""
    from_email = extract_email_only(raw_from)

    # TO (same multi-header handling)
    to_headers = msg.get_all("to", [])
    raw_to = to_headers[-1] if to_headers else ""
    to_email = extract_email_only(raw_to)

    reply_to_headers = msg.get_all("reply-to", [])
    raw_reply_to = reply_to_headers[-1] if reply_to_headers else ""
    reply_to_email = extract_email_only(raw_reply_to)

    date_headers = msg.get_all("date", [])
    date = date_headers[-1] if date_headers else ""
    date = date.strip() if date else ""

    headers_raw = str(msg)

    body_text_parts = []
    attachment_text_parts = []

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition", "") or "").lower()

            # Attachments
            if "attachment" in disp or part.get_filename():
                try:
                    payload = part.get_content()
                    if isinstance(payload, bytes):
                        try:
                            payload = payload.decode(part.get_content_charset() or "utf-8", errors="ignore")
                        except:
                            payload = payload.decode("utf-8", errors="ignore")
                    if isinstance(payload, str):
                        cleaned = clean_html_simple(payload)
                        attachment_text_parts.append(cleaned)
                except:
                    pass
                continue

            # Inline body parts
            try:
                payload = part.get_content()
            except:
                payload = None

            if payload is None:
                continue

            if isinstance(payload, bytes):
                try:
                    payload = payload.decode(part.get_content_charset() or "utf-8", errors="ignore")
                except:
                    payload = payload.decode("utf-8", errors="ignore")

            if not isinstance(payload, str):
                continue

            if ctype == "text/plain":
                body_text_parts.append(payload)
            elif ctype == "text/html":
                body_text_parts.append(clean_html_simple(payload))

    else:
        try:
            payload = msg.get_content()
        except:
            payload = ""

        if isinstance(payload, bytes):
            try:
                payload = payload.decode(msg.get_content_charset() or "utf-8", errors="ignore")
            except:
                payload = payload.decode("utf-8", errors="ignore")

        if msg.get_content_type() == "text/html":
            body_text_parts.append(clean_html_simple(payload))
        else:
            body_text_parts.append(payload)

    body_text = " ".join([p for p in body_text_parts if p]).strip()
    attachment_text = " ".join([p for p in attachment_text_parts if p]).strip()

    # URL extraction
    url_list = []
    url_list.extend(extract_urls(subject))
    url_list.extend(extract_urls(body_text))

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

    from_email = extract_email_only(row[cfg["sender_col"]]) if cfg.get("sender_col") in row else ""
    to_email = extract_email_only(row[cfg["receiver_col"]]) if cfg.get("receiver_col") in row else ""

    raw_label = row[cfg["label_col"]] if cfg["label_col"] in row else None
    label = normalize_label(raw_label, cfg.get("pos_labels"), cfg.get("neg_labels"))
    if label is None:
        return None

    urls = []
    urls.extend(extract_urls(subject))
    urls.extend(extract_urls(body_text))
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

    try:
        df_raw = pd.read_csv(path)
    except UnicodeDecodeError:
        df_raw = pd.read_csv(path, encoding="latin-1")

    rows = []
    for _, row in df_raw.iterrows():
        x = unify_row_from_csv(row, cfg)
        if x is not None:
            if x["subject"] or x["body_text"]:
                rows.append(x)

    if not rows:
        print(f"[WARN] All rows dropped for {cfg['name']}")
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    df = pd.DataFrame(rows)
    df = df[UNIFIED_COLUMNS]
    print(f"[INFO] Loaded {len(df)} rows from {cfg['name']}")
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