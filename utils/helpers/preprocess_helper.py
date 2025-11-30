from data.constants import UNIFIED_COLUMNS, EMAIL_REGEX, URL_REGEX, BINARY_TYPES
from html import unescape
import re
import pandas as pd

    
def extract_email_only(s: str) -> str:
    if not isinstance(s, str):
        return ""
    m = EMAIL_REGEX.search(s)
    return m.group(0) if m else ""

def clean_payload(payload, charset=None):
    if isinstance(payload, bytes):
        try:
            return payload.decode(charset or "utf-8", errors="ignore")
        except:
            return payload.decode("latin-1", errors="ignore")
    return str(payload)

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

def is_real_attachment(part):

    ctype = part.get_content_type()
    disp = str(part.get("Content-Disposition", "") or "").lower()
    filename = part.get_filename()

    # 1. Explicit attachments (best signal)
    if "attachment" in disp:
        return True

    # 2. Filename but only if disposition is attachment-like
    if filename:
        if "attachment" in disp:
            return True
        return False

    # 3. True binary document types
    if ctype in BINARY_TYPES:
        return True
    
    return False

# ---- Extract header fields manually by regex ----
def extract_raw_header_field(raw_headers: bytes, name: str) -> str:
    pattern = rb"(?im)^" + name.encode() + rb":\s*(.+)$"
    m = re.search(pattern, raw_headers)
    if not m:
        return ""
    try:
        return m.group(1).decode("utf-8", errors="ignore").strip()
    except:
        return ""
    

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

# Merge / Clean
def clean_and_merge(dfs):
    dfs = [df for df in dfs if df is not None and not df.empty]
    if not dfs:
        print("[ERROR] No datasets loaded!")
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    final = pd.concat(dfs, ignore_index=True)

    # Compare true attachments vs extracted attachments
    true_attach = final[final["headers_raw"].str.contains("Content-Disposition:", case=False, na=False)]
    extracted_attach = final[final["attachment_text"].str.len() > 0]
    
    print("Headers suggest attachments :", len(true_attach))
    print("Parser extracted attachments :", len(extracted_attach))
    
    missing = true_attach[true_attach["attachment_text"].str.len() == 0]
    
    print("\nPossible missing extractions:", len(missing))
    print(missing[["subject", "from_email"]].head(20))


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