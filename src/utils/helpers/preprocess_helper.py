from data.constants import EMAIL_REGEX, URL_REGEX
from html import unescape
import re
import pandas as pd

    
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