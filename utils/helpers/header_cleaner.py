import re
from data.constants import MAX_HEADER_LEN, MAX_RECEIVED_LINES


def truncate_header(h: str, max_len: int = MAX_HEADER_LEN) -> str:
    """Truncate overly long raw headers to prevent transformer leakage."""
    if not isinstance(h, str):
        return ""
    return h[:max_len]

def limit_received_headers(h: str, max_lines: int = MAX_RECEIVED_LINES) -> str:
    """Limit Received: lines to avoid forensic-chain length bias."""
    if not isinstance(h, str):
        return ""
    lines = h.splitlines()
    kept = []
    rcv = 0
    for L in lines:
        if L.lower().startswith("received:"):
            if rcv < max_lines:
                kept.append(L)
                rcv += 1
        else:
            kept.append(L)
    return "\n".join(kept)

def remove_b64_and_weird_encoding(h: str) -> str:
    """Remove base64-like header fields that give away phishing."""
    if not isinstance(h, str):
        return ""
    h = re.sub(r"=\?utf-8.*?\?=", "", h, flags=re.IGNORECASE)
    h = re.sub(r"[A-Za-z0-9+/]{20,}={0,2}", "", h)   # base64 chunks
    return h

def collapse_blank_lines(h: str) -> str:
    """Normalize whitespace."""
    if not isinstance(h, str):
        return ""
    h = re.sub(r"\n\s*\n", "\n", h)
    return h.strip()

def clean_header(h: str) -> str:
    """Master cleaning pipeline."""
    h = truncate_header(h)
    h = limit_received_headers(h)
    h = remove_b64_and_weird_encoding(h)
    h = collapse_blank_lines(h)
    return h
