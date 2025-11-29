# dataset.py
"""
Dataset utilities for the phishing email multimodal model.

Produces tensors matching the architecture:
- subject_texts: list[str] (passed to text encoder/tokenizer)
- body_texts: list[str]
- attachment_texts: list[list[str]]
- header_token_ids: LongTensor [L_hdr]
- header_field_ids: LongTensor [L_hdr]
- header_mask: LongTensor [L_hdr]
- url_char_seqs: LongTensor [U_max, char_seq_len]
- sender_idx: LongTensor []
- domain_idx: LongTensor []
- sender_numeric: FloatTensor [8]  (7 anomaly flags + normalized received_count)
- label: FloatTensor []
- source_idx: LongTensor [] (optional)

Usage:
  ds = UnifiedEmailDataset(csv_path="data/processed/final_combined.csv", mappings_dir="mappings/")
  loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
"""

import os
import re
import json
import math
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# Columns in final_combined.csv (from your preprocess.py).
UNIFIED_COLUMNS = [
    "id",
    "subject",
    "body_text",
    "attachment_text",
    "headers_raw",
    "from_email",
    "to_email",
    "reply_to_email",
    "date",
    "urls",
    "header_from_reply_mismatch",
    "header_domain_mismatch",
    "header_suspicious_tld",
    "header_received_count",
    "header_to_anomaly",
    "header_x_mailer_anomaly",
    "header_date_malformed",
    "label",
    "source",
]

# Field identifiers for headers raw parsing
HEADER_FIELD_NAMES = [
    "FROM", "TO", "REPLY_TO", "RETURN_PATH", "RECEIVED", "X_MAILER", "DATE", "OTHER"
]
FIELD_NAME_TO_ID = {n: i for i, n in enumerate(HEADER_FIELD_NAMES)}

# default unknown / pad ids
PAD_ID = 0
UNK_ID = 1

# -----------------------
# Helpers
# -----------------------
def safe_lower(s):
    return s.lower() if isinstance(s, str) else ""

def normalize_received_count(x):
    # simple normalization: log(1 + x)
    try:
        v = float(x)
        return math.log1p(max(0.0, v))
    except:
        return 0.0

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# -----------------------
# Header tokenizer & vocab creation
# -----------------------
def simple_header_tokenize(headers_raw):
    """
    Split headers_raw into a list of (field_name, token) pairs.
    We split by lines, detect header field via regex, and tokenize the remainder on whitespace/punctuation.
    """
    if not headers_raw or not isinstance(headers_raw, str) or not headers_raw.strip():
        return []

    toks = []
    lines = headers_raw.splitlines()
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        # detect field name (e.g., "From:", "Received:", "X-Mailer:")
        m = re.match(r"^\s*([A-Za-z0-9\-]+)\s*:\s*(.*)$", ln)
        if m:
            field = m.group(1).strip()
            value = m.group(2).strip()
            # normalize field to our set
            f = "OTHER"
            if field.lower() == "from":
                f = "FROM"
            elif field.lower() == "to":
                f = "TO"
            elif field.lower() in ("reply-to", "reply_to", "replyto"):
                f = "REPLY_TO"
            elif field.lower() == "return-path":
                f = "RETURN_PATH"
            elif field.lower() == "received":
                f = "RECEIVED"
            elif field.lower() == "x-mailer":
                f = "X_MAILER"
            elif field.lower() == "date":
                f = "DATE"
            # split value into lightweight tokens (split on non-word, preserve emails/urls as tokens)
            # keep emails and urls intact
            parts = re.findall(r"[A-Za-z0-9\._%+-]+@[A-Za-z0-9\.-]+|https?://\S+|www\.\S+|[A-Za-z0-9\-_]+", value)
            if not parts:
                # fallback: split by whitespace
                parts = re.split(r"\s+", value)
            for p in parts:
                if not p:
                    continue
                toks.append((f, p))
        else:
            # line without colon — treat as OTHER
            parts = re.findall(r"[A-Za-z0-9\._%+-]+@[A-Za-z0-9\.-]+|https?://\S+|www\.\S+|[A-Za-z0-9\-_]+", ln)
            for p in parts:
                if not p:
                    continue
                toks.append(("OTHER", p))
    return toks  # list of (field_name, token_str)

def build_header_vocab_from_series(series_iterable, max_vocab=20000, min_freq=2):
    """
    Build a small header token vocabulary from a pandas Series of headers_raw strings.
    Returns dict token->id where id 0 = PAD, 1 = UNK, and others start at 2.
    """
    counter = Counter()
    for h in series_iterable:
        toks = simple_header_tokenize(h)
        for _, t in toks:
            counter[t.lower()] += 1
    # keep most common
    most = [tok for tok, freq in counter.most_common(max_vocab) if freq >= min_freq]
    vocab = {"<PAD>": PAD_ID, "<UNK>": UNK_ID}
    idx = 2
    for tok in most:
        vocab[tok] = idx
        idx += 1
    return vocab

# -----------------------
# URL char sequence helper
# -----------------------
def url_to_char_seq(url, char_seq_len=200):
    """
    Convert a single URL string into a fixed-length array of ints (0..255).
    Truncate/pad to char_seq_len. Non-ASCII chars are clipped via ord % 256.
    """
    seq = [0] * char_seq_len
    if not url or not isinstance(url, str):
        return seq
    s = url.strip()[:char_seq_len]
    for i, ch in enumerate(s):
        seq[i] = ord(ch) % 256
    # padding (already 0)
    return seq

def urls_field_to_matrix(urls_field, U_max=8, char_seq_len=200):
    """
    input: urls_field is a string with space-separated URLs (as your preprocess writes)
    output: numpy array [U_max, char_seq_len] dtype=int64
    """
    mat = np.zeros((U_max, char_seq_len), dtype=np.int64)
    if not urls_field or not isinstance(urls_field, str):
        return mat
    urls = urls_field.strip().split()
    for i, u in enumerate(urls[:U_max]):
        mat[i] = np.array(url_to_char_seq(u, char_seq_len), dtype=np.int64)
    return mat

# -----------------------
# Mappings persistence
# -----------------------
def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------
# Dataset class
# -----------------------
class UnifiedEmailDataset(Dataset):
    def __init__(
        self,
        csv_path,
        mappings_dir="mappings",
        build_mappings=True,
        header_vocab_max=20000,
        header_min_freq=2,
        L_hdr=128,
        U_max=8,
        char_seq_len=200,
        sender_min_freq=1,
        domain_min_freq=1,
        verbose=True,
    ):
        """
        csv_path: path to your final_combined.csv (produced by preprocess.py). :contentReference[oaicite:1]{index=1}
        build_mappings: if True, build header vocab & sender/domain maps and persist them in mappings_dir.
                        If False, mappings_dir must contain sender2id.json, domain2id.json, header_vocab.json
        """
        assert os.path.isfile(csv_path), f"CSV not found: {csv_path}"
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path, dtype=str, low_memory=False)
        # ensure expected columns exist; fill missing with empty strings
        for c in UNIFIED_COLUMNS:
            if c not in self.df.columns:
                self.df[c] = ""
        self.df = self.df[UNIFIED_COLUMNS].fillna("")
        self.n = len(self.df)
        self.mappings_dir = mappings_dir
        ensure_dir(self.mappings_dir)
        self.L_hdr = L_hdr
        self.U_max = U_max
        self.char_seq_len = char_seq_len

        # build or load header vocab
        header_vocab_path = os.path.join(mappings_dir, "header_vocab.json")
        sender_map_path = os.path.join(mappings_dir, "sender2id.json")
        domain_map_path = os.path.join(mappings_dir, "domain2id.json")
        source_map_path = os.path.join(mappings_dir, "source2id.json")

        if build_mappings or not (os.path.isfile(header_vocab_path) and os.path.isfile(sender_map_path) and os.path.isfile(domain_map_path)):
            # build header vocab
            if verbose:
                print("[INFO] Building header vocab and sender/domain mappings...")
            header_vocab = build_header_vocab_from_series(self.df["headers_raw"].astype(str).values, max_vocab=header_vocab_max, min_freq=header_min_freq)
            save_json(header_vocab, header_vocab_path)
            # build sender/domain maps
            senders = Counter([safe_lower(x) for x in self.df["from_email"].astype(str).values if x and x.strip()])
            domains = Counter()
            for s, c in senders.items():
                dom = s.split("@")[-1] if "@" in s else ""
                if dom:
                    domains[dom] += c
            # create maps sorted by freq
            sender2id = {"<UNK>": 0}
            domain2id = {"<UNK>": 0}
            idx = 1
            for s, f in senders.most_common():
                if f < sender_min_freq:
                    continue
                sender2id[s] = idx
                idx += 1
            idx = 1
            for d, f in domains.most_common():
                if f < domain_min_freq:
                    continue
                domain2id[d] = idx
                idx += 1
            save_json(sender2id, sender_map_path)
            save_json(domain2id, domain_map_path)
            # source map
            unique_sources = sorted(set(self.df["source"].astype(str).values))
            source2id = {s: i for i, s in enumerate(unique_sources)}
            save_json(source2id, source_map_path)
        else:
            header_vocab = load_json(header_vocab_path)
            sender2id = load_json(sender_map_path)
            domain2id = load_json(domain_map_path)
            source2id = load_json(source_map_path) if os.path.isfile(source_map_path) else {}

        # store mappings
        self.header_vocab = header_vocab
        self.sender2id = sender2id
        self.domain2id = domain2id
        self.source2id = source2id

        # reverse lookup if required
        # keep trivial sizes small
        if verbose:
            print(f"[INFO] Dataset loaded: {self.n} rows")
            print(f"[INFO] header_vocab size = {len(self.header_vocab)}")
            print(f"[INFO] sender_map size = {len(self.sender2id)}")
            print(f"[INFO] domain_map size = {len(self.domain2id)}")

    def __len__(self):
        return self.n

    def header_tokens_to_ids(self, headers_raw):
        """
        Convert headers_raw string to:
          - token_ids: [L_hdr] (int)
          - field_ids: [L_hdr] (int)
          - mask: [L_hdr] (1 for token, 0 for pad)
        Uses simple_header_tokenize to extract (field, token) pairs.
        Unknown tokens -> UNK_ID
        """
        token_ids = np.zeros(self.L_hdr, dtype=np.int64)
        field_ids = np.zeros(self.L_hdr, dtype=np.int64)
        mask = np.zeros(self.L_hdr, dtype=np.int64)

        toks = simple_header_tokenize(headers_raw)
        if not toks:
            return token_ids, field_ids, mask

        pos = 0
        for field_name, tok in toks:
            if pos >= self.L_hdr:
                break
            tid = self.header_vocab.get(tok.lower(), UNK_ID)
            fid = FIELD_NAME_TO_ID.get(field_name, FIELD_NAME_TO_ID["OTHER"])
            token_ids[pos] = int(tid)
            field_ids[pos] = int(fid)
            mask[pos] = 1
            pos += 1
        return token_ids, field_ids, mask

    def sender_to_idx(self, from_email):
        s = safe_lower(from_email)
        return int(self.sender2id.get(s, 0))

    def domain_to_idx(self, from_email):
        s = safe_lower(from_email)
        dom = s.split("@")[-1] if "@" in s else ""
        return int(self.domain2id.get(dom, 0))

    def header_anomaly_vector(self, row):
        # order: 7 flags + normalized received_count
        flags = [
            float(row.get("header_from_reply_mismatch") or 0.0),
            float(row.get("header_domain_mismatch") or 0.0),
            float(row.get("header_suspicious_tld") or 0.0),
            float(row.get("header_to_anomaly") or 0.0),
            float(row.get("header_x_mailer_anomaly") or 0.0),
            float(row.get("header_date_malformed") or 0.0),
            0.0  # reserve for future
        ]
        # header_received_count normalized
        rc = row.get("header_received_count", "") 
        rn = normalize_received_count(rc)
        flags[-1] = rn
        return np.array(flags, dtype=np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subj = str(row.get("subject", "") or "")
        body = str(row.get("body_text", "") or "")
        attach_txt = []
        at = row.get("attachment_text", "")
        if isinstance(at, str) and at.strip():
            # attachments were concatenated by preprocess; split heuristically by "\n\n" or '===='
            parts = re.split(r"\n{2,}|={3,}", at)
            attach_txt = [p.strip() for p in parts if p.strip()]
        headers_raw = str(row.get("headers_raw", "") or "")
        urls_field = str(row.get("urls", "") or "")
        sender = str(row.get("from_email", "") or "")
        source = str(row.get("source", "") or "")
        label = 1.0 if str(row.get("label", "0")).strip() in ("1", "True", "true", "yes") else 0.0

        # header tokenization
        hdr_tok_ids, hdr_field_ids, hdr_mask = self.header_tokens_to_ids(headers_raw)

        # url char seqs
        url_mat = urls_field_to_matrix(urls_field, U_max=self.U_max, char_seq_len=self.char_seq_len)

        # sender/domain indices
        sidx = self.sender_to_idx(sender)
        didx = self.domain_to_idx(sender)

        # header anomaly features (7-d)
        h_anom = self.header_anomaly_vector(row)  # np array length 7

        # source idx (optional)
        source_idx = self.source2id.get(source, -1) if hasattr(self, "source2id") else -1

        sample = {
            "subject_texts": subj,
            "body_texts": body,
            "attachment_texts": attach_txt,
            "attachment_images": None,
            "header_token_ids": torch.from_numpy(hdr_tok_ids).long(),
            "header_field_ids": torch.from_numpy(hdr_field_ids).long(),
            "header_mask": torch.from_numpy(hdr_mask).long(),
            "url_char_seqs": torch.from_numpy(url_mat).long(),
            "sender_idx": torch.tensor(sidx, dtype=torch.long),
            "domain_idx": torch.tensor(didx, dtype=torch.long),
            "sender_numeric": torch.from_numpy(h_anom).float(),
            "label": torch.tensor(float(label), dtype=torch.float),
            "source_idx": torch.tensor(int(source_idx) if source_idx >= 0 else -1, dtype=torch.long),
        }
        return sample

# -----------------------
# Collate function
# -----------------------
def collate_fn(batch):
    """
    Collate a list of samples into batched tensors.
    Leaves subject_texts / body_texts as lists of strings (tokenizers can be applied in-model).
    """
    B = len(batch)
    out = {}
    out["subject_texts"] = [b["subject_texts"] for b in batch]
    out["body_texts"] = [b["body_texts"] for b in batch]
    out["attachment_texts"] = [b["attachment_texts"] for b in batch]
    out["attachment_images"] = None  # not supported in this dataset file
    out["header_token_ids"] = torch.stack([b["header_token_ids"] for b in batch], dim=0)
    out["header_field_ids"] = torch.stack([b["header_field_ids"] for b in batch], dim=0)
    out["header_mask"] = torch.stack([b["header_mask"] for b in batch], dim=0)
    out["url_char_seqs"] = torch.stack([b["url_char_seqs"] for b in batch], dim=0)  # [B, U_max, char_seq_len]
    out["sender_idx"] = torch.stack([b["sender_idx"] for b in batch], dim=0)
    out["domain_idx"] = torch.stack([b["domain_idx"] for b in batch], dim=0)
    out["sender_numeric"] = torch.stack([b["sender_numeric"] for b in batch], dim=0)
    out["label"] = torch.stack([b["label"] for b in batch], dim=0)
    out["source_idx"] = torch.stack([b["source_idx"] for b in batch], dim=0)
    return out

# -----------------------
# Quick CLI example
# -----------------------
if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/processed/final_combined.csv", help="Path to your final CSV (from preprocess.py).")
    parser.add_argument("--mappings_dir", type=str, default="mappings", help="Where to save/load sender/domain/header vocab maps.")
    parser.add_argument("--build", action="store_true", help="Build mappings from scratch (default False if mappings exist).")
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    ds = UnifiedEmailDataset(csv_path=args.csv, mappings_dir=args.mappings_dir, build_mappings=args.build)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2, collate_fn=collate_fn)

    # iterate quickly
    for i, batch in enumerate(loader):
        print("Batch keys:", list(batch.keys()))
        print("header_token_ids shape:", batch["header_token_ids"].shape)
        print("url_char_seqs shape:", batch["url_char_seqs"].shape)
        print("sender_numeric shape:", batch["sender_numeric"].shape)
        if i >= 2:
            break

    print("Done. Mappings saved to", args.mappings_dir)
