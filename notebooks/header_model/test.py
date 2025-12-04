#!/usr/bin/env python
# coding: utf-8

# # Stage-4: Raw Header ML Pipeline (XGBoost + LightGBM Blended)
# Production-ready notebook for phishing detection using email raw headers

# ## 1️⃣ Imports & Setup
import os
import re
import uuid
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, precision_recall_curve,
    ConfusionMatrixDisplay
)
from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb
import lightgbm as lgb

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths
DATA_ROOT = "datasets/headers"
OUT_PATH  = "datasets/processed"
os.makedirs(OUT_PATH, exist_ok=True)

# ---------------------------
# 2️⃣ Raw header extraction
def _parse_eml_file(path: str):
    try:
        with open(path, "rb") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[WARN] Cannot read {path}: {e}")
        return None

    raw_header_block = b""
    sep_index = 0
    for i, line in enumerate(lines):
        raw_header_block += line
        if line.strip() == b"":  # blank line -> end of headers
            sep_index = i + 1
            break

    raw_body_block = b"".join(lines[sep_index:])
    try:
        headers_raw = (raw_header_block + raw_body_block).decode("utf-8", errors="ignore")
    except Exception:
        headers_raw = ""

    return headers_raw

# ---------------------------
# 3️⃣ Feature extraction (upgraded)
FREEMAIL_DOMAINS = {
    "gmail.com","yahoo.com","hotmail.com","outlook.com","live.com",
    "aol.com","icloud.com","mail.com","gmx.com","yandex.ru","yandex.com",
    "zoho.com","proton.me","protonmail.com"
}

HEADER_PRESENCE_LIST = [
    "from","to","subject","date","reply-to","sender","return-path",
    "received","message-id","authentication-results","dkim-signature",
    "domainkey-signature","received-spf","x-spam-flag","x-spam-status",
    "x-originating-ip","x-mailer","mime-version","content-type",
    "content-transfer-encoding","list-id","list-unsubscribe","x-priority"
]

def _extract_first_header_line(headers: str, field: str) -> str:
    pattern = re.compile(rf"^{re.escape(field)}\s*:(.*)$", re.IGNORECASE | re.MULTILINE)
    m = pattern.search(headers)
    return m.group(1).strip() if m else ""

def _extract_email_domain(text: str) -> str:
    m = re.search(r"[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})", text, re.IGNORECASE)
    return m.group(1).lower().strip("[]<>") if m else ""

def _is_freemail(domain: str) -> int:
    return int(domain in FREEMAIL_DOMAINS)

def _tld(domain: str) -> str:
    return domain.split(".")[-1] if domain and "." in domain else ""

def _token_stats(headers: str) -> dict:
    h = headers or ""
    length = len(h)
    lines = h.count("\n")+1
    digits = sum(c.isdigit() for c in h)
    uppers = sum(c.isupper() for c in h)
    lowers = sum(c.islower() for c in h)
    specials = sum(c in "!#$%&'*+,-./:;<=>?@[]^_`{|}~" for c in h)
    spaces = sum(c.isspace() for c in h)
    words = re.findall(r"\S+", h)
    num_words = len(words)
    avg_word_len = (sum(len(w) for w in words)/num_words) if num_words else 0
    return {
        "len_chars": length, "len_lines": lines, "count_digits": digits,
        "count_uppers": uppers, "count_lowers": lowers, "count_specials": specials,
        "count_spaces": spaces, "num_words": num_words, "avg_word_len": avg_word_len,
        "ratio_digits": digits/length if length>0 else 0.0,
        "ratio_specials": specials/length if length>0 else 0.0,
        "ratio_uppers": uppers/length if length>0 else 0.0
    }

def _presence_features(headers: str) -> dict:
    h_lower = headers.lower()
    feats = {}
    for name in HEADER_PRESENCE_LIST:
        feats[f"has_{name.replace('-', '_')}"] = int(bool(re.search(rf"^{re.escape(name)}\s*:", h_lower, flags=re.MULTILINE)))
    return feats

def _spf_dkim_dmarc_features(headers: str) -> dict:
    h = headers.lower()
    feats = {}
    for proto in ["spf","dkim","dmarc"]:
        res = re.search(rf"{proto}=(pass|fail|softfail|neutral|none|temperror|permerror)", h)
        val = res.group(1) if res else "none"
        for v in ["pass","fail","softfail","neutral","none","temperror","permerror"]:
            feats[f"{proto}_{v}"] = int(val==v)
    return feats

def _received_features(headers: str) -> dict:
    h = headers.lower()
    received_lines = re.findall(r"^received:", h, flags=re.MULTILINE)
    num_received = len(received_lines)
    private_ip_pattern = r"(10\.\d{1,3}\.\d{1,3}\.\d{1,3})|(192\.168\.\d{1,3}\.\d{1,3})|(172\.(1[6-9]|2[0-9]|3[0-1])\.\d{1,3}\.\d{1,3})"
    has_private_ip = int(bool(re.search(private_ip_pattern,h)))
    public_ips = re.findall(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", h)
    num_public_ips = len(public_ips)
    return {"num_received": num_received,"has_private_ip_in_received": has_private_ip,"num_public_ips": num_public_ips}

def _x_spam_features(headers: str) -> dict:
    h_lower = headers.lower()
    feats = {"has_x_spam_flag":0,"x_spam_flag_yes":0,"has_x_spam_status":0,"x_spam_status_yes":0}
    m_flag = re.search(r"^x-spam-flag:\s*(\S+)", h_lower, flags=re.MULTILINE)
    if m_flag: feats["has_x_spam_flag"]=1; feats["x_spam_flag_yes"]=int("yes" in m_flag.group(1))
    m_stat = re.search(r"^x-spam-status:\s*(\S+)", h_lower, flags=re.MULTILINE)
    if m_stat: feats["has_x_spam_status"]=1; feats["x_spam_status_yes"]=int("yes" in m_stat.group(1))
    return feats

def extract_header_features(headers: str) -> dict:
    feats = {}
    feats.update(_token_stats(headers))
    feats.update(_presence_features(headers))
    feats.update(_spf_dkim_dmarc_features(headers))
    feats.update(_received_features(headers))
    feats.update(_x_spam_features(headers))

    # From / Reply-To / Return-Path / Sender / To / Message-ID
    from_line = _extract_first_header_line(headers,"From")
    reply_line = _extract_first_header_line(headers,"Reply-To")
    return_path = _extract_first_header_line(headers,"Return-Path")
    sender_line = _extract_first_header_line(headers,"Sender")
    to_line = _extract_first_header_line(headers,"To")
    msgid_line = _extract_first_header_line(headers,"Message-ID")

    from_dom = _extract_email_domain(from_line)
    reply_dom = _extract_email_domain(reply_line)
    return_dom = _extract_email_domain(return_path)
    sender_dom = _extract_email_domain(sender_line)
    to_dom = _extract_email_domain(to_line)

    feats["from_has_domain"] = int(bool(from_dom))
    feats["reply_to_has_domain"] = int(bool(reply_dom))
    feats["return_path_has_domain"] = int(bool(return_dom))
    feats["sender_has_domain"] = int(bool(sender_dom))
    feats["to_has_domain"] = int(bool(to_dom))

    feats["from_is_freemail"] = _is_freemail(from_dom)
    feats["reply_to_is_freemail"] = _is_freemail(reply_dom)
    feats["return_path_is_freemail"] = _is_freemail(return_dom)
    feats["sender_is_freemail"] = _is_freemail(sender_dom)

    feats["from_domain_len"] = len(from_dom)
    feats["from_domain_entropy"] = math.log2(len(from_dom)+1)
    feats["return_path_domain_len"] = len(return_dom)
    feats["return_path_domain_entropy"] = math.log2(len(return_dom)+1)

    feats["from_reply_mismatch"] = int(bool(from_dom and reply_dom and from_dom!=reply_dom))
    feats["from_return_path_mismatch"] = int(bool(from_dom and return_dom and from_dom!=return_dom))
    feats["from_sender_mismatch"] = int(bool(from_dom and sender_dom and from_dom!=sender_dom))

    # TLD features
    for prefix, dom in [("from",from_dom),("reply_to",reply_dom),("return_path",return_dom),
                        ("sender",sender_dom),("to",to_dom)]:
        tld = _tld(dom)
        feats[f"{prefix}_tld_len"] = len(tld)
        feats[f"{prefix}_tld_is_numeric"] = int(tld.isdigit())
        feats[f"{prefix}_tld_is_country"] = int(len(tld)==2)

    # MIME / base64 / List headers
    h_lower = headers.lower()
    feats["has_mime_encoded_word"] = int("=?utf-8?" in h_lower or "=?iso-" in h_lower)
    base64_like = re.findall(r"[A-Za-z0-9+/]{40,}={0,2}", headers)
    feats["num_base64_like_chunks"] = len(base64_like)
    feats["has_list_headers"] = int("list-id:" in h_lower or "list-unsubscribe:" in h_lower)
    feats["has_x_originating_ip"] = int("x-originating-ip:" in h_lower)
    date_line = _extract_first_header_line(headers,"Date")
    feats["date_has_tz_offset"] = int(bool(re.findall(r"([+-]\d{4})", date_line)))

    return feats

# ---------------------------
# 4️⃣ Load Dataset (CSV with "raw_headers" column)
CSV_PATH = os.path.join(DATA_ROOT,"raw_headers_dataset.csv")
df = pd.read_csv(CSV_PATH)

df["raw_headers"] = df["raw_headers"].astype(str)

# Features extraction
feature_dicts = []
for idx, row in df.iterrows():
    feat = extract_header_features(row["raw_headers"])
    feature_dicts.append(feat)

features_df = pd.DataFrame(feature_dicts)
print("[INFO] Features extracted:", features_df.shape)

# ---------------------------
# 5️⃣ Train / Validation Split
X = features_df.values
y = df["label"].astype(int).values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=SEED, stratify=y
)

# ---------------------------
# 6️⃣ Model Definitions
xgb_model = xgb.XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=SEED,
    n_jobs=-1, use_label_encoder=False, eval_metric="logloss"
)

lgb_model = lgb.LGBMClassifier(
    n_estimators=500, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=SEED,
    n_jobs=-1
)

# ---------------------------
# 7️⃣ Model Training
xgb_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)

# Blend with simple averaging
def blend_predict(X):
    xgb_probs = xgb_model.predict_proba(X)[:,1]
    lgb_probs = lgb_model.predict_proba(X)[:,1]
    return (xgb_probs + lgb_probs)/2

# ---------------------------
# 8️⃣ Evaluation
y_pred_probs = blend_predict(X_val)
y_pred = (y_pred_probs >= 0.5).astype(int)

acc = accuracy_score(y_val, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", zero_division=0)

print("\n[INFO] Stage-4 Header Model Evaluation")
print(f"Accuracy : {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit","Phishing"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Stage-4 Header Model")
plt.show()

# Precision-Recall Curve
prec_curve, rec_curve, thresholds = precision_recall_curve(y_val, y_pred_probs)
plt.figure(figsize=(8,4))
plt.plot(rec_curve, prec_curve)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Stage-4 Header Model")
plt.grid(True)
plt.show()

# ---------------------------
# 9️⃣ Inference Helper
def predict_header_file(file_path: str):
    headers_raw = _parse_eml_file(file_path)
    if not headers_raw:
        return {"error":"Cannot read header"}
    
    feats = extract_header_features(headers_raw)
    X_single = np.array([list(feats.values())])
    prob = blend_predict(X_single)[0]
    pred = "PHISHING" if prob >= 0.5 else "LEGIT"
    return {"prediction": pred, "probability_phishing": float(prob)}

# Example usage
# predict_header_file("path/to/sample.eml")
