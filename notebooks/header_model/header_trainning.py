import sys, os
ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import random
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, classification_report
)
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from modules.eml_loader import EmlLoader
from utils.helpers.header_cleaner import clean_header

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", device)

CSV_PATH = "../../datasets/processed/final_combined.csv"

df = pd.read_csv(CSV_PATH)

headers_df = df[[
    "headers_raw","canonical_raw","label",
    "header_from_reply_mismatch",
    "header_domain_mismatch",
    "header_suspicious_tld",
    "header_received_count",
    "header_to_anomaly",
    "header_x_mailer_anomaly",
    "header_date_malformed"
]].copy()

headers_df["headers_raw"] = headers_df["headers_raw"].fillna("").astype(str)
headers_df["canonical_raw"] = headers_df["canonical_raw"].fillna("").astype(str)

headers_df["headers_raw_clean"] = headers_df["headers_raw"].apply(clean_header)

# Remove empty/missing headers
headers_df["headers_raw_clean"] = headers_df["headers_raw_clean"].fillna("").astype(str)
headers_df = headers_df[headers_df["headers_raw_clean"].str.strip().astype(bool)]


def build_header_input(row):
    anomaly_text = (
        f"from_reply_mismatch: {row['header_from_reply_mismatch']} "
        f"domain_mismatch: {row['header_domain_mismatch']} "
        f"suspicious_tld: {row['header_suspicious_tld']} "
        f"received_count: {row['header_received_count']} "
        f"to_anomaly: {row['header_to_anomaly']} "
        f"x_mailer_anomaly: {row['header_x_mailer_anomaly']} "
        f"date_malformed: {row['header_date_malformed']}"
    )

    return (
        "[RAW]\n" + row["headers_raw_clean"] +
        "\n\n[CANONICAL]\n" + row["canonical_raw"] +
        "\n\n[ANOMALIES]\n" + anomaly_text
    )

headers_df["header_input"] = headers_df.apply(build_header_input, axis=1).astype(str)


# Drop duplicates
headers_df = headers_df.drop_duplicates(subset=["header_input"])

print("Header-only rows:", headers_df.shape[0])
print(headers_df["label"].value_counts(normalize=True))

train_df, val_df = train_test_split(
    headers_df,
    test_size=0.1,
    random_state=SEED,
    stratify=headers_df["label"]
)

print("Train size:", len(train_df), "Val size:", len(val_df))
print("Train label counts:", train_df["label"].value_counts())
print("Val label counts:", val_df["label"].value_counts())
print(headers_df["header_input"].head())

MODEL_NAME = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Assigned pad_token =", tokenizer.pad_token)

MAX_LEN = 512

def tokenize_headers(batch):
    out = tokenizer(
        batch["header_input"],
        truncation=True,
        max_length=MAX_LEN
    )
    numeric_cols = [
        "header_from_reply_mismatch",
        "header_domain_mismatch",
        "header_suspicious_tld",
        "header_received_count",
        "header_to_anomaly",
        "header_x_mailer_anomaly",
        "header_date_malformed"
    ]
    for col in numeric_cols:
        out[col] = batch[col]
    return out

train_ds = Dataset.from_pandas(train_df.rename(columns={"label":"labels"}))
val_ds   = Dataset.from_pandas(val_df.rename(columns={"label":"labels"}))

# remove ONLY the columns we are no longer using
train_ds_tokenized = train_ds.map(tokenize_headers, batched=True, remove_columns=["headers_raw","headers_raw_clean","canonical_raw","header_input"])
val_ds_tokenized   = val_ds.map(tokenize_headers, batched=True, remove_columns=["headers_raw","headers_raw_clean","canonical_raw","header_input"])

torch_cols = ["input_ids", "attention_mask", "labels"] + [
    "header_from_reply_mismatch",
    "header_domain_mismatch",
    "header_suspicious_tld",
    "header_received_count",
    "header_to_anomaly",
    "header_x_mailer_anomaly",
    "header_date_malformed"
]

train_ds_tokenized.set_format("torch", columns=torch_cols)
val_ds_tokenized.set_format("torch", columns=torch_cols)

data_collator = DataCollatorWithPadding(tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.config.pad_token_id = tokenizer.pad_token_id
# model.gradient_checkpointing_enable()  # Gradient checkpointing
model.to(device)

print("Model loaded:", MODEL_NAME)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, preds))
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

training_args = TrainingArguments(
    output_dir="header_deberta_large_out",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    lr_scheduler_type="linear",
    warmup_ratio=0.06,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # effective batch size = 16
    num_train_epochs=3,
    weight_decay=0.01,
    bf16=False,
    fp16=True,
    max_grad_norm=1.0,
    gradient_checkpointing=False,
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    optim="adamw_torch",
    label_smoothing_factor=0.1,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds_tokenized,
    eval_dataset=val_ds_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model("header_deberta_large")
tokenizer.save_pretrained("header_deberta_large_tokenizer")
print("Training complete and model saved.")

preds = trainer.predict(val_ds_tokenized)
y_true = preds.label_ids
y_pred = preds.predictions.argmax(-1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit","Phishing"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Headers DeBERTa v3-Large")
plt.show()

df_logs = pd.DataFrame(trainer.state.log_history)
loss_df = df_logs[df_logs["loss"].notna()]
plt.figure(figsize=(10,5))
plt.plot(loss_df["step"], loss_df["loss"], marker="o")
plt.title("Training Loss Curve")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Metrics per epoch
eval_df = df_logs[df_logs["eval_loss"].notna()].copy()
if "epoch" not in eval_df.columns:
    eval_df["epoch"] = range(1, len(eval_df)+1)

plt.figure(figsize=(14,6))
for col, label in [
    ("eval_accuracy", "Accuracy"),
    ("eval_precision", "Precision"),
    ("eval_recall", "Recall"),
    ("eval_f1", "F1 Score")
]:
    if col in eval_df:
        plt.plot(eval_df["epoch"], eval_df[col], marker="o", label=label)

plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.title("Evaluation Metrics per Epoch")
plt.legend()
plt.grid(True)
plt.show()

probs = torch.softmax(torch.tensor(preds.predictions), dim=1)[:, 1].numpy()
prec, rec, thresh = precision_recall_curve(y_true, probs)

plt.figure(figsize=(10,5))
plt.plot(rec, prec)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.show()

final_preds = preds.predictions.argmax(-1)
print(classification_report(y_true, final_preds, target_names=["Legit","Phishing"]))

from utils.helpers.preprocess_helper import extract_header_anomalies
def predict_header_eml(eml_path: str):
    loader = EmlLoader()

    # Extract raw header block from EML
    try:
        _, _, headers_raw, canonical_raw = loader.separate_header_blocks(eml_path)
    except:
        return {"error": "Unable to read EML file."}

    if not isinstance(headers_raw, str):
        headers_raw = "" if headers_raw is None else str(headers_raw)

    if not isinstance(canonical_raw, str):
        canonical_raw = "" if canonical_raw is None else str(canonical_raw)

    headers_raw_clean = clean_header(headers_raw)

    parsed = {
        "subject": "",
        "from_email": "",
        "to_email": "",
        "reply_to_email": "",
        "date": "",
        "headers_raw": headers_raw_clean
    }
    anomaly_dict = extract_header_anomalies(parsed)

    # ensure numeric fields exist
    numeric_vals = [
        anomaly_dict["header_from_reply_mismatch"],
        anomaly_dict["header_domain_mismatch"],
        anomaly_dict["header_suspicious_tld"],
        anomaly_dict["header_received_count"],
        anomaly_dict["header_to_anomaly"],
        anomaly_dict["header_x_mailer_anomaly"],
        anomaly_dict["header_date_malformed"],
    ]

    anomaly_text = (
        f"from_reply_mismatch: {numeric_vals[0]} "
        f"domain_mismatch: {numeric_vals[1]} "
        f"suspicious_tld: {numeric_vals[2]} "
        f"received_count: {numeric_vals[3]} "
        f"to_anomaly: {numeric_vals[4]} "
        f"x_mailer_anomaly: {numeric_vals[5]} "
        f"date_malformed: {numeric_vals[6]}"
    )

    header_input = (
        "[RAW]\n" + headers_raw_clean +
        "\n\n[CANONICAL]\n" + canonical_raw +
        "\n\n[ANOMALIES]\n" + anomaly_text
    )

    inputs = tokenizer(
        [header_input],
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred   = int(np.argmax(probs))

    return {
        "prediction": "PHISHING" if pred == 1 else "LEGIT",
        "prob_legit": float(probs[0]),
        "prob_phishing": float(probs[1]),
        "clean_header": headers_raw_clean[:2000],   # optional preview
        "canonical_header": canonical_raw[:2000],   # optional preview
        "anomalies": anomaly_dict
    }

# --- Quick test ---
FILE = "4s.eml"
for key, value in predict_header_eml(FILE).items():
    print(f"{key}: {value}")

