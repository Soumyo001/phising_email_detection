#!/usr/bin/env python
# coding: utf-8

# # STEP 1 — Imports

# In[1]:


import pandas as pd
from datasets import Dataset
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, precision_recall_curve

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

import numpy as np


# # STEP 2 — Load Dataset

# In[2]:


# Path to your big preprocessed file
data_path = "../../datasets/processed/final_combined.csv"  # adjust if needed

df = pd.read_csv(data_path)

print(df.columns)
print(df.shape)

body_df = df.copy()

# Keep only rows with non-empty body_text
body_df["body_text"] = body_df["body_text"].fillna("").astype(str)
body_df = body_df[body_df["body_text"].str.strip().astype(bool)]

# Keep only needed columns for this stage
body_df = body_df[["body_text", "label"]]

print("Body-only rows:", body_df.shape[0])
print(body_df["label"].value_counts(normalize=True))


# # STEP 3 — Train/Validation Split

# In[3]:


train_df, val_df = train_test_split(
    body_df,
    test_size=0.2,
    random_state=42,
    stratify=body_df["label"],
)

print("Train:", train_df.shape, "Val:", val_df.shape)


# # STEP 4 — Build HuggingFace Datasets

# In[4]:


train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True))


# # STEP 5 — Tokenizer (XLM-RoBERTa-large) and tokenize function

# In[5]:


model_name = "xlm-roberta-large"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# For XLM-R, pad token is usually already defined (<pad>), but just in case:
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Assigned pad_token =", tokenizer.pad_token)

def tokenize_body(batch):
    return tokenizer(
        batch["body_text"],
        truncation=True,
        max_length=512,       # start with 256; later experiment with 512
    )

train_ds_tokenized = train_ds.map(tokenize_body, batched=True, remove_columns=["body_text"])
val_ds_tokenized   = val_ds.map(tokenize_body,   batched=True, remove_columns=["body_text"])


# # STEP 6 — Data collator (dynamic padding)

# In[6]:


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# # STEP 7 — Define Metrics

# In[7]:


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(labels, preds)

    print("Confusion Matrix:")
    print(cm)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# # STEP 8 — Load XLM-RoBERTa-large with stabilizing tricks

# In[8]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", device)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
)

# Set pad token id in model config
model.config.pad_token_id = tokenizer.pad_token_id

# Gradient checkpointing for VRAM
model.gradient_checkpointing_enable()

model.to(device)


# # STEP 9 — TrainingArguments for Stage-2

# In[9]:


training_args = TrainingArguments(
    output_dir="body_model_xlmr_large_out",

    eval_strategy="epoch",          # with your HF version this works
    save_strategy="epoch",

    learning_rate=1e-5,
    lr_scheduler_type="linear",
    warmup_ratio=0.06,

    per_device_train_batch_size=4,   # safe starting point; auto_find can increase
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,   # effective batch size = 16

    num_train_epochs=3,
    weight_decay=0.01,

    fp16=False,                     # mixed precision
    bf16=True,
    max_grad_norm=1.0,
    # auto_find_batch_size=True,    # let HF try to grow until OOM

    gradient_checkpointing=True,    # ties with model.gradient_checkpointing_enable()

    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",

    optim="adamw_torch",
    label_smoothing_factor=0.1,
    report_to="none",
)


# # STEP 10 — Trainer & Training

# In[ ]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds_tokenized,
    eval_dataset=val_ds_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

print("===== BODY MODEL VALIDATION METRICS =====")
metrics = trainer.evaluate()
print(metrics)


df = pd.DataFrame(trainer.state.log_history)
eval_df = df[df["eval_loss"].notna()].copy()

if "epoch" not in eval_df.columns:
    eval_df["epoch"] = range(1, len(eval_df) + 1)

eval_df["epoch"] = eval_df["epoch"].astype(int)
print("Eval columns:", list(eval_df.columns))

df.head()


# # STEP 11 — Plot Train Loss Curve

# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(df[df["loss"].notna()]["step"], df[df["loss"].notna()]["loss"], label="Train Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.show()


# # STEP 12 — Plot Evaluation Metrics Per Epoch

# In[ ]:


plt.figure(figsize=(14,6))

if "eval_accuracy" in eval_df:
    plt.plot(eval_df["epoch"], eval_df["eval_accuracy"], marker="o", label="Accuracy")

if "eval_precision" in eval_df:
    plt.plot(eval_df["epoch"], eval_df["eval_precision"], marker="o", label="Precision")

if "eval_recall" in eval_df:
    plt.plot(eval_df["epoch"], eval_df["eval_recall"], marker="o", label="Recall")

if "eval_f1" in eval_df:
    plt.plot(eval_df["epoch"], eval_df["eval_f1"], marker="o", label="F1 Score")

plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.title("Evaluation Metrics per Epoch")
plt.legend()
plt.grid(True)
plt.show()


# # STEP 13 — Confusion Matrix for body model

# In[ ]:


predictions = trainer.predict(val_ds)
y_true = predictions.label_ids
y_pred = predictions.predictions.argmax(-1)

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Phishing"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix for XLM-R Large Body Model")
plt.show()


# # STEP 14 — Classification Report

# In[ ]:


print(classification_report(y_true, y_pred, target_names=["Legit", "Phishing"]))


# # STEP 15 — Precision–Recall Curve (For imbalanced phishing data)

# In[ ]:


predictions = trainer.predict(val_ds)
y_true = predictions.label_ids
y_probs = torch.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1]  # phishing probability
y_probs = y_probs.numpy()

prec, rec, thresh = precision_recall_curve(y_true, y_probs)

plt.figure(figsize=(10,5))
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.show()


# # STEP 16 — Save Model + Tokenizer

# In[ ]:


trainer.save_model("body_encoder_xlmr_large")
tokenizer.save_pretrained("body_encoder_xlmr_large_tokenizer")

print("Training complete and model saved.")


# # STEP 17 — Test model with user input

# In[ ]:


def predict_body(text: str) -> str:
    model.eval()
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    with torch.no_grad():
        output = model(**encoded)
        pred = torch.argmax(output.logits, dim=1).item()

    return "🔴 PHISHING (body)" if pred == 1 else "🟢 LEGITIMATE (body)"

# Example:
print(predict_body("Dear customer, your PayPal account has been limited. Please click the link to restore access."))

