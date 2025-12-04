#!/usr/bin/env python3
import os
import random
import shutil

ENRON_ROOT = "headers/enron"         
OUT_DIR    = "headers/enron_down_50k"
TARGET     = 50000

def is_email_file(fname):
    return not fname.lower().endswith((
        ".txt", ".csv", ".json", ".zip", ".tar", ".db", ".py", ".html", ".htm"
    ))

print("Scanning Enron maildir...")

all_files = []
for dirpath, _, files in os.walk(ENRON_ROOT):
    for fname in files:
        if is_email_file(fname):
            all_files.append(os.path.join(dirpath, fname))

print(f"Found {len(all_files)} Enron HAM emails.")

random.shuffle(all_files)
sampled = all_files[:TARGET]

print(f"Sampling {len(sampled)} emails -> {OUT_DIR}")

for fpath in sampled:
    rel = os.path.relpath(fpath, ENRON_ROOT)
    outpath = os.path.join(OUT_DIR, rel)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    shutil.copy2(fpath, outpath)

print("DONE — Enron downsample prepared.")
