#!/usr/bin/env python3
"""
preprocess.py (print-only version)

Build a unified phishing-email dataset from:
- CSV datasets (subject/body/label/sender/receiver)
- .eml datasets (headers/body/attachments)

Output: data/processed/final_master.csv
"""

import os
from data.configs import get_csv_datasets, get_eml_datasets
from modules.csv_loader import CSVLoader
from modules.eml_loader import EmlLoader
from data.constants import OUT_DIR
import utils.helpers.preprocess_helper as helper

# Main
def main():
    all_dfs = []

    # Load CSV datasets
    for cfg in get_csv_datasets(seed=42):
        loader = CSVLoader(cfg)
        all_dfs.append(loader.load_csv_dataset())

    # Load EML datasets
    for cfg in get_eml_datasets(seed=42):
        loader = EmlLoader(cfg)
        all_dfs.append(loader.load_eml_dir())

    # Merge
    master = helper.clean_and_merge(all_dfs)

    
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "final_combined_2.csv")
    master.to_csv(out_path, index=False, encoding="utf-8", quoting=1, escapechar="\\")

    print(f"[DONE] Saved unified dataset with {len(master)} rows → {out_path}")


if __name__ == "__main__":
    main()