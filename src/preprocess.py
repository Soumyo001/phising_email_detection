#!/usr/bin/env python3
"""
preprocess.py (print-only version)

Build a unified phishing-email dataset from:
- CSV datasets (subject/body/label/sender/receiver)
- .eml datasets (headers/body/attachments)

Output: data/processed/final_master.csv
"""

import os
from src.data.configs import CSV_DATASETS, EML_DATASETS
from src.modules.csv_loader import CSVLoader
from src.modules.eml_loader import EmlLoader
from src.data.constants import OUT_DIR
import src.utils.helpers.preprocess_helper as helper

# Main
def main():
    all_dfs = []

    # Load CSV datasets
    for cfg in CSV_DATASETS:
        loader = CSVLoader(cfg)
        all_dfs.append(loader.load_csv_dataset())

    # Load EML datasets
    for cfg in EML_DATASETS:
        loader = EmlLoader(cfg)
        all_dfs.append(loader.load_eml_dir())

    # Merge
    master = helper.clean_and_merge(all_dfs)

    
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "final_combined.csv")
    master.to_csv(out_path, index=False, encoding="utf-8")

    print(f"[DONE] Saved unified dataset with {len(master)} rows → {out_path}")


if __name__ == "__main__":
    main()