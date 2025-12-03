#!/usr/bin/env python

import os
import requests
from data.constants import MALICIOUS_URL_LINKS, LEGIT_URL_LINKS

OUT_DIR  = os.path.join("datasets", "url_feeds")
os.makedirs(OUT_DIR, exist_ok=True)

URL_HAUS_FULL_CSV = "https://urlhaus.abuse.ch/downloads/csv/"
URL_HAUS_RECENT_CSV = "https://urlhaus.abuse.ch/downloads/csv_recent/"
URL_HAUS_ONLINE_CSV = "https://urlhaus.abuse.ch/downloads/csv_online/"

def download_file(url, out_path):
    print(f"[INFO] Downloading: {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)
    print(f"[INFO] Saved to: {out_path}")

def download_urlhaus():
    full_path_recent = os.path.join(OUT_DIR, "urlhaus_recent.csv")
    full_path_full = os.path.join(OUT_DIR, "urlhaus_full.csv.zip")
    full_path_online = os.path.join(OUT_DIR, "urlhaus_online.csv")
    if not os.path.exists(full_path_recent):
        download_file(URL_HAUS_RECENT_CSV, full_path_recent)
    else:
        print(f"[SKIP] {full_path_recent} already exists")

    if not os.path.exists(full_path_online):
        download_file(URL_HAUS_ONLINE_CSV, full_path_online)
    else:
        print(f"[SKIP] {full_path_online} already exists")

    if not os.path.exists(full_path_full):
        download_file(URL_HAUS_FULL_CSV, full_path_full)
        print(f"[INFO] Please manually unzip the file at {full_path_full}")
    else:
        print(f"[SKIP] {full_path_full} already exists")

def download_legit_csv():
    for db_name, url in LEGIT_URL_LINKS.items():
        full_path = os.path.join(OUT_DIR, db_name)
        if not os.path.exists(full_path):
            download_file(url, full_path)
            if "cisco" in full_path and full_path.endswith(".zip"):
                print(f"[INFO] Please manually unzip the file at {full_path}")
        else:
            print(f"[SKIP] {full_path} already exists")

def download_malicious_urls():
    for db_name, url in MALICIOUS_URL_LINKS.items():
        full_path = os.path.join(OUT_DIR, db_name)
        if not os.path.exists(full_path):
            download_file(url, full_path)
            if db_name == "url_haus.txt":
                start_index = -1
                with open(full_path, "r") as f:
                    lines = f.readlines()
                for i, line in enumerate(lines):
                    if line.strip() == "# url":
                        print("[INFO] found comment end in urlhause database. removing till that...")
                        start_index = i + 1
                        break
                if start_index == -1:
                    print(f"[WARN] Could not find comment end. please manually remove the comments from {full_path}")

                else:
                    url_lines = lines[start_index:]
                    with open(full_path, "w") as f:
                        f.writelines(url_lines)

            elif "hagezi" in db_name:
                start_index = -1
                with open(full_path, "r") as f:
                    lines = f.readlines()
                for i, line in enumerate(lines):
                    if line.strip().startswith("#"):
                        continue
                    start_index = i
                    break
                if start_index == -1:
                    print(f"[WARN] Could not find comment end. please manually remove the comments from {full_path}")
                else:
                    domain_lines = lines[start_index:]
                    with open(full_path, "w") as f:
                        f.writelines(domain_lines)
                    
        else:
            print(f"[SKIP] {full_path} already exists")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    download_urlhaus()
    download_legit_csv()
    download_malicious_urls()
    print("[DONE] Stage-3 Step-1 complete.")
