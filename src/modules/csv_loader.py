import src.utils.helpers.preprocess_helper as helper
import pandas as pd
import os, uuid
from src.data.constants import UNIFIED_COLUMNS

class CSVLoader:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    # CSV processing
    def _unify_row_from_csv(self, row):
        subject = str(row[self.cfg["subject_col"]]) if self.cfg["subject_col"] in row and pd.notna(row[self.cfg["subject_col"]]) else ""
        body_text = str(row[self.cfg["body_col"]]) if self.cfg["body_col"] in row and pd.notna(row[self.cfg["body_col"]]) else ""

        from_email = helper.extract_email_only(row[self.cfg["sender_col"]]) if self.cfg.get("sender_col") in row else ""
        to_email = helper.extract_email_only(row[self.cfg["receiver_col"]]) if self.cfg.get("receiver_col") in row else ""

        raw_label = row[self.cfg["label_col"]] if self.cfg["label_col"] in row else None
        label = helper.normalize_label(raw_label, self.cfg.get("pos_labels"), self.cfg.get("neg_labels"))
        if label is None:
            return None

        urls = []
        urls.extend(helper.extract_urls(subject))
        urls.extend(helper.extract_urls(body_text))
        seen = set()
        uniq = [u for u in urls if not (u in seen or seen.add(u))]
        urls_str = " ".join(uniq)

        date = str(row[self.cfg["date_col"]]).strip() if self.cfg.get("date_col") in row else ""

        base = {
            "id": str(uuid.uuid4()),
            "subject": subject.strip(),
            "body_text": body_text.strip(),
            "attachment_text": "",
            "headers_raw": "",          # no raw headers in CSV
            "from_email": from_email,
            "to_email": to_email,
            "reply_to_email": "",       # CSV usually won't have this
            "date": date,
            "urls": urls_str,
            "label": label,
            "source": self.cfg["name"],
        }

        anoms = helper.extract_header_anomalies(base)
        base.update(anoms)

        return base

    def load_csv_dataset(self) -> pd.DataFrame:
        path = self.cfg["path"]
        if not os.path.isfile(path):
            print(f"[WARN] CSV not found: {path}")
            return pd.DataFrame(columns=UNIFIED_COLUMNS)

        print(f"[INFO] Loading CSV dataset: {self.cfg['name']} from {path}")
    
        rows = []

        try:
            # Try utf-8 with chunks
            reader = pd.read_csv(
                path,
                chunksize=5000,      # process 5000 rows at a time
                dtype=str,
                low_memory=False
            )
        except UnicodeDecodeError:
            reader = pd.read_csv(
                path,
                chunksize=5000,
                dtype=str,
                low_memory=False,
                encoding="latin-1"
            )

        # Process chunk by chunk
        total_rows = 0
        for df_raw in reader:
            for _, row in df_raw.iterrows():
                x = self._unify_row_from_csv(row, self.cfg)
                if x is not None:
                    if x["subject"] or x["body_text"]:
                        rows.append(x)
            total_rows += len(df_raw)

        if not rows:
            print(f"[WARN] All rows dropped for {self.cfg['name']}")
            return pd.DataFrame(columns=UNIFIED_COLUMNS)

        df = pd.DataFrame(rows)
        df = df[UNIFIED_COLUMNS]
        print(f"[INFO] Loaded {len(df)} rows from {self.cfg['name']} (scanned {total_rows} raw rows)")
        return df