import utils.helpers.preprocess_helper as helper
import pandas as pd
import os, uuid
from email import policy
from email.parser import BytesParser
from data.constants import UNIFIED_COLUMNS

class EmlLoader:
    def __init__(self, config: dict):
        self.config = config

    # EML parsing
    def _parse_eml_file(self, path: str):
        """High-quality EML parsing with RAW header extraction (safe for malformed phishing headers)."""

        # raw header extraction
        try:
            with open(path, "rb") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"[WARN] Cannot read {path}: {e}")
            return None

        raw_header_block = b""
        raw_body_block = b""
        sep_index = 0

        # Split headers and body manually (no Python email parsing!)
        for i, line in enumerate(lines):
            raw_header_block += line
            if line.strip() == b"":     # blank line → end of headers
                sep_index = i + 1
                break

        raw_body_block = b"".join(lines[sep_index:])

        raw_subject    = helper.extract_raw_header_field(raw_header_block, "Subject")
        raw_from       = helper.extract_raw_header_field(raw_header_block, "From")
        raw_to         = helper.extract_raw_header_field(raw_header_block, "To")
        raw_reply_to   = helper.extract_raw_header_field(raw_header_block, "Reply-To")
        raw_date       = helper.extract_raw_header_field(raw_header_block, "Date")

        subject        = raw_subject
        from_email     = helper.extract_email_only(raw_from)
        to_email       = helper.extract_email_only(raw_to)
        reply_to_email = helper.extract_email_only(raw_reply_to)
        date           = raw_date

        # Store raw headers
        try:
            headers_raw = raw_header_block.decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"[WARN] header decode failed for {path}: {e}")
            headers_raw = ""

        # Use Python parser ONLY for the body (not the headers)
        try:
            msg = BytesParser(policy=policy.default).parsebytes(raw_body_block)
        except Exception as e:
            print(f"[WARN] Body parse failed for {path}: {e}")
            msg = None

        body_text_parts = []
        attachment_text_parts = []

        if msg and msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                disp = str(part.get("Content-Disposition", "") or "").lower()

                # Attachments
                if "attachment" in disp or part.get_filename():
                    try:
                        payload = part.get_content()
                        if isinstance(payload, bytes):
                            payload = payload.decode(part.get_content_charset() or "utf-8", errors="ignore")
                        if isinstance(payload, str):
                            attachment_text_parts.append(helper.clean_html_simple(payload))
                    except:
                        pass
                    continue

                # Inline parts
                try:
                    payload = part.get_content()
                except:
                    payload = None

                if payload is None:
                    continue

                if isinstance(payload, bytes):
                    payload = payload.decode(part.get_content_charset() or "utf-8", errors="ignore")

                if isinstance(payload, str):
                    if ctype == "text/plain":
                        body_text_parts.append(payload)
                    elif ctype == "text/html":
                        body_text_parts.append(helper.clean_html_simple(payload))

        elif msg:
            try:
                payload = msg.get_content()
            except:
                payload = ""

            if isinstance(payload, bytes):
                payload = payload.decode(msg.get_content_charset() or "utf-8", errors="ignore")

            if isinstance(payload, str):
                if msg.get_content_type() == "text/html":
                    body_text_parts.append(helper.clean_html_simple(payload))
                else:
                    body_text_parts.append(payload)

        body_text = " ".join([p for p in body_text_parts if p]).strip()
        attachment_text = " ".join([p for p in attachment_text_parts if p]).strip()

        # URL extraction
        url_list = []
        url_list.extend(helper.extract_urls(subject))
        url_list.extend(helper.extract_urls(body_text))

        # Deduplicate URLs
        seen = set()
        uniq_urls = []
        for u in url_list:
            if u not in seen:
                uniq_urls.append(u)
                seen.add(u)

        return {
            "id": str(uuid.uuid4()),
            "subject": subject,
            "body_text": body_text,
            "attachment_text": attachment_text,
            "headers_raw": headers_raw,
            "from_email": from_email,
            "to_email": to_email,
            "reply_to_email": reply_to_email,
            "date": date,
            "urls": " ".join(uniq_urls),
        }

    def load_eml_dir(self) -> pd.DataFrame:
        root = self.config["root_dir"]
        name = self.config["name"]
        label_for_all = self.config.get("label_for_all", None)

        rows = []

        print(f"[INFO] Loading EML dataset: {name} from {root}")

        for dirpath, _, files in os.walk(root):
            for fname in files:
                if not fname.lower().endswith(".eml"):
                    continue

                fpath = os.path.join(dirpath, fname)
                data = self._parse_eml_file(fpath)
                if data is None:
                    continue
                
                # Compute header anomaly features
                anoms = helper.extract_header_anomalies(data)

                row = {
                    **data,    
                    **anoms,         
                    "label": label_for_all,
                    "source": name,
                }
                rows.append(row)

        if not rows:
            print(f"[WARN] No EML rows found in {root}")
            return pd.DataFrame(columns=UNIFIED_COLUMNS)

        df = pd.DataFrame(rows)
        for col in UNIFIED_COLUMNS:
            if col not in df.columns:
                df[col] = None

        df = df[UNIFIED_COLUMNS]
        print(f"[INFO] Loaded {len(df)} EML rows from {root}")
        return df