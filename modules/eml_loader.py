import utils.helpers.preprocess_helper as helper
import pandas as pd
import os, uuid, re
from email import policy
from email.parser import BytesParser
from charset_normalizer import from_bytes
from data.constants import UNIFIED_COLUMNS, TEXT_EXTS, TEXT_ATTACHMENT_TYPES
from modules.attachment_extractor import extract_attachment_text_from_bytes

class EmlLoader:
    def __init__(self, config: dict = {}):
            self.config = config

    def _decode_header_safely(self, raw_bytes: bytes, path: str) -> str:
        try:
            return raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            print(f"[WARN] UTF-8 decoding did not work for {path}. Trying to guess..")
            pass

        try:
            best = from_bytes(raw_bytes).best()
            if best and best.encoding:
                try:
                    print(f"[INFO] Found encoding for {path}: {best.encoding}")
                    return raw_bytes.decode(best.encoding)
                except Exception:
                    pass
        except Exception:
            pass

        print(f"[WARN] Header decode issue in {path}: some characters replaced")
        return raw_bytes.decode("utf-8", errors="replace")

    def separate_header_blocks(self, path: str):
        # raw header extraction
        try:
            with open(path, "rb") as f:
                data = f.read()
        except Exception as e:
            print(f"[WARN] Cannot read {path}: {e}")
            return None

        # Try CRLFCRLF first
        sep = data.find(b"\r\n\r\n")
        sep_len = 4 

        # Try LFLF if not found
        if sep == -1:
            sep = data.find(b"\n\n")
            sep_len = 2 

        # No separator, whole file is considered headers
        if sep == -1:
            raw_header_block = data
            raw_body_block = b""
        else:
            raw_header_block = data[:sep]
            raw_body_block = data[sep + sep_len:]

        # Store raw headers
        headers_raw = self._decode_header_safely(raw_header_block, path)
        return raw_header_block, raw_body_block, headers_raw


    # EML parsing
    def _parse_eml_file(self, path: str):
        """High-quality EML parsing with RAW header extraction (safe for malformed phishing headers)."""

        raw_header_block, raw_body_block, headers_raw = self.separate_header_blocks(path)


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

        # Use Python parser ONLY to walk MIME tree (we don't trust its header parsing)
        try:
            msg = BytesParser(policy=policy.default).parse(open(path, "rb"))
        except Exception as e:
            print(f"[WARN] Body parse failed for {path}: {e}")
            msg = None

        body_text_parts = []
        attachment_text_parts = []

        if msg and msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                disp = str(part.get("Content-Disposition", "") or "").lower()
                filename = part.get_filename()

                has_attachment_flag = (
                    ("attachment" in disp)
                    or ("filename=" in disp)
                    or (filename and disp.startswith("inline"))
                )
                is_text_like = (
                    (filename and os.path.splitext(filename)[1].lower() in TEXT_EXTS)
                    or ctype in TEXT_ATTACHMENT_TYPES
                    or ctype.startswith("text/")
                )

                if has_attachment_flag and is_text_like:
                    try:
                        payload = part.get_payload(decode=True)
                    except:
                        payload = None
                    
                    if payload:
                        txt = extract_attachment_text_from_bytes(
                            content_type=ctype,
                            filename=filename,
                            data=payload,
                            charset=part.get_content_charset()
                        )
                        if txt:
                            attachment_text_parts.append(txt)
                    continue

                # Inline parts (body)
                try:
                    payload = part.get_payload(decode=True) or part.get_payload()
                    text = helper.clean_payload(payload, part.get_content_charset())
                except:
                    continue

                if ctype == "text/plain":
                    body_text_parts.append(text)

                elif ctype == "text/html":
                    text = re.sub(r"<script.*?>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
                    body_text_parts.append(helper.clean_html_simple(text))

        elif msg:
            # Single-part message: decide if it's body or attachment
            try:
                payload = msg.get_payload(decode=True)
            except Exception:
                payload = None

            try:
                if payload:
                    text = helper.clean_payload(payload, msg.get_content_charset())
                    ctype = msg.get_content_type()
                    disp = str(msg.get("Content-Disposition", "") or "").lower()
                    filename = msg.get_filename()

                    has_attachment_flag = (
                        ("attachment" in disp)
                        or (filename and disp.startswith("inline"))
                        or ("filename=" in disp)
                    )
                    is_text_like = (
                        (filename and os.path.splitext(filename)[1].lower() in TEXT_EXTS)
                        or ctype in TEXT_ATTACHMENT_TYPES
                        or ctype.startswith("text/")
                    )

                    if has_attachment_flag and is_text_like:
                        txt = extract_attachment_text_from_bytes(
                            content_type=ctype,
                            filename=filename,
                            data=payload,
                            charset=msg.get_content_charset()
                        )
                        if txt:
                            attachment_text_parts.append(txt)
                    else:
                        if ctype == "text/html":
                            text = re.sub(r"<script.*?>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
                            body_text_parts.append(helper.clean_html_simple(text))
                        else:
                            body_text_parts.append(text)
                else:
                    body_text_parts.append(helper.clean_payload(raw_body_block, "utf-8"))
            except:
                body_text_parts.append(helper.clean_payload(raw_body_block, "utf-8"))

        else:
            # No msg: fallback, dump raw body as text
            body_text_parts.append(helper.clean_payload(raw_body_block, "utf-8"))

        body_text = " ".join([p for p in body_text_parts if p]).strip()
        attachment_text = " ".join([p for p in attachment_text_parts if p]).strip()

        # URL extraction
        url_list = []
        url_list.extend(helper.extract_urls(subject))
        url_list.extend(helper.extract_urls(body_text))
        url_list.extend(helper.extract_urls(attachment_text))

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
                if fname.startswith('.') or fname.endswith('.tar') or fname.endswith('.gz'):
                    continue
                fpath = os.path.join(dirpath, fname)
                # if not helper.is_email_file(fpath):
                #     continue
                
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