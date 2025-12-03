import os
import io
import re
import mimetypes
import pytesseract
import unicodedata
from data.constants import TEXT_EXTS

# Optional deps – we degrade gracefully if they are missing
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception:
    pdf_extract_text = None
from pdf2image import convert_from_bytes
try:
    import docx
except Exception:
    docx = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None


# ---------- small helpers ----------

def _safe_decode_text(data, charset: str | None = None) -> str:
    """Decode bytes -> str with a few fallback encodings."""
    if isinstance(data, str):
        return data
    if not isinstance(data, (bytes, bytearray)):
        return ""

    tried = []
    if charset:
        tried.append(charset)
    tried.extend(["utf-8", "latin-1"])

    for enc in tried:
        try:
            return data.decode(enc, errors="ignore")
        except Exception:
            continue
    return ""


def _normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split())

def _normalize_text(s: str) -> str:
    """Normalize text to NFKC and remove weird characters."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\x00", "")
    # Collapse huge whitespace
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def pdf_ocr_extract(pdf_bytes: bytes) -> str:
    """Extract text by OCR when pdfminer fails."""
    try:
        images = convert_from_bytes(pdf_bytes)
    except Exception as e:
        print(f"[WARN] OCR convert_from_bytes failed: {e}")
        return ""

    text_parts = []
    for idx, img in enumerate(images):
        try:
            txt = pytesseract.image_to_string(img)
            text_parts.append(txt)
        except Exception as e:
            print(f"[WARN] OCR failed on page {idx}: {e}")

    combined = "\n".join(text_parts)
    return _normalize_text(combined)


# ---------- core extractor from BYTES ----------

def extract_attachment_text_from_bytes(
    content_type: str | None,
    filename: str | None,
    data: bytes | bytearray | None,
    charset: str | None = None,
) -> str:
    """
    Turn attachment bytes into CLEAN TEXT, if reasonably possible.
    Supports:
      - text/* (plain, html, xml, json, csv, code, logs...)
      - application/pdf
      - application/msword, application/vnd.openxmlformats-officedocument.wordprocessingml.document
      - application/vnd.ms-excel, application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
      - plus simple fallbacks
    Returns "" if we can't reliably extract text.
    """
    if not data:
        return ""

    ct = (content_type or "").lower()
    ext = (os.path.splitext(filename)[1].lower() if filename else "")

    # ---------- HTML ----------
    if ct in {"text/html", "application/xhtml+xml"} or ext in {".html", ".htm"}:
        raw = _safe_decode_text(data, charset)
        if not raw:
            return ""

        # Prefer BeautifulSoup when available
        if BeautifulSoup is not None:
            try:
                soup = BeautifulSoup(raw, "lxml")
                text = soup.get_text(" ", strip=True)
                return _normalize_whitespace(text)
            except Exception:
                pass

        # Fallback: regex-based tag strip
        text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
        text = re.sub(r"(?s)<.*?>", " ", text)
        return _normalize_whitespace(text)

    # ---------- Plain text / code / config / CSV / JSON / XML etc. ----------
    text_like_exts = {
        # Basic text
        ".txt", ".log", ".md", ".rst",
        ".json", ".yaml", ".yml", ".xml",
    
        # Config/infra
        ".ini", ".cfg", ".conf", ".toml",
    
        # Data files
        ".csv", ".tsv",
    
        # LaTeX / markup
        ".tex", ".html", ".htm", ".xhtml",
    
        # Source code
        ".py", ".js", ".ts", ".c", ".cpp", ".h", ".hpp",
        ".java", ".cs", ".rb", ".php", ".pl", ".sh", ".bash", ".zsh",
        ".go", ".swift", ".kt", ".rs",
    
        # Script formats
        ".bat", ".cmd", ".ps1",
    }
    if ct.startswith("text/") or ext in TEXT_EXTS:
        raw = _safe_decode_text(data, charset)
        if not raw:
            return ""
        return _normalize_whitespace(raw)

    # ---------- PDF ----------
    if ct == "application/pdf" or ext == ".pdf":
        if pdf_extract_text is None:
            return ""

        try:
            bio = io.BytesIO(data)
            text = pdf_extract_text(bio)
            text = _normalize_text(text)
        except Exception as e:
            print(f"[WARN] pdfminer failed: {e}")
            text = ""

        # ---- OCR fallback ----
        if len(text.strip()) < 30:
            print("[INFO] pdfminer output too small — switching to OCR fallback")
            ocr_text = pdf_ocr_extract(data)   # <-- IMPORTANT!
            if len(ocr_text.strip()) > 0:
                return ocr_text

        return text

    # ---------- DOC / DOCX ----------
    if (
        ct in {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        }
        or ext in {".docx", ".doc"}
    ):
        if docx is None:
            return ""
        try:
            bio = io.BytesIO(data)
            document = docx.Document(bio)
            text = "\n".join(p.text for p in document.paragraphs)
            return _normalize_whitespace(text)
        except Exception:
            return ""

    # ---------- XLS / XLSX ----------
    if (
        ct in {
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }
        or ext in {".xls", ".xlsx"}
    ):
        # Simple fallback: decode as text; if it's garbage it'll be filtered later by length
        raw = _safe_decode_text(data, charset)
        return _normalize_whitespace(raw)

    # ---------- Generic fallback: try decode as text ----------
    raw = _safe_decode_text(data, charset)
    if not raw:
        return ""

    # Skip binary
    if raw.count("\x00") > 3:
        return ""

    return _normalize_text(raw)


# ---------- convenience for local FILES (used in Stage-2B notebook) ----------

def extract_attachment_text(file_path: str) -> str:
    """
    File-path based helper (for your notebook inference).
    Uses mimetypes + extension to guess content_type,
    then reuses extract_attachment_text_from_bytes().
    """
    if not os.path.isfile(file_path):
        return ""

    ctype, _ = mimetypes.guess_type(file_path)
    if ctype is None:
        ctype = ""

    try:
        with open(file_path, "rb") as f:
            data = f.read()
    except Exception:
        return ""

    return extract_attachment_text_from_bytes(
        content_type=ctype,
        filename=os.path.basename(file_path),
        data=data,
        charset=None,
    )
