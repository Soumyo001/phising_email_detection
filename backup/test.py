from email import policy
from email.parser import BytesParser
from pathlib import Path
from charset_normalizer import from_bytes
import os

path = "/mnt/partition1/machine_learning/phising_email_detection/headers/3/0a7ba88b4e8d016347102318c79932ab.eml"

TEXT_EXTS = {
    ".txt",".csv",".json",".xml",".html",".htm",".md",".log",".ini",".cfg",".conf",
    ".tex",".yml",".yaml",".js",".ts",".java",".c",".cpp",".h",".hpp",".cs",".php",
    ".pl",".rb",".sh",".bat",".cmd",".ps1",".py",".r",".go",".rs",".swift",".m",".kt",
    ".ics"
}

TEXT_ATTACHMENT_TYPES = {"text/plain","text/html","text/markdown","text/csv","text/tab-separated-values","text/xml","application/xml","text/json","application/json","text/javascript","application/x-javascript","text/x-python","text/x-csrc","text/x-c++src","text/x-java-source","text/x-typescript","text/x-php","application/x-php","text/x-go","text/x-rustsrc","text/x-swift","text/x-objcsrc","application/x-sh","application/x-powershell","application/x-perl","application/x-ruby","application/x-tex","text/calendar","application/vnd.openxmlformats-officedocument.wordprocessingml.document","application/msword","application/vnd.ms-excel","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet","application/xhtml+xml","application/pdf"
}

def _decode_header_safely(raw_bytes: bytes, path: str) -> str:
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

def separate_header_blocks(path: str):
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
    headers_raw = _decode_header_safely(raw_header_block, path)
    return raw_header_block, raw_body_block, headers_raw

a, b, c = separate_header_blocks(path)

msg = BytesParser(policy=policy.default).parse(open(path, "rb"))

# for name, value in msg.items():
#     print(f"{name}: {value}")

def extract_headers_from_eml_bytes(data: bytes) -> str:
    """
    Return canonical header block (RFC style) for an EML bytes blob.
    This avoids accidentally preserving body parts as headers.
    """
    try:
        msg = BytesParser(policy=policy.default).parsebytes(data)
    except Exception:
        # fallback: try lenient parsing
        from email import message_from_bytes
        msg = message_from_bytes(data)

    # Build header string from items (preserves order)
    hdr_lines = []
    for name, value in msg.items():
        # value may be folded; produce a single-line representation
        hdr_lines.append(f"{name}: {value}")
    return "\r\n".join(hdr_lines)


headers_only = extract_headers_from_eml_bytes(open(path, "rb").read())
print(c.replace("\r\n", "\n").replace("\r", "\n"))

print("============")
print(headers_only)

