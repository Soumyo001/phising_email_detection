import os
import re
from email import policy
from email.parser import BytesParser

BINARY_TYPES = {
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/octet-stream",
    "application/zip",
    "application/x-rar",
    "application/javascript",
    "application/x-javascript",
}

def scan_eml_for_attachments(root):
    total = 0
    with_attach = 0
    extracted_text = 0
    mime_count = {}

    print(f"\n===== SCANNING EML FOLDER: {root} =====")

    for dirpath, _, files in os.walk(root):
        for f in files:
            if not f.lower().endswith(".eml"):
                continue
            total += 1

            path = os.path.join(dirpath, f)
            try:
                msg = BytesParser(policy=policy.default).parse(open(path, "rb"))
            except:
                continue

            has_attachment = False

            for part in msg.walk():
                ctype = part.get_content_type()
                disp = str(part.get("Content-Disposition", "") or "").lower()

                filename = part.get_filename()
                is_attach = (
                    "attachment" in disp
                    or filename
                    or ctype in BINARY_TYPES
                    or ctype.startswith("application/")
                )

                if is_attach:
                    has_attachment = True
                    mime_count[ctype] = mime_count.get(ctype, 0) + 1

            if has_attachment:
                with_attach += 1

    print(f"Path                    : {root}")
    print(f"Total EML files         : {total:,}")
    print(f"Emails with attachments : {with_attach:,}")
    print(f"Attachment ratio        : {with_attach/total*100:.2f}%")
    print("\nMIME Type Counts:")
    for k, v in sorted(mime_count.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


scan_eml_for_attachments("../headers/1")
scan_eml_for_attachments("../headers/2")
scan_eml_for_attachments("../headers/3")
scan_eml_for_attachments("../headers/4")
scan_eml_for_attachments("../headers/5")
