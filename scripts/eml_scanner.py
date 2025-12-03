import os
from email import policy
from email.parser import BytesParser
from data.configs import get_eml_datasets
from utils.helpers.preprocess_helper import is_email_file
from data.constants import TEXT_ATTACHMENT_TYPES, TEXT_EXTS

def scan_eml_for_attachments(root):
    total = 0
    with_attach = 0
    extracted_text = 0
    mime_count = {}

    print(f"\n===== SCANNING EML FOLDER: {root} =====")

    for dirpath, _, files in os.walk(root):
        for f in files:
            path = os.path.join(dirpath, f)
            # if not is_email_file(path):
            #     continue
            total += 1

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
                    ("attachment" in disp)
                    or ("filename=" in disp)
                    or (filename and disp.startswith("inline"))
                )
                is_type_text = (
                    (filename and os.path.splitext(filename)[1].lower() in TEXT_EXTS)
                    or ctype in TEXT_ATTACHMENT_TYPES
                    or ctype.startswith("text/")
                )

                if is_attach and is_type_text:
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


for path in get_eml_datasets():
    scan_eml_for_attachments(path.get("root_dir"))