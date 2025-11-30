import os
import PyPDF2
import docx
import pytesseract
from PIL import Image
import utils.helpers.preprocess_helper as helper

def extract_attachment_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    # TXT
    if ext == ".txt":
        return open(file_path, "r", errors="ignore").read()

    # HTML
    if ext in [".html", ".htm"]:
        raw = open(file_path, "r", errors="ignore").read()
        return helper.clean_html_simple(raw) 

    # PDF
    if ext == ".pdf":
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text

    # DOCX
    if ext == ".docx":
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    # Images (OCR)
    if ext in [".jpg", ".jpeg", ".png"]:
        img = Image.open(file_path)
        return pytesseract.image_to_string(img)

    # DEFAULT — return empty or unsupported
    return ""
