import re

EMAIL_REGEX = re.compile(r'[\w\.-]+@[\w\.-]+')
URL_REGEX = re.compile(r'(https?://[^\s"<>\]]+)', flags=re.IGNORECASE)
UNIFIED_COLUMNS = [
    "id",
    "subject",
    "body_text",
    "attachment_text",
    "headers_raw",
    "from_email",
    "to_email",
    "reply_to_email",
    "date",
    "urls",
    "header_from_reply_mismatch",
    "header_domain_mismatch",
    "header_suspicious_tld",
    "header_received_count",
    "header_to_anomaly",
    "header_x_mailer_anomaly",
    "header_date_malformed",
    "label",
    "source",
]

OUT_DIR = "datasets/processed"