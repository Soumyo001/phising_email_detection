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

OUT_DIR = "datasets/final"

BINARY_TYPES = [
        'application/x-tex', 
        'application/x-sh', 
        'application/x-powershell', 
        'application/msword', 
        'application/javascript', 
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
        'text/xml', 
        'application/x-bat', 
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 
        'application/x-toml', 
        'application/x-python', 
        'application/x-javascript', 
        'text/x-csrc', 
        'application/x-yaml', 
        'text/x-python', 
        'application/x-latex', 
        'application/x-perl', 
        'text/html', 
        'text/plain', 
        'application/json', 
        'text/json', 
        'text/csv', 
        'application/vnd.ms-excel', 
        'application/x-cmd', 
        'application/x-ruby', 
        'text/javascript', 
        'text/x-c++src', 
        'application/pdf', 
        'text/markdown', 
        'application/xml'
    ]