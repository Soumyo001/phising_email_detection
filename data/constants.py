import re

EMAIL_REGEX = re.compile(r'[\w\.-]+@[\w\.-]+')
URL_REGEX = re.compile(r'(https?://[^\s"<>\]]+)', flags=re.IGNORECASE)
HEADER_REGEX = re.compile(
    r"(?i)^(from|to|subject|date|cc|bcc|received|message-id|mime-version|content-type):"
)
MAX_HEADER_LEN = 4096     # hard boundary for leakage prevention
MAX_RECEIVED_LINES = 15   # keep only first 10 hops
MAX_RAW_LENGTH = 4096
MAX_CANON_LENGTH = 4096

UNIFIED_COLUMNS = [
    "id",
    "subject",
    "body_text",
    "attachment_text",
    "headers_raw",
    "canonical_raw",
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

TEXT_ATTACHMENT_TYPES = {
    "text/plain",
    "text/html",
    "text/markdown",
    "text/csv",
    "text/tab-separated-values",
    "text/xml",
    "application/xml",
    "text/json",
    "application/json",
    "text/javascript",
    "application/x-javascript",
    "text/x-python",
    "text/x-csrc",
    "text/x-c++src",
    "text/x-java-source",
    "text/x-typescript",
    "text/x-php",
    "application/x-php",
    "text/x-go",
    "text/x-rustsrc",
    "text/x-swift",
    "text/x-objcsrc",
    "application/x-sh",
    "application/x-powershell",
    "application/x-perl",
    "application/x-ruby",
    "application/x-tex",
    "text/calendar",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/xhtml+xml",
    "application/pdf",
}

TEXT_EXTS = {
    ".txt",".csv",".json",".xml",".html",".htm",".md",".log",".ini",".cfg",".conf",
    ".tex",".yml",".yaml",".js",".ts",".java",".c",".cpp",".h",".hpp",".cs",".php",
    ".pl",".rb",".sh",".bat",".cmd",".ps1",".py",".r",".go",".rs",".swift",".m",".kt",
    ".ics"
}


BENIGN_ROOTS = {
    "google.com","gstatic.com","googleusercontent.com","gmail.com",
    "microsoft.com","office.com","live.com","outlook.com","windows.com",
    "facebook.com","fbcdn.net","instagram.com","linkedin.com","twitter.com",
    "amazon.com","aws.amazon.com","apple.com","icloud.com","cdn.apple.com",
    "cloudflare.com","mozilla.org","wikipedia.org",
}

SUSPICIOUS_TLDS = {
    ".tk",".ml",".ga",".gq",".cf",".xyz",".top",".cyou",".monster",".club",
    ".click",".link",".email",".shop",".info",".live",".fit",".rest",".lol",
}

TRACKING_EXT = (".gif",".png",".jpg",".jpeg",".bmp",".ico")

MALICIOUS_URL_LINKS = {
    "url_database_mitchellkrogza.txt": "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt",
    "domain_database_mitchellkrogza.txt": "https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/refs/heads/master/phishing-domains-ACTIVE.txt",
    "url_haus.txt": "https://urlhaus.abuse.ch/downloads/text/",
    "open_phish_url.txt": "https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt",
    "romainmarcoux_full_domain_aa.txt": "https://github.com/romainmarcoux/malicious-domains/raw/refs/heads/main/full-domains-aa.txt",
    "romainmarcoux_full_domain_ab.txt": "https://github.com/romainmarcoux/malicious-domains/raw/refs/heads/main/full-domains-ab.txt",
    "romainmarcoux_full_domain_ac.txt": "https://github.com/romainmarcoux/malicious-domains/raw/refs/heads/main/full-domains-ac.txt",
    "hagezi_pro_plus.txt": "https://raw.githubusercontent.com/hagezi/dns-blocklists/refs/heads/main/domains/pro.plus.txt",
    "hagezi_nrd_35_29.txt": "https://raw.githubusercontent.com/hagezi/dns-blocklists/refs/heads/main/domains/nrd35-29.txt"
}

TRANCO_1M_DAILY_CSV = "https://tranco-list.eu/download/KW3PW/1000000"
TRANCO_1M_FULL_CSV = "https://tranco-list.eu/download/KW3PW/full"
CISCO_TOP_1M_CSV = "https://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip"
MAJESTIC_1M_CSV = "https://downloads.majestic.com/majestic_million.csv"

LEGIT_URL_LINKS = {
    "tranco_1m_daily.csv": TRANCO_1M_DAILY_CSV,
    "tranco_1m_full.csv": TRANCO_1M_FULL_CSV,
    "majestic_1m.csv": MAJESTIC_1M_CSV,
    "cisco_top_1m.csv.zip": CISCO_TOP_1M_CSV
}