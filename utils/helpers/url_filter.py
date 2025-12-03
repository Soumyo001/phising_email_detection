import re
from urllib.parse import urlparse
from data.constants import TRACKING_EXT, BENIGN_ROOTS, SUSPICIOUS_TLDS

def looks_too_short(u: str) -> bool:
    return len(u) < 5

def looks_too_long(u: str) -> bool:
    return len(u) > 512


def defang_cleanup(u: str) -> str:
    """
    Convert defanged URLs to normal form as much as possible:
    hxxp -> http, [.] -> ., etc.
    """
    if not isinstance(u, str):
        return ""
    u = u.strip()

    u = u.replace("hxxp://", "http://")
    u = u.replace("hxxps://", "https://")
    u = u.replace("hxxp:", "http:")
    u = u.replace("hxxps:", "https:")

    # Common defanging patterns
    u = u.replace("[.]", ".")
    u = u.replace("(.)", ".")
    u = u.replace(" . ", ".")
    u = u.replace("[dot]", ".")
    u = u.replace("{.}", ".")
    u = u.replace(":///", "://")

    # Remove spaces just in case
    u = re.sub(r"\s+", "", u)

    return u


def normalize_url_min(u: str) -> str:
    """
    Very lightweight normalization for deduping & modeling:
    - strip
    - lowercase
    - remove leading protocol & www.
    """
    if not isinstance(u, str):
        return ""
    u = u.strip()
    u = u.lower()

    # Strip protocol
    u = re.sub(r"^https?://", "", u)
    # Strip leading www.
    u = re.sub(r"^www\.", "", u)

    return u


def keep_domain_like(u: str) -> bool:
    if not isinstance(u, str) or not u:
        return False
    return True if extract_domain(u) != "" else False

# normalize urls
def normalize_url(u: str) -> str:
    u = u.strip()
    u = u.replace("\\n", "").replace("\\r", "")
    u = re.sub(r"\s+", "", u)
    u = u.lower()
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    # remove surrounding angle brackets etc
    u = u.strip(" <>\"'")
    return u

def extract_domain(url_norm: str) -> str:
    """extract root domain from normalized URL"""
    try:
        # ensure scheme exists or urlparse fails
        parsed = urlparse("http://" + url_norm)
        host = parsed.netloc.lower()
        # Take last 2 components: domain + tld
        parts = host.split(".")
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return host
    except:
        return ""

def looks_benign(url):
    d = extract_domain(url)
    return (d in BENIGN_ROOTS) or url.endswith(TRACKING_EXT)


def probable_phish(url):
    u = url.lower()

    # 1) suspicious keywords common in phishing
    KEYWORDS = [
        "login","secure","verify","update","account",
        "password","session","token","reset","confirm"
    ]
    if any(k in u for k in KEYWORDS):
        return True

    # 2) suspicious TLDs
    if any(u.endswith(tld) for tld in SUSPICIOUS_TLDS):
        return True

    # 3) very long random-looking tokens
    if re.search(r"[0-9a-z]{15,}", u):
        return True

    return False