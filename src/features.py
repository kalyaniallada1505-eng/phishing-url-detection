"""URL feature engineering for phishing detection."""
import math
import re
from collections import Counter
from urllib.parse import urlparse

import pandas as pd

SUSPICIOUS_WORDS = [
    "login", "verify", "secure", "account", "update", "bank", "confirm",
    "signin", "password", "paypal", "free", "bonus", "wallet",
]
SHORTENERS = ("bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd")
IP_PATTERN = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

FEATURE_NAMES = [
    "url_length", "host_length", "path_length", "num_dots", "num_hyphens",
    "num_digits", "digit_ratio", "num_special_chars", "has_at_symbol",
    "has_ip_host", "is_https", "num_subdomains", "num_suspicious_words",
    "uses_shortener", "entropy", "num_query_params", "has_double_slash_path",
]


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def extract_features(url: str) -> dict:
    """Turn one raw URL into a dict of numeric features."""
    url = str(url).strip()
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    lowered = url.lower()
    digits = sum(ch.isdigit() for ch in url)
    parts = host.split(".") if host else []

    return {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(path),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": digits,
        "digit_ratio": digits / len(url) if url else 0.0,
        "num_special_chars": sum(not ch.isalnum() for ch in url),
        "has_at_symbol": int("@" in url),
        "has_ip_host": int(bool(IP_PATTERN.match(host))),
        "is_https": int(parsed.scheme == "https"),
        "num_subdomains": max(len(parts) - 2, 0),
        "num_suspicious_words": sum(w in lowered for w in SUSPICIOUS_WORDS),
        "uses_shortener": int(any(s in host for s in SHORTENERS)),
        "entropy": _entropy(url),
        "num_query_params": len([q for q in parsed.query.split("&") if q]),
        "has_double_slash_path": int("//" in path),
    }


def build_feature_frame(urls) -> pd.DataFrame:
    """Vectorise many URLs into a DataFrame with a fixed column order."""
    rows = [extract_features(u) for u in urls]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)
