import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import FEATURE_NAMES, build_feature_frame, extract_features  # noqa: E402


def test_feature_columns_are_stable():
    df = build_feature_frame(["https://example.com", "http://1.2.3.4/login"])
    assert list(df.columns) == FEATURE_NAMES
    assert len(df) == 2


def test_ip_host_and_at_symbol_detected():
    f = extract_features("http://192.168.1.5/login@verify")
    assert f["has_ip_host"] == 1
    assert f["has_at_symbol"] == 1
    assert f["is_https"] == 0


def test_clean_url_has_no_flags():
    f = extract_features("https://www.python.org/downloads/")
    assert f["has_ip_host"] == 0
    assert f["has_at_symbol"] == 0
    assert f["is_https"] == 1
    assert f["num_suspicious_words"] == 0


def test_suspicious_words_counted():
    f = extract_features("http://secure-login-verify.example.xyz/account")
    assert f["num_suspicious_words"] >= 4
