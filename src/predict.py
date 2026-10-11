"""Score one or more URLs with a trained model.

Usage:
    python src/predict.py --model models/model.joblib http://example.com/login
"""
import argparse
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(__file__))
from features import build_feature_frame  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("urls", nargs="+")
    parser.add_argument("--model", default="models/model.joblib")
    args = parser.parse_args()

    model = joblib.load(args.model)
    proba = model.predict_proba(build_feature_frame(args.urls))[:, 1]
    for url, p in zip(args.urls, proba):
        verdict = "PHISHING" if p >= 0.5 else "legitimate"
        print(f"{verdict:<11} p(phishing)={p:.2f}  {url}")


if __name__ == "__main__":
    main()
