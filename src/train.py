"""Train and evaluate a phishing URL classifier.

Usage:
    python src/train.py --data data/sample_urls.csv --out models/model.joblib

The CSV needs two columns: `url` and `label` (1 = phishing, 0 = legitimate).
"""
import argparse
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_NAMES, build_feature_frame  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_urls.csv")
    parser.add_argument("--out", default="models/model.joblib")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data).dropna(subset=["url", "label"])
    df = df.drop_duplicates(subset="url")
    print(f"Loaded {len(df)} URLs ({int(df.label.sum())} phishing, "
          f"{int((df.label == 0).sum())} legitimate)")

    X = build_feature_frame(df["url"])
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    model = RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight="balanced",
        random_state=args.seed,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\nHold-out results")
    print(classification_report(y_test, preds, target_names=["legitimate", "phishing"]))
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(confusion_matrix(y_test, preds))

    folds = min(5, int(y.value_counts().min()))
    if folds >= 2:
        scores = cross_val_score(model, X, y, cv=folds)
        print(f"\n{folds}-fold CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    importances = sorted(zip(FEATURE_NAMES, model.feature_importances_),
                         key=lambda t: t[1], reverse=True)
    print("\nTop features:")
    for name, score in importances[:8]:
        print(f"  {name:<24}{score:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(model, args.out)
    print(f"\nSaved model to {args.out}")


if __name__ == "__main__":
    main()
