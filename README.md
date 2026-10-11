# AI-Driven Phishing Website Detection Using URL Patterns

A machine learning project that classifies a URL as **phishing** or **legitimate** using only patterns in the URL string itself (no page download needed).

## How it works

1. **Feature engineering** (`src/features.py`): each URL becomes 17 numeric features, such as length, number of dots and hyphens, digit ratio, `@` symbol, IP-address host, HTTPS, subdomain count, suspicious keywords (`login`, `verify`, `secure`, ...), URL shorteners, and character entropy.
2. **Training** (`src/train.py`): a Random Forest (class-balanced) is trained on a labelled CSV, evaluated on a stratified hold-out set and with cross-validation, and saved with joblib. Feature importances are printed so you can see which URL patterns separate phishing from legitimate sites.
3. **Prediction** (`src/predict.py`): score any URL from the command line.

## Quick start

```bash
pip install -r requirements.txt
python src/train.py --data data/sample_urls.csv --out models/model.joblib
python src/predict.py "https://www.python.org/downloads/" "http://paypal.com.verify-login-secure.xyz/account"
python -m pytest tests
```

Example output:

```
legitimate  p(phishing)=0.00  https://www.python.org/downloads/
PHISHING    p(phishing)=0.94  http://paypal.com.verify-login-secure.xyz/account
```

## Important note about the data

`data/sample_urls.csv` is a **small hand-written demo set (40 URLs)** so the project runs out of the box. Its metrics (near-perfect accuracy) are **not meaningful**. For real results, replace it with a large labelled dataset that has `url` and `label` columns (1 = phishing, 0 = legitimate), for example a public phishing URL dataset from Kaggle, or PhishTank/OpenPhish feeds combined with a list of popular legitimate domains. Then re-run training and report the hold-out numbers you get.

## Project structure

```
phishing-url-detection/
├── data/sample_urls.csv     # demo data (replace with a real dataset)
├── src/
│   ├── features.py          # URL feature extraction
│   ├── train.py             # training + evaluation
│   └── predict.py           # command-line scoring
├── tests/test_features.py
└── requirements.txt
```

## Ideas to extend

- Add WHOIS/domain-age and TLS certificate features
- Compare Random Forest with Gradient Boosting and Logistic Regression
- Serve the model with a small Flask or FastAPI endpoint
- Build a Power BI or Streamlit dashboard of detection results

## Tech stack

Python, Pandas, NumPy, Scikit-learn, Joblib
