from flask import Flask, render_template, request, redirect, url_for, session, flash
import joblib
import numpy as np
import re
from urllib.parse import urlparse
import tldextract

# NEW: for simple web mining (content-based)
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
app.secret_key = "change_this_secret_key"

# ---------- LOGIN CONFIG ----------
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin"   # you can change this

# ---------- LOAD MODEL ----------
model = joblib.load("rf_phishing_model.pkl")   # your best model


# ---------- SIMPLE WEB-MINING CONFIG ----------
# A very basic keyword list for terrorism / phishing content.
# You can expand / refine this list and even load from a file.
TERRORISM_KEYWORDS = [
    "terror", "terrorism", "bomb", "bombing", "attack", "jihad", "isis",
    "al-qaeda", "extremist", "radicalization", "martyrdom",
    "suicide attack", "lone wolf", "caliphate"
]

PHISHING_KEYWORDS = [

    "verify your account", "update your password", "bank login",
    "limited time", "urgent action required", "confirm your identity",
    "reset your password", "card details", "account suspended",
    # NEW, more generic phishing-ish tokens:
    "sign in", "log in", "login", "enter your password",
    "credit card", "debit card", "security code", "cvv",
    "billing information", "account verification", "validate your account"
]



ALL_WEBMINING_KEYWORDS = TERRORISM_KEYWORDS + PHISHING_KEYWORDS


def fetch_page_text(url: str) -> str:
    """
    Fetch the page and extract visible text.
    If anything fails, returns empty string.
    """
    try:
        # Ensure scheme
        if not re.match(r'^https?://', url):
            url_fetch = 'http://' + url
        else:
            url_fetch = url

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; FreedomGuard/1.0)"
        }
        resp = requests.get(url_fetch, timeout=6, headers=headers, verify=True)

        # Only handle text/html
        content_type = resp.headers.get("Content-Type", "")
        text = resp.text

        if "text/html" in content_type.lower():
            soup = BeautifulSoup(text, "html.parser")
            # remove scripts/styles
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()
            page_text = soup.get_text(separator=" ", strip=True)
            return page_text
        else:
            # fallback: raw text
            return text
    except Exception as e:
        print("Web mining fetch error:", e)
        return ""


def compute_webmining_risk(url: str):
    """
    Very simple keyword-based 'web mining' score.
    Returns (unsafe_percentage, matched_keywords).
    If we can't fetch / parse, returns (None, []).
    """
    page_text = fetch_page_text(url)
    if not page_text:
        return None, []

    text_lower = page_text.lower()
    words = text_lower.split()
    total_words = len(words)
    if total_words == 0:
        return None, []

    hits = 0
    matched_keywords = set()

    for kw in ALL_WEBMINING_KEYWORDS:
        kw_lower = kw.lower()
        count_kw = text_lower.count(kw_lower)
        if count_kw > 0:
            hits += count_kw
            matched_keywords.add(kw)

    # Very naive scoring: hits / total_words scaled.
    # You can tune this factor.
    raw_score = hits / max(total_words, 1)
    # scale up and cap at 100%
    unsafe_percentage = min(100.0, raw_score * 10000)  # arbitrary scaling factor

    return round(unsafe_percentage, 2), sorted(matched_keywords)


# ---------- FEATURE EXTRACTION ----------
def extract_features(url: str):
    if not re.match(r'^https?://', url):
        url = 'http://' + url

    parsed = urlparse(url)
    ext = tldextract.extract(url)

    hostname = parsed.netloc
    path = parsed.path
    query = parsed.query

    url_length = len(url)
    hostname_length = len(hostname)
    path_length = len(path)

    num_dots = hostname.count('.')
    num_hyphen = hostname.count('-')
    num_slash = url.count('/')
    num_digits = sum(c.isdigit() for c in url)
    num_params = query.count('=') if query else 0

    has_ip = 1 if re.fullmatch(r'(?:\d{1,3}\.){3}\d{1,3}', hostname.split(':')[0]) else 0
    has_at = 1 if '@' in url else 0
    uses_https = 1 if parsed.scheme == 'https' else 0

    return [
        url_length,
        hostname_length,
        path_length,
        num_dots,
        num_hyphen,
        num_slash,
        num_digits,
        num_params,
        has_ip,
        has_at,
        uses_https,
    ]


# ---------- LOGIN REQUIRED DECORATOR ----------
from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


# ---------- ROUTES ----------
@app.route("/")
def root():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["user"] = username
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/home")
@login_required
def home():
    return render_template("home.html", username=session.get("user"))


import pandas as pd

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    result = None
    url = None
    safe_percentage = None
    unsafe_percentage = None
    batch_results = None
    csv_error = None

    # NEW: for single URL web mining
    webmining_unsafe_percentage = None
    webmining_keywords = []

    if request.method == "POST":
        # CSV upload
        if "csv_file" in request.files and request.files["csv_file"].filename != "":
            file = request.files["csv_file"]
            try:
                df = pd.read_csv(file)

                if "url" in df.columns:
                    urls = df["url"].astype(str)
                else:
                    first_col = df.columns[0]
                    urls = df[first_col].astype(str)

                records = []
                for u in urls:
                    if not isinstance(u, str) or not u.strip():
                        continue

                    feats = np.array(extract_features(u)).reshape(1, -1)
                    proba = model.predict_proba(feats)[0]

                    prob_legit = float(proba[0])
                    prob_phish = float(proba[1])

                    safe_pct = round(prob_legit * 100, 2)
                    unsafe_pct = round(prob_phish * 100, 2)

                    label = "Safe" if prob_legit >= prob_phish else "Unsafe"

                    # OPTIONAL: also do web mining per URL (can be slow for big CSV)
                    # wm_unsafe, wm_keywords = compute_webmining_risk(u)
                    # For now we skip to keep it fast.

                    records.append({
                        "url": u,
                        "label": label,
                        "safe": safe_pct,
                        "unsafe": unsafe_pct,
                    })

                batch_results = records

            except Exception as e:
                print("CSV processing error:", e)
                csv_error = "Could not process the CSV file. Please make sure it is a valid CSV and try again."

        # Single URL
        elif request.form.get("url"):
            url = request.form.get("url")

            feats = np.array(extract_features(url)).reshape(1, -1)
            proba = model.predict_proba(feats)[0]   # [P(legit), P(phishing)]

            prob_legit = float(proba[0])
            prob_phish = float(proba[1])

            safe_percentage = round(prob_legit * 100, 2)
            unsafe_percentage = round(prob_phish * 100, 2)

            label = "Safe" if prob_legit >= prob_phish else "Unsafe"
            is_safe = (label == "Safe")

            # NEW: web mining risk
            webmining_unsafe_percentage, webmining_keywords = compute_webmining_risk(url)

            result = {
                "label": label,
                "is_safe": is_safe,
                "prob_legit": safe_percentage,
                "prob_phish": unsafe_percentage,
            }

    return render_template(
        "predict.html",
        username=session.get("user"),
        url=url,
        result=result,
        safe_percentage=safe_percentage,
        unsafe_percentage=unsafe_percentage,
        batch_results=batch_results,
        csv_error=csv_error,
        # NEW:
        webmining_unsafe_percentage=webmining_unsafe_percentage,
        webmining_keywords=webmining_keywords,
    )


@app.route('/contact')
def contact():
    return render_template('contact.html')


from flask import Flask, request, render_template
from flask import flash
from flask_material import Material
import numpy as np
import joblib
from keras.models import load_model

from flask import render_template, request
from tensorflow.keras.models import load_model
import joblib
import numpy as np

@app.route('/main', methods=["GET", "POST"])
def analyze():
    if request.method == 'POST':
        msg = request.form['msg']

        # Load the saved model, vectorizer, and encoder
        model = load_model('terrorism_model.h5')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        encoder = joblib.load('label_encoder.pkl')

        # Transform input text
        text_tfidf = vectorizer.transform([msg]).toarray()

        # Predict probabilities for each class
        y_pred_proba = model.predict(text_tfidf)[0]  # e.g. [p_class0, p_class1]

        # Map probabilities to safe / unsafe using the label encoder
        classes = list(encoder.classes_)  # e.g. [0, 1] or similar

        # Assume binary classification with labels 0 = safe, 1 = terrorist
        # but still handle generic encoder ordering
        try:
            idx_unsafe = classes.index(1)
            idx_safe = classes.index(0)
        except ValueError:
            # Fallback: assume first class = safe, second = unsafe
            idx_safe, idx_unsafe = 0, 1

        unsafe_prob = float(y_pred_proba[idx_unsafe])
        safe_prob = float(y_pred_proba[idx_safe])

        safe_percentage = round(safe_prob * 100, 2)
        unsafe_percentage = round(unsafe_prob * 100, 2)

        # Predicted label (same as your original logic)
        y_pred_label = encoder.inverse_transform([np.argmax(y_pred_proba)])[0]

        if y_pred_label == 1:
            class1 = "Terrorist Activity Detected"
            is_safe = False
        else:
            class1 = "No Terrorist Activity"
            is_safe = True

        return render_template(
            'contact.html',
            class1=class1,
            res=True,                 # used in template to show result block
            msg=msg,
            safe_percentage=safe_percentage,
            unsafe_percentage=unsafe_percentage,
            is_safe=is_safe
            # add username=username here if you already have it
        )

    # GET request – just show an empty form
    return render_template('contact.html')

    

@app.route("/logout")
@login_required
def logout():
    session.pop("user", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
