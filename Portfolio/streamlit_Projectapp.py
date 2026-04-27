import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

from sklearn.pipeline import Pipeline
import shap

from joblib import dump, load


# ── Setup ────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")


# ── Feature Configuration ────────────────────────────────────────────────
# Mirrors FEATURE_KEYS / FEATURE_LABELS / MODEL_INFO in the HW6 template.
# Must match the FEATURE_KEYS list used in ML_Project_Part_4_Final.ipynb.

FEATURE_KEYS = [
    'loan_amnt', 'int_rate_clean', 'installment', 'annual_inc',
    'dti', 'fico_avg', 'revol_util_clean', 'term_months',
    'emp_length_years', 'open_acc', 'pub_rec', 'delinq_2yrs',
    'loan_to_income', 'installment_to_income',
    'log_annual_inc', 'log_loan_amnt', 'log_installment', 'log_revol_bal',
    'credit_history_years', 'issue_year', 'issue_month',
    'mort_acc', 'pub_rec_bankruptcies', 'total_acc', 'revol_bal',
]

FEATURE_LABELS = {
    'loan_amnt':             'Loan Amount ($)',
    'int_rate_clean':        'Interest Rate (%)',
    'installment':           'Monthly Installment ($)',
    'annual_inc':            'Annual Income ($)',
    'dti':                   'Debt-to-Income Ratio (%)',
    'fico_avg':              'FICO Score (Average)',
    'revol_util_clean':      'Revolving Utilization (%)',
    'term_months':           'Loan Term (months)',
    'emp_length_years':      'Employment Length (years)',
    'open_acc':              'Open Credit Accounts',
    'pub_rec':               'Derogatory Public Records',
    'delinq_2yrs':           'Delinquencies (last 2 yrs)',
    'loan_to_income':        'Loan-to-Income Ratio',
    'installment_to_income': 'Installment-to-Income Ratio',
    'log_annual_inc':        'Log(Annual Income)',
    'log_loan_amnt':         'Log(Loan Amount)',
    'log_installment':       'Log(Installment)',
    'log_revol_bal':         'Log(Revolving Balance)',
    'credit_history_years':  'Credit History (years)',
    'issue_year':            'Issue Year',
    'issue_month':           'Issue Month',
    'mort_acc':              'Mortgage Accounts',
    'pub_rec_bankruptcies':  'Public Record Bankruptcies',
    'total_acc':             'Total Credit Accounts',
    'revol_bal':             'Revolving Balance ($)',
}

MODEL_INFO = {
    "endpoint":   None,                             # filled from st.secrets at runtime
    "explainer":  "explainer_lendingclub.shap",
    "pipeline":   "best_model_lendingclub.tar.gz",
    "keys":       FEATURE_KEYS,
    "inputs": [
        {"name": "loan_amnt",            "label": "Loan Amount ($)",            "min": 500.0,    "max": 40000.0,  "default": 10000.0, "step": 500.0},
        {"name": "int_rate_clean",       "label": "Interest Rate (%)",          "min": 5.0,      "max": 31.0,     "default": 13.0,    "step": 0.1},
        {"name": "installment",          "label": "Monthly Installment ($)",    "min": 10.0,     "max": 1600.0,   "default": 300.0,   "step": 5.0},
        {"name": "term_months",          "label": "Loan Term (months)",         "min": 36.0,     "max": 60.0,     "default": 36.0,    "step": 24.0},
        {"name": "annual_inc",           "label": "Annual Income ($)",          "min": 10000.0,  "max": 300000.0, "default": 65000.0, "step": 1000.0},
        {"name": "dti",                  "label": "Debt-to-Income Ratio (%)",   "min": 0.0,      "max": 50.0,     "default": 15.0,    "step": 0.5},
        {"name": "fico_avg",             "label": "FICO Score (Average)",       "min": 580.0,    "max": 850.0,    "default": 700.0,   "step": 1.0},
        {"name": "revol_util_clean",     "label": "Revolving Utilization (%)",  "min": 0.0,      "max": 150.0,    "default": 45.0,    "step": 1.0},
        {"name": "revol_bal",            "label": "Revolving Balance ($)",      "min": 0.0,      "max": 200000.0, "default": 15000.0, "step": 500.0},
        {"name": "emp_length_years",     "label": "Employment Length (years)",  "min": 0.0,      "max": 10.0,     "default": 5.0,     "step": 1.0},
        {"name": "open_acc",             "label": "Open Credit Accounts",       "min": 0.0,      "max": 60.0,     "default": 10.0,    "step": 1.0},
        {"name": "total_acc",            "label": "Total Credit Accounts",      "min": 1.0,      "max": 100.0,    "default": 25.0,    "step": 1.0},
        {"name": "credit_history_years", "label": "Credit History (years)",     "min": 0.0,      "max": 40.0,     "default": 10.0,    "step": 1.0},
        {"name": "mort_acc",             "label": "Mortgage Accounts",          "min": 0.0,      "max": 20.0,     "default": 1.0,     "step": 1.0},
        {"name": "pub_rec",              "label": "Derogatory Public Records",  "min": 0.0,      "max": 10.0,     "default": 0.0,     "step": 1.0},
        {"name": "delinq_2yrs",          "label": "Delinquencies (last 2 yrs)", "min": 0.0,      "max": 20.0,     "default": 0.0,     "step": 1.0},
        {"name": "pub_rec_bankruptcies", "label": "Public Record Bankruptcies", "min": 0.0,      "max": 5.0,      "default": 0.0,     "step": 1.0},
        {"name": "issue_year",           "label": "Issue Year",                 "min": 2007.0,   "max": 2025.0,   "default": 2018.0,  "step": 1.0},
        {"name": "issue_month",          "label": "Issue Month (1–12)",         "min": 1.0,      "max": 12.0,     "default": 6.0,     "step": 1.0},
    ],
}


# ── Access Secrets ───────────────────────────────────────────────────────
# Mirrors the st.secrets block in the HW6 Streamlit template.
# Add these in Streamlit Cloud → Settings → Secrets (TOML format).

aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

MODEL_INFO["endpoint"] = aws_endpoint


# ── AWS Session Management ───────────────────────────────────────────────
# Mirrors get_session() + sm_session in HW6.

@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1',
    )

session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)


# ── Feature Engineering ──────────────────────────────────────────────────
# Compute the 6 derived Feature Set B columns from raw user inputs.

def engineer_features(raw: dict) -> pd.DataFrame:
    d = dict(raw)
    annual_safe  = max(d["annual_inc"], 1.0)
    monthly_safe = max(d["annual_inc"] / 12.0, 1.0)

    d["loan_to_income"]        = d["loan_amnt"]   / annual_safe
    d["installment_to_income"] = d["installment"] / monthly_safe
    d["log_annual_inc"]        = np.log1p(d["annual_inc"])
    d["log_loan_amnt"]         = np.log1p(d["loan_amnt"])
    d["log_installment"]       = np.log1p(d["installment"])
    d["log_revol_bal"]         = np.log1p(d["revol_bal"])

    return pd.DataFrame([{k: d[k] for k in FEATURE_KEYS}])


# ── Load Pipeline from S3 ────────────────────────────────────────────────
# Mirrors load_pipeline() in HW6 — downloads tar.gz from S3, extracts joblib.

def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename  = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=posixpath.join(key, os.path.basename(filename)),
    )

    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    return joblib.load(joblib_file)


# ── Load SHAP Explainer from S3 ──────────────────────────────────────────
# Mirrors load_shap_explainer() in HW6.

def load_shap_explainer(_session, bucket, s3_key, local_path):
    s3_client = _session.client('s3')
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=s3_key)
    with open(local_path, "rb") as f:
        return load(f)


# ── Prediction Logic ─────────────────────────────────────────────────────
# Mirrors call_model_api() in HW6 — calls SageMaker endpoint via Predictor.

def call_model_api(input_df: pd.DataFrame):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer(),
    )
    try:
        raw_pred   = predictor.predict(input_df.values.astype(np.float64))
        pred_class = int(raw_pred[0][0])
        pred_prob  = float(raw_pred[0][1])
        mapping    = {0: "NON-DEFAULT", 1: "DEFAULT"}
        return mapping[pred_class], pred_prob, 200
    except Exception as e:
        return f"Error: {str(e)}", None, 500


# ── SHAP Explanation ─────────────────────────────────────────────────────
# Mirrors display_explanation() in HW6 — preprocesses input, runs SHAP,
# plots waterfall / bar chart, and surfaces the top business insight.

def display_explanation(input_df: pd.DataFrame, _session, bucket):
    explainer_name = MODEL_INFO["explainer"]
    local_path     = os.path.join(tempfile.gettempdir(), explainer_name)

    explainer = load_shap_explainer(
        _session, bucket,
        posixpath.join("explainer", explainer_name),
        local_path,
    )

    best_pipeline  = load_pipeline(_session, bucket, "sklearn-pipeline-deployment")
    preprocessing  = Pipeline(steps=best_pipeline.steps[:-1])
    X_transformed  = preprocessing.transform(input_df)
    n_features     = X_transformed.shape[1]
    feat_names     = FEATURE_KEYS[:n_features]
    X_df           = pd.DataFrame(X_transformed, columns=feat_names)

    shap_values = explainer(X_df)

    # Handle both old-style (ndarray/list) and new Explanation object
    if hasattr(shap_values, "values"):
        sv_arr = shap_values.values
        # Binary tree: shape (1, n_features) or (1, n_features, 2)
        if sv_arr.ndim == 3:
            sv = sv_arr[0, :, 1]
        else:
            sv = sv_arr[0]
    elif isinstance(shap_values, list):
        sv = np.array(shap_values[1][0])
    else:
        sv = np.array(shap_values[0])

    st.subheader("🔍 Decision Transparency (SHAP)")

    sv_abs    = pd.Series(np.abs(sv), index=feat_names).sort_values(ascending=False)
    top_items = sv_abs.head(12).sort_values()

    high_thresh = sv_abs.quantile(0.75)
    colors      = ["#d32f2f" if v >= high_thresh else "#1976d2" for v in top_items.values]

    fig, ax = plt.subplots(figsize=(10, 5))
    top_items.plot(kind="barh", ax=ax, color=colors)
    ax.set_xlabel("|SHAP Value| — contribution to default probability")
    ax.set_title("Top 12 Feature Contributions to This Prediction")
    ax.set_yticklabels([FEATURE_LABELS.get(f, f) for f in top_items.index])
    plt.tight_layout()
    st.pyplot(fig)

    top_feat = sv_abs.index[0]
    st.info(
        f"**Business Insight:** The most influential factor in this prediction "
        f"was **{FEATURE_LABELS.get(top_feat, top_feat)}**."
    )


# ── Streamlit UI ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LendingClub Default Predictor",
    page_icon="🏦",
    layout="wide",
)

with st.sidebar:
    st.header("ℹ️ About This App")
    st.markdown("""
**Project:** LendingClub Loan Default Prediction
**Course:** ML Finance — Spring 2026
**Milestone:** 4 — Final Report

---
**Models evaluated:**
- Logistic Regression (L1 / L2 / Elastic Net)
- Random Forest
- Gradient Boosting
- Huber (SGD Modified Huber)
- Decision Tree (Pruned, depth=5)
- K-Nearest Neighbors (k=15)
- XGBoost
- *(Grid-searched tuned variants)*

**Primary metric:** AUC
**LendingClub baseline (Sifrain 2023):** AUC = 0.67

---
**To deploy:**
1. Run `ML_Project_Part_4_Final.ipynb` through the deployment section
2. Copy the printed values into Streamlit Secrets
3. Push this file + `requirements.txt` to GitHub
4. Connect repo on [share.streamlit.io](https://share.streamlit.io)
    """)
    st.divider()
    st.caption("Sifrain (2023) — *Journal of Financial Risk Management*")

st.title("🏦 LendingClub Loan Default Predictor")
st.markdown(
    "Enter borrower and loan details below. The model will predict whether "
    "the loan is likely to **default** and explain which factors drove the decision."
)
st.divider()

# ── Input Form ───────────────────────────────────────────────────────────

with st.form("pred_form"):
    st.subheader("Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["label"],
                min_value=float(inp["min"]),
                max_value=float(inp["max"]),
                value=float(inp["default"]),
                step=float(inp["step"]),
            )

    submitted = st.form_submit_button("Run Prediction")

# ── Results ──────────────────────────────────────────────────────────────

if submitted:
    input_df = engineer_features(user_inputs)

    label, prob, status = call_model_api(input_df)

    if status == 200:
        outcome_icon = "⚠️" if label == "DEFAULT" else "✅"

        if prob >= 0.60:
            risk = "🔴 High Risk"
        elif prob >= 0.35:
            risk = "🟡 Medium Risk"
        else:
            risk = "🟢 Low Risk"

        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction",           f"{outcome_icon}  {label}")
        col2.metric("Default Probability",  f"{prob:.1%}")
        col3.metric("Risk Level",           risk)

        st.progress(min(prob, 1.0))

        display_explanation(input_df, session, aws_bucket)

    else:
        st.error(label)
