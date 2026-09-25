import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ==============================
# Load Model and Encoder
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "fraud_xgb_model.pkl"
ENCODER_PATH = MODEL_DIR / "type_encoder.pkl"
DEST_COUNTS_PATH = MODEL_DIR / "dest_counts.pkl"

@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    dest_counts = joblib.load(DEST_COUNTS_PATH)

    return model, encoder, dest_counts


model, encoder, dest_counts = load_resources()


# =========================
# Prediction Function
# =========================

def predict_fraud(transaction):

    new_df = pd.DataFrame([transaction])

    # Destination transaction count
    destination = new_df["nameDest"].iloc[0]

    new_df["dest_transaction_count"] = dest_counts.get(destination, 0)

    new_df["dest_transaction_count"] = (
        dest_counts.get(destination, 0)
    )

    # Engineered features
    new_df["balance_diff_orig"] = (
        new_df["oldbalanceOrg"]
        - new_df["amount"]
        - new_df["newbalanceOrig"]
    )

    new_df["balance_diff_dest"] = (
        new_df["oldbalanceDest"]
        + new_df["amount"]
        - new_df["newbalanceDest"]
    )

    # Remove account IDs
    new_df = new_df.drop(
        columns=["nameOrig", "nameDest"],
        errors="ignore"
    )

    # Encode transaction type
    type_encoded = encoder.transform(
        new_df[["type"]]
    )

    type_encoded_df = pd.DataFrame(
        type_encoded,
        columns=encoder.get_feature_names_out(["type"])
    )

    # Remove original type
    new_num = new_df.drop(columns=["type"])

    # Combine features
    new_final = pd.concat(
        [
            new_num.reset_index(drop=True),
            type_encoded_df.reset_index(drop=True)
        ],
        axis=1
    )

    # Match training column order
    new_final = new_final[
        [
            "step",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "balance_diff_orig",
            "balance_diff_dest",
            "dest_transaction_count",
            "type_CASH_IN",
            "type_CASH_OUT",
            "type_DEBIT",
            "type_PAYMENT",
            "type_TRANSFER"
        ]
    ]

    # Prediction
    prediction = model.predict(new_final)[0]

    probability = model.predict_proba(new_final)[0][1]

    return prediction, probability


# =========================
# Streamlit UI
# =========================

st.title("🛡️ FraudShield")

st.subheader(
    "Intelligent Transaction Fraud Detection System"
)

st.write(
    "Enter transaction details below to predict "
    "whether a transaction is potentially fraudulent."
)


# =========================
# Input Fields
# =========================

step = st.number_input(
    "Step",
    min_value=1,
    value=76
)

transaction_type = st.selectbox(
    "Transaction Type",
    [
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER"
    ]
)

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=10000.0
)

oldbalanceOrg = st.number_input(
    "Old Balance (Origin)",
    min_value=0.0,
    value=10000.0
)

newbalanceOrig = st.number_input(
    "New Balance (Origin)",
    min_value=0.0,
    value=0.0
)

oldbalanceDest = st.number_input(
    "Old Balance (Destination)",
    min_value=0.0,
    value=0.0
)

newbalanceDest = st.number_input(
    "New Balance (Destination)",
    min_value=0.0,
    value=0.0
)

nameOrig = st.text_input(
    "Origin Account ID",
    value="C123456"
)

nameDest = st.text_input(
    "Destination Account ID",
    value="C987654"
)


# =========================
# Prediction Button
# =========================

if st.button("🔍 Predict Fraud"):

    transaction = {
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "nameOrig": nameOrig,
        "nameDest": nameDest,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }

    prediction, probability = predict_fraud(transaction)

    st.write("### Prediction")

    st.metric(
        "Fraud Probability",
        f"{probability:.2%}"
    )

    if probability >= 0.70:
        st.error("🚨 HIGH RISK - Potential Fraud Detected")

    elif probability >= 0.30:
        st.warning("⚠️ MEDIUM RISK - Review Transaction")

    else:
        st.success("✅ LOW RISK - Transaction Appears Legitimate")
