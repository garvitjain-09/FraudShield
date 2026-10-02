import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


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


def predict_fraud(transaction):

    df = pd.DataFrame([transaction])

    destination = df["nameDest"].iloc[0]

    dest_count = dest_counts.get(destination, 0)
    df["dest_transaction_count"] = dest_count

    df["balance_diff_orig"] = (
        df["oldbalanceOrg"]
        - df["amount"]
        - df["newbalanceOrig"]
    )

    df["balance_diff_dest"] = (
        df["oldbalanceDest"]
        + df["amount"]
        - df["newbalanceDest"]
    )

    df.drop(
        columns=["nameOrig", "nameDest"],
        inplace=True,
        errors="ignore"
    )

    encoded_type = encoder.transform(df[["type"]])

    encoded_df = pd.DataFrame(
        encoded_type,
        columns=encoder.get_feature_names_out(["type"])
    )

    df.drop(columns=["type"], inplace=True)

    df = pd.concat(
        [
            df.reset_index(drop=True),
            encoded_df.reset_index(drop=True)
        ],
        axis=1
    )

    columns = [
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

    df = df[columns]

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return prediction, probability, dest_count


st.title("🛡️ FraudShield")

st.subheader("Intelligent Transaction Fraud Detection System")

st.write(
    "Enter transaction details below to check whether "
    "the transaction is potentially fraudulent."
)


step = st.number_input(
    "Step",
    min_value=1,
    max_value=95,
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


if st.button("🔍 Predict Fraud"):

    error = None

    if amount <= 0:

        error = (
            "Transaction amount must be greater than 0."
        )

    elif (
        transaction_type
        in ["CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT"]
        and amount > oldbalanceOrg
    ):

        error = (
            "Transaction amount cannot be greater than "
            "the origin account balance."
        )

    elif not nameOrig.strip():

        error = "Please enter the origin account ID."

    elif not nameDest.strip():

        error = "Please enter the destination account ID."

    if error:

        st.error("❌ " + error)

        st.info(
            "Please correct the transaction details "
            "before running the prediction."
        )

    else:

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

        prediction, probability, dest_count = predict_fraud(
            transaction
        )

        st.write("### Transaction Summary")

        st.write(
            f"**Transaction Type:** {transaction_type}"
        )

        st.write(
            f"**Amount:** ₹{amount:,.2f}"
        )

        st.write(
            f"**Origin Account:** {nameOrig}"
        )

        st.write(
            f"**Destination Account:** {nameDest}"
        )

        st.write(
            f"**Origin Balance:** ₹{oldbalanceOrg:,.2f}"
        )

        st.write(
            f"**Destination Balance:** ₹{oldbalanceDest:,.2f}"
        )

        if dest_count == 0:

            st.info(
                "Destination account was not found in the "
                "training history. Transaction count was set to 0."
            )

        else:

            st.write(
                f"**Previous Destination Transactions:** {dest_count}"
            )

        st.write("### Prediction")

        if prediction == 1:

            st.error("🚨 Potential Fraud Detected")

        else:

            st.success("✅ Transaction Appears Legitimate")

        st.metric(
            "Fraud Probability",
            f"{probability:.2%}"
        )

        if probability >= 0.70:

            st.error(
                "🚨 HIGH RISK - Potential Fraud Detected"
            )

        elif probability >= 0.30:

            st.warning(
                "⚠️ MEDIUM RISK - Review Transaction"
            )

        else:

            st.success(
                "✅ LOW RISK - Transaction Appears Legitimate"
            )