import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.predict import load_model, predict_single, predict_batch
from src.config import MODEL_PATH

st.set_page_config(page_title="Nigeria Loan Default Predictor", layout="wide")
st.title("Nigeria Loan Default Predictor")
st.markdown(
    "Predict whether a loan applicant will default based on their profile and loan history."
)


@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Run python src/train.py first.")
        st.stop()
    return load_model()


model = get_model()
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.header("Single Loan Application")
    col1, col2, col3 = st.columns(3)
    with col1:
        loannumber = st.number_input("Loan Number", 1, 100, 5)
        loanamount = st.number_input("Loan Amount (NGN)", 1000.0, 5000000.0, 30000.0)
        totaldue = st.number_input("Total Due (NGN)", 1000.0, 6000000.0, 36000.0)
        termdays = st.number_input("Term Days", 7, 365, 30)
        is_referred = st.selectbox(
            "Referred by another customer?",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
    with col2:
        bank_account_type = st.selectbox(
            "Bank Account Type", ["Savings", "Current", "Unknown"]
        )
        employment_status_clients = st.selectbox(
            "Employment Status",
            ["Permanent", "Self-Employed", "Unemployed", "Student", "Unknown"],
        )
        bank_name_clients = st.selectbox(
            "Bank Name", ["GTBank", "Access", "First Bank", "Zenith", "UBA", "Unknown"]
        )
        age = st.number_input("Age", 18, 80, 30)
    with col3:
        prev_num_loans = st.number_input("Previous Number of Loans", 0, 50, 3)
        prev_late_payments = st.number_input("Previous Late Payments", 0, 20, 0)
        prev_late_payment_rate = st.slider("Previous Late Payment Rate", 0.0, 1.0, 0.0)
        approval_month = st.selectbox("Approval Month", list(range(1, 13)))
        approval_year = st.selectbox("Approval Year", [2017, 2018, 2019, 2020])

    if st.button("Predict Default Risk", type="primary"):
        input_data = {
            "loannumber": loannumber,
            "loanamount": loanamount,
            "totaldue": totaldue,
            "termdays": termdays,
            "days_to_approve": 1,
            "interest_amount": totaldue - loanamount,
            "interest_rate": (totaldue - loanamount) / loanamount,
            "is_referred": is_referred,
            "approval_month": approval_month,
            "approval_year": approval_year,
            "bank_account_type": bank_account_type,
            "longitude_gps": 3.3792,
            "latitude_gps": 6.5244,
            "bank_name_clients": bank_name_clients,
            "employment_status_clients": employment_status_clients,
            "age": age,
            "prev_num_loans": prev_num_loans,
            "prev_avg_loan_amount": loanamount,
            "prev_total_borrowed": loanamount * prev_num_loans,
            "prev_avg_totaldue": totaldue,
            "prev_avg_termdays": termdays,
            "prev_max_loannumber": loannumber,
            "prev_avg_interest_rate": (totaldue - loanamount) / loanamount,
            "prev_avg_loan_duration": termdays,
            "prev_late_payments": prev_late_payments,
            "prev_late_payment_rate": prev_late_payment_rate,
            "prev_avg_days_to_repay": 0.0,
        }
        result = predict_single(model, input_data)
        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Prediction", result["label"])
        c2.metric("Default Probability", f"{result['probability']:.1%}")
        if result["prediction"] == 1:
            st.error("High default risk. Consider declining this loan application.")
        else:
            st.success("Low default risk. This applicant is likely to repay.")

with tab2:
    st.header("Batch Prediction from CSV")
    st.write("Upload a CSV file with loan application data.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        if st.button("Run Predictions", type="primary"):
            results = predict_batch(model, df)
            st.dataframe(results[["Prediction", "Probability", "Label"]].head(20))
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Applications", len(results))
            c2.metric("Predicted Defaults", int(results["Prediction"].sum()))
            c3.metric("Default Rate", f"{results['Prediction'].mean():.1%}")
            fig, ax = plt.subplots(figsize=(5, 4))
            results["Label"].value_counts().plot(
                kind="bar", ax=ax, color=["steelblue", "tomato"], edgecolor="black"
            )
            ax.set_title("Prediction Distribution")
            ax.tick_params(axis="x", rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.download_button(
                "Download Predictions", results.to_csv(index=False), "predictions.csv"
            )
