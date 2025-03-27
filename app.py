import streamlit as st
import pickle
import pandas as pd
import joblib

# Load model and transformers
model = pickle.load(open("fair_rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

st.set_page_config(page_title="Fair Credit Risk Scoring", layout="centered")
st.title("💼 Fair Credit Risk Scorer")
st.markdown("Enter borrower details below to assess their credit risk:")

# 📝 Input form
age = st.number_input("Age", min_value=18, max_value=100)
savings = st.number_input("Log of Savings (SAVINGS_log)", min_value=0.0)
r_debt_income = st.number_input("Log of Recent Debt / Income (R_DEBT_INCOME_log)", min_value=0.0)
debt_income = st.number_input("Log of Debt / Income Ratio", min_value=0.0)
savings_income = st.number_input("Log of Savings / Income Ratio", min_value=0.0)

# Dropdowns for categorical inputs
education = st.selectbox("Education", encoder.categories_[0])
marital_status = st.selectbox("Marital Status", encoder.categories_[1])
occupation = st.selectbox("Occupation", encoder.categories_[2])
relationship = st.selectbox("Relationship", encoder.categories_[3])
cat_debt = st.selectbox("Debt Category", encoder.categories_[4])
cat_savings = st.selectbox("Savings Account Category", encoder.categories_[5])

if st.button("🧠 Predict Credit Risk"):
    try:
        # Step 1: Create the input dataframe
        input_df = pd.DataFrame([{
            "age": age,
            "SAVINGS_log": savings,
            "R_DEBT_INCOME_log": r_debt_income,
            "DEBT_to_INCOME_log": debt_income,
            "SAVINGS_to_INCOME_log": savings_income,
            "education": education,
            "marital.status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "CAT_DEBT": cat_debt,
            "CAT_SAVINGS_ACCOUNT": cat_savings
        }])

        # Step 2: Define column groups
        cat_cols = ["education", "marital.status", "occupation", "relationship", "CAT_DEBT", "CAT_SAVINGS_ACCOUNT"]
        num_cols = ["age", "SAVINGS_log", "R_DEBT_INCOME_log", "DEBT_to_INCOME_log", "SAVINGS_to_INCOME_log"]

        # Step 3: Encode categorical features
        encoded = encoder.transform(input_df[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

        # Step 4: Merge numerical + encoded features
        final_df = pd.concat([input_df[num_cols].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

        # Step 5: Scale input
        scaled_input = scaler.transform(final_df)

        # Step 6: Predict
        prediction = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1]

        # Step 7: Display results
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ High Credit Risk (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Low Credit Risk (Probability: {prob:.2f})")

    except Exception as e:
        st.error(f" Something went wrong during prediction:\n\n{str(e)}")
