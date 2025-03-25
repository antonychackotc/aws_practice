
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the saved logistic regression model
logi_model = joblib.load('logistic_regression_model.pkl')
st.write("Model loaded successfully!")

# Load scaler
scaler_3 = joblib.load('scaler-3.pkl')

# Streamlit app title
st.title('Claim Prediction App')
st.subheader('Claim Prediction Based on History Claim Less Than 2')
st.write('A claim is filed based on the following conditions:\n'
         '- Claim history more than 2\n'
         '- Policy type health more than 3\n'
         '- Policy type more than 1\n'
         '- Gender female with policy type health more than 4\n'
         '- Vehicle/Property Age more than 15')

# Input fields for model features
st.subheader('Enter Input Values')
customer_age = st.number_input('Customer Age', value=30)
annual_income_raw = st.number_input('Annual Income (Raw Value)', value=50000.0)
premium_amount_raw = st.number_input('Premium Amount (Raw Value)', value=1000.0)
vehicle_property_age = st.number_input('Vehicle/Property Age', value=5)
claim_history = st.number_input('Claim History', value=1)

# Categorical inputs
policy_type = st.selectbox('Policy Type', ['Auto', 'Health', 'Life', 'Property'])
gender = st.selectbox('Gender', ['Female', 'Male', 'Other'])

# One-hot encoding for policy type and gender
policy_type_encoded = [1 if policy_type == 'Auto' else 0,
                       1 if policy_type == 'Health' else 0,
                       1 if policy_type == 'Life' else 0,
                       1 if policy_type == 'Property' else 0]

gender_encoded = [1 if gender == 'Female' else 0,
                  1 if gender == 'Male' else 0,
                  1 if gender == 'Other' else 0]

# Scale numerical values
scaling_input_3 = pd.DataFrame([[annual_income_raw, premium_amount_raw]],
                                columns=['Annual_Income', 'Premium_Amount'])
scaled_values_3 = scaler_3.transform(scaling_input_3)[0]

# Prepare input data
new_data = pd.DataFrame({
    'Customer_Age': [customer_age],
    'Policy_Type_Auto': [policy_type_encoded[0]],
    'Policy_Type_Health': [policy_type_encoded[1]],
    'Policy_Type_Life': [policy_type_encoded[2]],
    'Policy_Type_Property': [policy_type_encoded[3]],
    'Gender_Female': [gender_encoded[0]],
    'Gender_Male': [gender_encoded[1]],
    'Gender_Other': [gender_encoded[2]],
    'Annual_Income': [scaled_values_3[0]],
    'Vehicle_Age_Property_Age': [vehicle_property_age],
    'Premium_Amount': [scaled_values_3[1]],
    'Claim_History': [claim_history]
})

# Predict class probabilities and class
if st.button('Predict'):
    try:
        predictions = logi_model.predict_proba(new_data)
        predicted_class = logi_model.predict(new_data)

        # Map the prediction to risk levels
        risk_labels = {0: 'Filed Claim 2 or Below / Did Not File a Claim (0)',
                       1: 'Already Filed Claims More Than 2 (1)'}
        risk_score = risk_labels.get(predicted_class[0], 'Unknown')

        st.subheader('Prediction Results')
        st.write(f"Predicted class: {predicted_class[0]}")
        st.write(f"Risk Score: {risk_score}")
        st.write(f"Class probabilities: {predictions}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
