import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib  # Ensure joblib is imported
from sklearn.preprocessing import MinMaxScaler

# Configuring the page with Title, icon, and layout
st.set_page_config(
    page_title="stage Prediction",
    page_icon="/home/hdoop//U5MRIcon.png",
    layout="wide",
    #initial_sidebar_state="collapsed",  # Optional, collapses the sidebar by default
    menu_items={
        'Get Help': 'https://helppage.ethipiau5m.com',
        'Report a Bug': 'https://bugreport.ethipiau5m.com',
        'About': 'https://ethiopiau5m.com',
    },
)

# Custom CSS to adjust spacing
custom_css = """
<style>
    div.stApp {
        margin-top: -90px !important;  /* We can adjust this value as needed */
    }
</style>
"""
# st.markdown(custom_css, unsafe_allow_html=True)
# st.image("cancer.jpg", width=800)  # Change "logo.png" to the path of your logo image file
# Setting the title with Markdown and center-aligning
st.markdown('<h1 style="text-align: center;">MultiModel Credit_Score Prediction</h1>', unsafe_allow_html=True)

# Defining background color
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Defining  header color and font
st.markdown(
    """
    <style>
    h1 {
        color: #800080;  /* Blue color */
        font-family: 'Helvetica', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def horizontal_line(height=1, color="blue", margin="0.5em 0"):
    return f'<hr style="height: {height}px; margin: {margin}; background-color: {color};">'

# Load the CatBoost model
loaded_model = joblib.load('xgb_model2.pkl')

# Load the label encoders
label_encoders = joblib.load('label_encoders2.pkl')

# Load the MinMax scaler parameters
minmax_scalers = joblib.load('scaler_params2.pkl')

# Feature names and types
features = {
    'Occupation': 'categorical',
    'Age': 'numerical',
    'Annual_Income': 'numerical',
    'Interest_Rate': 'numerical',
    'Type_of_Loan': 'categorical',
    'Delay_from_due_date': 'numerical',
    'Num_of_Delayed_Payment': 'numerical',
    'Payment_Behaviour': 'categorical',
    'Monthly_Balance': 'numerical',
    'APPROVED_AMOUNT': 'numerical',
    'TENURE': 'categorical',
    'TERM': 'categorical',
    'LOAN_DESCRIPTION': 'categorical',
    'LOAN_PRODUCT': 'categorical',
    'LTYPE': 'categorical',
    # 'CUST_SHORTNAME': 'categorical',
    # 'DAO_NAME': 'categorical',
    'PRINCIPAL_OS': 'numerical',
    'INTEREST_OS': 'numerical',
    'PRINCIPAL_ARREARS': 'numerical',
    # 'INTEREST_ARREARS': 'numerical',
    'CURRENT_COMMITTMENT': 'numerical',
    'INSTALLMENT_AMOUNT': 'numerical',
    'INSTALLMENT_FREQ_PRINCIPAL': 'numerical',
    'INSTALLMENT_FREQ_INTEREST': 'numerical',
    'RISK_GRADE': 'categorical',
    'ECONOMIC_SECTOR': 'categorical', 
    'INDUSTRY': 'categorical',
    # 'OWNERSHIP': 'categorical',
    'SECTOR': 'categorical',
    'TERM_OF_PAYMENT': 'categorical',
    # 'PRODUCT_OWNER': 'categorical',
    'COLLATTERAL': 'categorical',
    'COLLATERAL_VALUE': 'numerical',
}

# Sidebar title
st.sidebar.title("Input Parameters")
st.sidebar.markdown("""
[Example XLSX input file](https://master/penguins_example.csv)
""")

# Create dictionary for grouping labels
group_labels = {
    'Demographic Data': ['Occupation', 'Age'],
    'Credit_information': ['Annual_Income', 'Interest_Rate', 'Type_of_Loan',
       'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Payment_Behaviour',
       'Monthly_Balance', 'APPROVED_AMOUNT', 'TENURE', 'TERM',
       'LOAN_DESCRIPTION', 'LOAN_PRODUCT', 'LTYPE', 'PRINCIPAL_OS',
       'INTEREST_OS', 'PRINCIPAL_ARREARS', 'CURRENT_COMMITTMENT',
       'INSTALLMENT_AMOUNT', 'INSTALLMENT_FREQ_PRINCIPAL',
       'INSTALLMENT_FREQ_INTEREST', 'RISK_GRADE', 'ECONOMIC_SECTOR',
       'INDUSTRY', 'SECTOR', 'TERM_OF_PAYMENT'],
    
    'Collateral_information': ['COLLATTERAL', 'COLLATERAL_VALUE'],
}

# Option for CSV file upload
uploaded_file = st.sidebar.file_uploader("Upload XLSX file", type=["XLSX"])

# If CSV file is uploaded, read the file
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

# If CSV file is not uploaded, allow manual input
else:
    # Create empty dataframe to store input values
    input_df = pd.DataFrame(index=[0])

    # Loop through features and get user input
    for group, features_in_group in group_labels.items():
        st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)
        st.sidebar.subheader(group)
        for feature in features_in_group:
            # Ensure each widget has a unique key
            widget_key = f"{group}_{feature}"

            # Display more descriptive labels
            if features[feature] == 'categorical':
                label = f"{feature.replace('_', ' ')}"
                input_df[feature] = st.sidebar.selectbox(label, label_encoders[feature].classes_, key=widget_key)
            else:
                label = f"{feature.replace('_', ' ')}"
                input_val = st.sidebar.text_input(label, key=widget_key)
                input_df[feature] = pd.to_numeric(input_val, errors='coerce')

# Additional styling for the overview section
st.markdown(
    """
    ### Welcome to Credit_Score prediction/credit healthy Tool!

    #### What You Can Do:
    1. Know the customers' properties proactively which helps decide loan default risks.
    2. Credit follow-up till to the credit collaction.
    3. Credit scoring (probability prediction) using Explainable AI to modify Existing the Risk grade.
    4. Based on the credit scoring (probability of prediction), each class may leads modify Existing the collateral.

    Dive into the rich data of  from 2014 to 2024, interact, and uncover valuable insights for decision making!
    """
)

# Display the input dataframe
st.write("Input Data (Before Encoding and Normalization):")
st.write(input_df)

# Make predictions using the loaded model
if st.sidebar.button("Predict"):
    # Apply label encoding to categorical features
    for feature, encoder in label_encoders.items():
        if feature != 'Credit_Score' and feature in input_df.columns:
            input_df[feature] = encoder.transform(input_df[feature])

    # Apply Min-Max scaling to numerical features
    for feature, scaler in minmax_scalers.items():
        if feature in input_df.columns:  # Check if feature exists in input_df
            try:
                # Perform scaling
                input_df[feature] = scaler.transform(input_df[feature].values.reshape(-1, 1))
            except ValueError as e:
                st.sidebar.write(f"Error scaling {feature}: {e}")
                # Optionally set to a default value if needed
                # input_df[feature] = np.nan  # or any default value you choose

    # Display the input data after encoding and normalization
    st.write("Input Data (After Encoding and Normalization):")
    st.write(input_df)

    # Prepare data for prediction
    model_features = loaded_model.get_booster().feature_names
    input_df_filtered = input_df[model_features]  # Only select model features

    # Make predictions
    prediction = loaded_model.predict(input_df_filtered)

    # Ensure prediction is a valid array with expected shape
    if isinstance(prediction, np.ndarray) and prediction.ndim > 0:
        # Assuming it's a classification problem, take the first prediction
        predicted_label = prediction[0]  # Get the first prediction

        # Output the prediction
        Stage = np.array(['Standard', 'Poor', 'Good'])  # Ensure correct labels based on your model
        prediction_index = int(predicted_label)  # Make sure this is correctly indexed

        # Output the prediction
        st.sidebar.write("Prediction:", Stage[prediction_index])

        # Show prediction probabilities if applicable
        if hasattr(loaded_model, 'predict_proba'):
            prediction_proba = loaded_model.predict_proba(input_df_filtered)
            st.subheader('Prediction (credit is healthy?)')
            st.write(f"Credit_Score: {Stage[prediction_index]}")

            st.subheader('Prediction Probability')
            probability_df = pd.DataFrame(prediction_proba, columns=Stage)
            st.write(probability_df)
    else:
        st.sidebar.write("Prediction could not be made.")
