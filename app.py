import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
try:
    model = joblib.load('model.joblib')
except FileNotFoundError:
    st.error("Model file 'model.joblib' not found. Please run 'main.py' first to train the model.")
    st.stop()

def main():
    st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
    
    st.title("🏠 House Price Prediction App")
    st.write("Enter the property details below to get an estimated price.")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            square_footage = st.number_input("Square Footage (sq ft)", min_value=500, max_value=10000, value=2000, step=50)
            bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3, step=1)
            bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2, step=1)
            
        with col2:
            age = st.number_input("Age of Property (years)", min_value=0, max_value=100, value=10, step=1)
            location = st.selectbox("Location", ["Downtown", "Suburban", "Rural"])
            
        submit_button = st.form_submit_button("Predict Price")
        
    if submit_button:
        # Create a DataFrame for the input
        input_data = pd.DataFrame({
            'SquareFootage': [square_footage],
            'Bedrooms': [bedrooms],
            'Bathrooms': [bathrooms],
            'Age': [age],
            'Location': [location]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Convert USD to PKR (approximate rate)
        usd_to_pkr_rate = 278.0
        prediction_pkr = prediction * usd_to_pkr_rate
        
        # Display result
        st.success(f"Estimated Price: PKR {prediction_pkr:,.2f}")
        
        # specific logic for display based on location
        if location == "Downtown":
            st.info("Note: Downtown properties usually command a premium.")
        elif location == "Rural":
            st.info("Note: Rural properties might be more affordable per square foot.")

if __name__ == "__main__":
    main()
