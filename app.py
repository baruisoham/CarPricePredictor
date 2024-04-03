import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Load the cleaned data
car = pd.read_csv('Cleaned data.csv')

# Drop the 'Unnamed: 0' and 'Price' columns if present
car = car.drop(['Unnamed: 0', 'Price'], axis=1, errors='ignore')

# Sidebar for user input
st.sidebar.title("Car Price Prediction")

# Get user input
company = st.sidebar.selectbox("Select Company", sorted(car['company'].unique()))
car_model = st.sidebar.selectbox("Select Car Model", sorted(car[car['company'] == company]['name'].unique()))
year = st.sidebar.selectbox("Select Year", sorted(car['year'].unique(), reverse=True))
kms_driven = st.sidebar.number_input("Enter Kilometers Driven", min_value=0, step=1)
fuel_type = st.sidebar.selectbox("Select Fuel Type", car['fuel_type'].unique())

# Check if all inputs are provided
if company and car_model and year and kms_driven and fuel_type:
    # Create a DataFrame with the user input
    user_input = pd.DataFrame(data=[[car_model, company, year, kms_driven, fuel_type]],
                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    # Predict the car price
    try:
        prediction = model.predict(user_input)
        predicted_price = np.round(prediction[0], 2)
        st.write(f"Predicted Car Price: ₹ {predicted_price}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")




else:
    st.warning("Please fill in all the required fields.")