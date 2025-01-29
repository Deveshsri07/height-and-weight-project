import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('height_weight_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Height-Weight Prediction App")
st.write("Enter your height to predict your weight.")

# User input
height = st.number_input("Enter your height (in cm)", min_value=100, max_value=250, value=170)

# Prediction
if st.button("Predict Weight"):
    weight_pred = model.predict(np.array(height).reshape(-1, 1))[0]
    st.success(f"Predicted Weight: {weight_pred:.2f} kg")
