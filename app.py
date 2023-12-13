import streamlit as st
from textblob import TextBlob
import plotly.graph_objects as go
import pandas as pd
import numpy as np



st.markdown("#WELCOME TO MY STREAMLIT APP!")


# Streamlit UI components
user_name = st.text_input("Enter your name", "")
st.write("Your name is", user_name)

# slider
age = st.slider('Please enter your age', 
                   min_value=0, max_value=100, value=10)
st.write("Your age is ", age)

income = st.slider('Please enter your income', min_value=5.0, max_value=9.0, value=7.0, step=0.1)
st.write("Your income is ", income)

education = st.slider('Please enter your education', min_value=4.0, max_value=9.0, value=7.0, step=0.1)
st.write("Your education level is ", education)



with st.form("user_form"):
    st.header("User Registration")
    gender = st.radio("Select your gender", ('Male', 'Female'))
    marital_status = st.radio("Select your marital status", ('Single', 'Married'))
    terms = st.checkbox("Accept terms and conditions")

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")

    if submitted:
        if terms:
            st.write("Marital Status:", marital_status)
        else:
            st.write("Please accept the terms and conditions.")


           

st.write("Thanks for visiting! Have a great day!")

