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


s = pd.read_csv ("social_media_usage.csv")

def clean_sm(x):
    x = np.where(x == 1, 1, 0)  
    return x

ss = s[['income', 'educ2', 'par', 'marital', 'gender', 'age', 'web1h']].copy()
# Handle the 'income' column as ordered numeric from 1 to 9, above 9 considered missing
ss['income'] = np.where((ss['income'] >= 1) & (ss['income'] <= 9), ss['income'], np.nan)
 
# Handle the 'education' column as ordered numeric from 1 to 8, above 8 considered missing
ss['educ2'] = np.where((ss['educ2'] >= 1) & (ss['educ2'] <= 8), ss['educ2'], np.nan)
 
# Apply clean_sm to the 'par' column
ss['par'] = np.where(ss['par'] == 1, 0, 1)
 
# Apply clean function to marital
ss['marital'] = clean_sm(s['marital'])
 
# Apply clean_sm to the 'gender' column
ss['gender'] = clean_sm(ss['gender'])
 
# Handle the 'age' column as numeric, above 98 considered missing
ss['age'] = np.where(ss['age'] <= 98, ss['age'], np.nan)
 
# Apply clean_sm to the target column
ss['sm_li'] = ss['web1h'].apply(clean_sm)

# Separate the target column 'sm_li' from the DataFrame
y = ss['sm_li']  # Target vector

# Selecting the features (excluding the target column 'sm_li')
X = ss.drop(['sm_li', 'web1h'], axis=1)  # Feature set


# Part 2
 
def load_model():

    # Load your trained model here

    # For example: model = joblib.load('your_model_file.pkl')

 return logistic_model
 
def load_scaler():
 return load_scaler
 
def predict_probability(features, scaler, model):

    # Standardize features

    features_scaled = scaler.transform(features)

    # Make prediction

    probability = model.predict_proba(features_scaled)[:, 1]

    return probability
 
def main():

    st.title("LinkedIn User Prediction App")
 
    st.markdown("""

            This app predicts the probability that a user is a LinkedIn member based on their 

            demographics and social attributes. It is trained using logistic regression to 

            estimate membership likelihood from parameters like income, education, age etc.

            """)
 
    # Sidebar with user input

    st.sidebar.header("User Input Features")

    income = st.sidebar.slider("Income", 1, 9, 5)

    education = st.sidebar.slider("Education", 1, 8, 4)

    parent = st.sidebar.radio("Parent", ["No", "Yes"])

    marital_status = st.sidebar.radio("Marital Status", ["Single", "Married"])

    gender = st.sidebar.radio("Gender", ["Male", "Female"])

    age = st.sidebar.slider("Age", 18, 98, 30)
 
    # Load the fitted scaler

    scaler = load_scaler()
 
    # Display the user input features

    st.write("## User Input Features")

    user_input = pd.DataFrame({'income': [income], 'educ2': [education], 'par': [1 if parent == "Yes" else 0],

                               'marital': [1 if marital_status == "Married" else 0],

                               'gender': [1 if gender == "Female" else 0], 'age': [age]})

    st.table(user_input)
 
    # Load the model and make predictions

    model = load_model()

    probability = predict_probability(user_input, scaler, model)
 
    # Display prediction results

    st.write("## Prediction")

    st.write(f"Probability of being a LinkedIn user: {probability[0]:.2f}")
 
    prediction = "LinkedIn User" if probability >= 0.5 else "Non-LinkedIn User"

    st.write(f"Prediction: {prediction}")
 
if __name__ == '__main__':

    main()

           

st.write("Thanks for visiting! Have a great day!")

