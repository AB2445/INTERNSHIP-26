# Create a Streamlit web application for Titanic survival prediction

import streamlit as st
import pandas as pd
import joblib

# Load the trained model and feature columns
model = joblib.load("models/titanic_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# Display the application title
st.title("🚢 Titanic Survival Prediction")

st.write("Enter passenger details to predict survival.")

# Get passenger class from the user
pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

# Get gender from the user
sex = st.selectbox(
    "Gender",
    ["male", "female"]
)

# Get age from the user
age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=25.0
)

# Get number of siblings or spouses
sibsp = st.number_input(
    "Siblings/Spouses Aboard",
    min_value=0,
    max_value=10,
    value=0
)

# Get number of parents or children
parch = st.number_input(
    "Parents/Children Aboard",
    min_value=0,
    max_value=10,
    value=0
)

# Get ticket fare
fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=30.0
)

# Get port of embarkation
embarked = st.selectbox(
    "Port of Embarkation",
    ["S", "C", "Q"]
)

# Make prediction when the button is clicked
if st.button("Predict Survival"):

    # Create a DataFrame containing the passenger information
    passenger = pd.DataFrame({
        "Pclass": [pclass],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Sex_male": [1 if sex == "male" else 0],
        "Embarked_Q": [1 if embarked == "Q" else 0],
        "Embarked_S": [1 if embarked == "S" else 0]
    })

    # Arrange the input columns in the same order used during training
    passenger = passenger.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Predict passenger survival
    prediction = model.predict(passenger)[0]

    # Display the prediction result
    if prediction == 1:
        st.success("🚢 Passenger is predicted to SURVIVE!")
    else:
        st.error("❌ Passenger is predicted NOT to survive.")