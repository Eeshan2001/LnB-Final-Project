import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=[
            "Iris Classification",
            "Heart Diasease Prediction",
            "Car Price Prediction",
            "House Price Prediction",
        ],
        icons=[
            "caret-right-fill",
            "caret-right-fill",
            "caret-right-fill",
            "caret-right-fill",
        ],
        default_index=0,
    )

if selected == "Iris Classification":
    modeliris = pickle.load(open("models/iris_model.pkl", "rb"))
    st.title("Iris Classification")
    sl = st.number_input("Sepal Length")
    sw = st.number_input("Sepal Width")
    pl = st.number_input("Petal Length")
    pw = st.number_input("Petal Width")
    X = np.array([sl, sw, pl, pw]).reshape(1, -1)
    if st.button("Predict"):
        prediction = modeliris.predict(X)
        st.success(prediction)

elif selected == "Heart Diasease Prediction":
    st.title("Heart Diasease Classification")
    modelheart = pickle.load(open("models/heart_disease_model.pkl", "rb"))
    age = st.number_input("Age")
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymtomatic"],
    )
    trestbps = st.number_input(
        "Resting Blood Pressure", min_value=94.0, max_value=200.0, step=0.5
    )
    chol = st.number_input(
        "Serum Cholesterol", min_value=126.0, max_value=564.0, step=0.5
    )
    fbs = st.selectbox(
        "Fasting Blood Sugar", ["Greater than 120 mg/dl", "Less than 120 mg/dl"]
    )
    restecg = st.selectbox(
        "Resting ECG Results",
        [
            "Normal",
            "Having ST-T wave abnormality",
            "Probable or definite left ventricular hypertrophy",
        ],
    )
    thalach = st.number_input(
        "Max Heart Rate", min_value=71.0, max_value=202.0, step=0.5
    )
    exang = st.selectbox("Exercise-induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("ST depression", min_value=0.0, max_value=6.2, step=0.2)
    slope = st.selectbox(
        "slope of the peak exercise ST segment", ["Upsloping", "Flat", "Downsloping"]
    )
    ca = st.number_input("Number of Major vessels", min_value=1, max_value=4, step=1)
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

    sex = 1 if sex == "Male" else 0
    if cp == "Typical Angina":
        cp = 0
    elif cp == "Atypical Angina":
        cp = 1
    elif cp == "Non-anginal Pain":
        cp = 2
    else:
        cp = 3
    fbs = 1 if fbs == "Greater than 120 mg/dl" else 0
    if restecg == "Normal":
        restecg = 0
    elif restecg == "Having ST-T wave abnormality":
        restecg = 1
    else:
        restecg = 2
    exang = 1 if exang == "Yes" else 0
    if thal == "Normal":
        thal = 0
    elif thal == "Fixed Defect":
        thal = 1
    else:
        thal = 2

    if slope == "Upsloping":
        slope = 0
    elif slope == "Flat":
        slope = 1
    else:
        slope = 2

    data = np.array(
        [
            [
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal,
            ]
        ]
    )
    if st.button("Predict"):
        prediction = modelheart.predict(data)
        st.success(prediction)

elif selected == "Car Price Prediction":
    st.title("Car Price Prediction")
    modelcar = pickle.load(open("models/car_price_model.pkl", "rb"))

else:
    st.title("House Price Prediction")
    modelhouse = pickle.load(open("models/house_predict_model.pkl", "rb"))
