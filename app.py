import streamlit as st
import json
from datetime import date
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
    sl = st.number_input("Sepal Length", min_value=4.3, max_value=7.9, step=0.1)
    sw = st.number_input("Sepal Width", min_value=3.1, max_value=4.4, step=0.1)
    pl = st.number_input("Petal Length", min_value=1.2, max_value=6.9, step=0.1)
    pw = st.number_input("Petal Width", min_value=0.1, max_value=2.5, step=0.1)
    X = np.array([sl, sw, pl, pw]).reshape(1, -1)
    if st.button("Predict"):
        prediction = modeliris.predict(X)
        st.success(f"Classified Iris Species is {prediction[0]}")

elif selected == "Heart Diasease Prediction":
    st.title("Heart Diasease Classification")
    modelheart = pickle.load(open("models/heart_disease_model.pkl", "rb"))
    age = st.number_input("Age", min_value=18, max_value=80)
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
        if prediction == 1:
            st.success("Sorry, You have Heart Diasease 😞")
        else:
            st.success("Hey, You don't have Heart Diasease 😎")

elif selected == "Car Price Prediction":
    st.title("Car Price Prediction")
    modelcar = pickle.load(open("models/car_price_model.pkl", "rb"))
    Year = st.number_input("Year", min_value=2005, max_value=2020, step=1)
    Present_Price = st.number_input(
        "Present Price (lakhs)", min_value=5.5, max_value=95.6, step=0.2
    )
    Kms_Driven = st.number_input(
        "Kilometers Driven", min_value=500, max_value=20000, step=100
    )
    Owner = st.number_input("No of Owner", min_value=0, max_value=4, step=1)
    Fuel_Type_Petrol = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    Seller_Type_Individual = st.selectbox("Seller Type", ["Individual", "Dealer"])
    Transmission_Manual = st.selectbox("Transmission Type", ["Manual", "Automatic Car"])
    if Fuel_Type_Petrol == "Petrol":
        Fuel_Type_Petrol = 1
        Fuel_Type_Diesel = 0
    else:
        Fuel_Type_Petrol = 0
        Fuel_Type_Diesel = 1
    Year = date.today().year - Year

    Seller_Type_Individual = 1 if Seller_Type_Individual == "Individual" else 0
    Transmission_Manual = 1 if Transmission_Manual == "Manual" else 0
    prediction = np.array(
        [
            [
                Year,
                Present_Price,
                Kms_Driven,
                Owner,
                Fuel_Type_Diesel,
                Fuel_Type_Petrol,
                Seller_Type_Individual,
                Transmission_Manual,
            ]
        ]
    )
    if st.button("Predict"):
        prediction = np.round(modelcar.predict(prediction), 2)
        st.success(f"You sell the car for {prediction[0]} lakhs")


else:
    st.title("House Price Prediction")
    modelhouse = pickle.load(open("models/house_predict_model.pkl", "rb"))
    with open("columns.json", "r") as f:
        columns = json.load(f)["data_columns"]
    location = st.selectbox("Location", columns[8:-1])
    area_type = st.selectbox("Area Type", columns[4:8])
    bhk = st.number_input("No of bedrooms", min_value=1, max_value=5)
    total_sqft = st.number_input("Total Square feet", min_value=10000, max_value=500000)
    bath = st.number_input("No of bathroom", min_value=1, max_value=5)
    balcony = st.number_input("No of Balcony", min_value=1, max_value=5)

    area_type_ind = columns.index(area_type)
    location_ind = columns.index(location)
    data = np.zeros(len(columns))
    data[0] = bhk
    data[1] = total_sqft
    data[2] = bath
    data[3] = balcony
    data[area_type_ind] = 1
    data[location_ind] = 1

    if st.button("Predict"):
        prediction = np.round(modelhouse.predict([data])[0], 2)
        st.success(f"You sell the house for {prediction} lakhs")
