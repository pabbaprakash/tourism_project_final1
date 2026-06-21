import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# CHANGE THIS: your HF model repo
model_path = hf_hub_download(
    repo_id="prakashpabba/tourism-model",
    filename="best_tourism_model_v1.joblib",
)
model = joblib.load(model_path)

st.title("Tourism Package Purchase Prediction")
st.write(
    """
Predict whether a customer will purchase the Wellness Tourism Package
based on their profile and interaction data.
"""
)

age = st.number_input("Age", min_value=18, max_value=80, value=35)
type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
occupation = st.selectbox(
    "Occupation", ["Salaried", "Freelancer", "Business", "Student", "Other"]
)
gender = st.selectbox("Gender", ["Male", "Female"])
num_person_visiting = st.number_input("Number of Persons Visiting", 1, 10, 2)
preferred_property_star = st.number_input("Preferred Property Star", 1, 5, 3)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
num_trips = st.number_input("Number of Trips per year", 0, 20, 2)
passport = st.selectbox("Has Passport", ["No", "Yes"])
own_car = st.selectbox("Owns Car", ["No", "Yes"])
num_children_visiting = st.number_input("Number of Children Visiting (<5 yrs)", 0, 5, 0)
designation = st.selectbox(
    "Designation", ["Entry Level", "Mid Level", "Senior Level", "Manager", "Executive"]
)
monthly_income = st.number_input("Monthly Income", 10000, 500000, 50000)
pitch_satisfaction_score = st.number_input(
    "Pitch Satisfaction Score", 1.0, 5.0, 3.5, step=0.1
)
product_pitched = st.selectbox(
    "Product Pitched",
    ["Wellness Package", "Adventure Package", "Luxury Package", "Family Package"],
)
num_followups = st.number_input("Number of Followups", 0, 10, 2)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", 5, 60, 20)

type_of_contact_map = {"Company Invited": 0, "Self Inquiry": 1}
occupation_map = {"Salaried": 0, "Freelancer": 1, "Business": 2, "Student": 3, "Other": 4}
gender_map = {"Male": 0, "Female": 1}
marital_status_map = {"Single": 0, "Married": 1, "Divorced": 2}
designation_map = {
    "Entry Level": 0,
    "Mid Level": 1,
    "Senior Level": 2,
    "Manager": 3,
    "Executive": 4,
}
passport_map = {"No": 0, "Yes": 1}
own_car_map = {"No": 0, "Yes": 1}
product_pitched_map = {
    "Wellness Package": 0,
    "Adventure Package": 1,
    "Luxury Package": 2,
    "Family Package": 3,
}

input_data = pd.DataFrame(
    [
        {
            "Age": age,
            "TypeofContact": type_of_contact_map[type_of_contact],
            "CityTier": city_tier,
            "Occupation": occupation_map[occupation],
            "Gender": gender_map[gender],
            "NumberOfPersonVisiting": num_person_visiting,
            "PreferredPropertyStar": preferred_property_star,
            "MaritalStatus": marital_status_map[marital_status],
            "NumberOfTrips": num_trips,
            "Passport": passport_map[passport],
            "OwnCar": own_car_map[own_car],
            "NumberOfChildrenVisiting": num_children_visiting,
            "Designation": designation_map[designation],
            "MonthlyIncome": monthly_income,
            "PitchSatisfactionScore": pitch_satisfaction_score,
            "ProductPitched": product_pitched_map[product_pitched],
            "NumberOfFollowups": num_followups,
            "DurationOfPitch": duration_of_pitch,
        }
    ]
)

if st.button("Predict Purchase"):
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]
    label = "Will Purchase" if pred == 1 else "Will Not Purchase"

    st.subheader("Prediction Result")
    st.write(f"Prediction: **{label}**")
    st.write(f"Purchase probability: **{proba:.2%}**")
